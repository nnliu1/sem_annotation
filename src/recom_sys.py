import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel


class OntologyRecommender:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        ontology_emb_path: str = "",
        ontology_info_path: str = "",
        device: str = None,
        top_k: int = 10
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.top_k = top_k

        self.ontology_emb = self.load_ontology_embeddings(ontology_emb_path)
        self.ontology_info = self.load_ontology_info(ontology_info_path)
        self.ontology_terms = list(self.ontology_info.keys())

    def load_ontology_embeddings(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ontology embedding file not found: {path}")
        emb = np.load(path)
        return F.normalize(torch.tensor(emb, dtype=torch.float32).to(self.device), p=2, dim=1)

    def load_ontology_info(self, path: str) -> Dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ontology info file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            return data
        
        elif isinstance(data, list):
            info_dict = {}
            for item in data:
                uri = item.get("uri")
                if uri:
                    info_dict[uri] = {
                        "label": item.get("label", "N/A"),
                        "comment": item.get("comment", "")
                    }
            return info_dict
        else:
            raise ValueError("Unsupported ontology info format (expected dict or list).")

    def generate_description_gpt(self, row: pd.Series, client, model="gpt-4o", max_tokens=100) -> str:
        prompt = f"""You are a scientific assistant. Based on the metadata below, generate one
concise but informative sentence to describe a dataset column. If a
unit symbol is given, convert it into natural language (for example,
kWh to kilowatt hours) and explain it in context:
- Column name: {row.get("name", "")}
- Description: {row.get("Description", "")}
- Unit: {row.get("unit", "")}
- Data type: {row.get("type", "")}"""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for semantic data annotation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] GPT generation failed for '{name}': {e}")
            return f"{row.get('name', '')}: {row.get('Description', '')}"

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        embeddings = []
        with torch.no_grad():
            for text in texts:
                encoded = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
                output = self.model(**encoded)
                cls_emb = output.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(cls_emb.cpu())
        return F.normalize(torch.stack(embeddings).to(self.device), p=2, dim=1)

    def prepare_text_with_gpt(
        self,
        input_file: str,
        output_text_file: str,
        client,
        model="gpt-4o"
    ) -> pd.DataFrame:
        if client is None:
            raise ValueError(" GPT client must be provided.")

        if input_file.endswith((".xls", ".xlsx")):
            df = pd.read_excel(input_file)
        elif input_file.endswith(".csv"):
            df = pd.read_csv(input_file)
        else:
            raise ValueError("Only Excel or CSV supported.")
        
        df["text"] = df.apply(lambda row: self.generate_description_gpt(row, client, model=model), axis=1)
        df.to_csv(output_text_file, index=False)
        print(f" GPT descriptions saved to: {output_text_file}")
        return df

    def get_bert_embeddings(self, texts: List[str]) -> torch.Tensor:
        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(cls_emb.cpu())

        stacked = torch.stack(embeddings).to(self.device)
        return F.normalize(stacked, p=2, dim=1)

    def weighted_fusion(self, emb1: torch.Tensor, emb2: torch.Tensor, alpha: float = 0.6) -> torch.Tensor:
        if emb1.shape != emb2.shape:
            raise ValueError("Embedding shapes must match for fusion.")
        return alpha * emb1 + (1 - alpha) * emb2
        
    def recommend(self, df_texts: pd.DataFrame, top_k: int = 10, alpha: float = 0.7) -> List[Dict]:
        if "text" not in df_texts.columns:
            raise ValueError("Missing 'text' column in input dataframe")

        label = df_texts["Description"].astype(str).fillna("").tolist()
        text = df_texts["text"].astype(str).fillna("").tolist()

        label_emb = self.get_bert_embeddings(label)
        text_emb = self.get_bert_embeddings(text)
        query_embeddings = self.weighted_fusion(label_emb, text_emb, alpha)
        
        #query_embeddings = self.get_bert_embeddings(df_texts["text"].tolist())
        similarity = torch.matmul(query_embeddings, self.ontology_emb.T)
        topk_scores, topk_indices = torch.topk(similarity, k=self.top_k, dim=1)

        all_recommendations = []
        for i, row in df_texts.iterrows():
            recs = []
            for j in range(self.top_k):
                idx = topk_indices[i][j].item()
                score = topk_scores[i][j].item()
                term_uri = self.ontology_terms[idx]
                info = self.ontology_info[term_uri]
                recs.append({
                    "rank": j + 1,
                    "label": info.get("label", "N/A"),
                    "uri": term_uri,
                    "score": round(score, 4),
                    "comment": info.get("comment", "")
                })
            all_recommendations.append({
                "text": row["text"],
                "recommendations": recs
            })
        return all_recommendations

    def save_recommendations(self, recommendations: List[Dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recommendations, f, ensure_ascii=False, indent=2)

