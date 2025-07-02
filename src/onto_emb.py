import os
import json
import numpy as np
import torch
from torch import device as torch_device
from rdflib import Graph, RDF, RDFS, OWL, Namespace
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class OntologyEmbedder:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", sep="[SEP]", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.sep = sep
        self.device = torch_device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model.to(self.device)
        self.IAO = Namespace("http://purl.obolibrary.org/obo/IAO_")

    def load_ontology(self, file_path: str) -> Graph:
        print(f"Loading ontology from: {file_path}")
        g = Graph()
        g.parse(file_path, format="xml")
        return g

    def extract_terms(self, g: Graph) -> list:
        terms = []
        for s in tqdm(g.subjects(), desc="Extracting terms"):
            types = list(g.objects(s, RDF.type))
            if OWL.Class in types or OWL.NamedIndividual in types:
                label = g.value(s, RDFS.label)
                comment = g.value(s, RDFS.comment)
                definition = g.value(s, self.IAO["0000115"])
                terms.append({
                    "uri": str(s),
                    "label": str(label or ""),
                    "definition": str(definition or comment or ""),
                    "type": "class" if OWL.Class in types else "individual"
                })
        return terms

    def get_wordnet_synonyms(self, word: str) -> list:
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace("_", " ").lower()
                if lemma_name != word.lower():
                    synonyms.add(lemma_name)
        return list(synonyms)

    def build_term_texts(self, terms: list) -> list:
        texts = []
        for term in terms:
            label = term["label"].strip()
            definition = term["definition"].strip()
            synonyms = self.get_wordnet_synonyms(label)
            syn_text = "; ".join(synonyms)
            combined = f"{label}{self.sep}{syn_text}{self.sep}{definition}".strip(self.sep)
            texts.append(combined)
        return texts

    def deduplicate_terms(self, terms: list) -> list:
        seen = set()
        unique_terms = []
        for term in terms:
            key = f"{term['type'].lower()}{self.sep}{term['label'].strip()}{self.sep}{term['definition'].strip()}"
            if key not in seen:
                seen.add(key)
                unique_terms.append(term)
        print(f"Removed duplicates: {len(terms) - len(unique_terms)}")
        return unique_terms

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

    def encode_texts(self, texts: list, batch_size=32) -> np.ndarray:
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with SBERT"):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs)
                embeddings = self.mean_pooling(outputs, inputs["attention_mask"]).cpu()
                all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0).numpy()

    def save_outputs(self, terms, embeddings, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "oeo_emb.npy"), embeddings)
        np.save(os.path.join(out_dir, "concept_ids_oeo.npy"), [t["uri"] for t in terms])
        with open(os.path.join(out_dir, "ontology_terms_oeo.json"), "w", encoding="utf-8") as f:
            json.dump(terms, f, indent=2, ensure_ascii=False)

    def run_pipeline(self, ontology_path: str, output_dir: str = "data/output"):
        g = self.load_ontology(ontology_path)
        terms = self.extract_terms(g)
        terms = self.deduplicate_terms(terms)
        texts = self.build_term_texts(terms)
        embeddings = self.encode_texts(texts)
        self.save_outputs(terms, embeddings, output_dir)
        print("Embedding pipeline complete.")


if __name__ == "__main__":
    embedder = OntologyEmbedder()
    embedder.run_pipeline("data/oeo.owl", "data/output")
