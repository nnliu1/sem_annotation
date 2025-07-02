from recom_sys import OntologyRecommender
import openai


client = openai.OpenAI(api_key="your-api-key") # please input your openai api key

output_json = "results/matches.json"
rec = OntologyRecommender(
    ontology_emb_path="data/output/oeo_emb.npy",
    ontology_info_path="data/output/ontology_terms_oeo.json"
)

## if you want to generate gpt description, please use this code

#df = rec.prepare_text_with_gpt(
#    input_file="dataset_metadata.xlsx",
#    output_text_file="text_prepared.csv",
#    client=client
#)

df = pd.read_csv("data/output/text_prepared.csv")
recs = rec.recommend(df_texts=df, top_k=10)
rec.save_recommendations(recs, output_json)