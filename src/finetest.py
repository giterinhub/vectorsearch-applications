from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('/workspaces/vectorsearch-applications/notebooks/models/') # allminilm-finetuned-256
embeddings = model.encode(sentences)
print(embeddings)