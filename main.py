from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import json
import faiss
import numpy as np


# Load your VAT guide or financial doc
loader = PyPDFLoader("data/how_vat_works.pdf")
pages = loader.load_and_split()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(pages)
with open ("chunks_with_meta.json", "w") as f:
    json.dump([
        {"text": doc.page_content, "metadata": doc.metadata}
         for doc in chunks
    ], f)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc.page_content for doc in chunks]
embeddings = embedder.encode(texts, convert_to_numpy=True)
metadata = [doc.metadata for doc in chunks]

# Convert to NumPy array if not already
embedding_array = np.array(embeddings)
# Create FAISS index
index = faiss.IndexFlatL2(embedding_array.shape[1])
index.add(embedding_array)
# Save to disk (optional)
faiss.write_index(index, "vat_index.faiss")

query = "Do I need to register for VAT?"
query_embedding = embedder.encode([query], convert_to_numpy=True)
D, I = index.search(np.array(query_embedding), k=5)
retrieved_chunks = [texts[i] for i in I[0]]

context = "\n".join(retrieved_chunks)
final_prompt = f"""Context:
{context}

Question:
{query}
"""
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

response = generator(final_prompt, max_new_tokens=300)
print(response[0]["generated_text"])


