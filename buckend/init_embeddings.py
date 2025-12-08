import os
import json
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = "data"
INDEX_FILE = "index.faiss"
META_FILE = "meta.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
metas = []

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".txt") or fname.endswith(".md"):
        path = os.path.join(DATA_DIR, fname)
        with open(path, encoding="utf-8") as f:
            content = f.read().split("\n\n")  # 段落ごとに分割
            for paragraph in content:
                paragraph = paragraph.strip()
                if paragraph:
                    documents.append(paragraph)
                    metas.append({"source": fname})

print(f"Encoding {len(documents)} documents...")
embeddings = model.encode(documents, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)

print("✅ index.faiss と meta.json を生成しました。")
