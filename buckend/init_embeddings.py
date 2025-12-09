import os
import json
import faiss
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

DATA_DIR = "data"
OUTPUT_INDEX = "index.faiss"
OUTPUT_META = "meta.json"

EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent?key=" 
    + API_KEY
)


def embed_text(text: str):
    payload = {
        "model": "text-embedding-004",
        "content": {"parts": [{"text": text}]},
    }

    res = requests.post(EMBED_URL, json=payload)
    data = res.json()

    # デバッグ用に表示（必要なら残してOK）
    print("\n--- EMBEDDING RESPONSE ---")
    print(data)
    print("---------------------------\n")

    # 正しいキーは "values"
    try:
        vec = data["embedding"]["values"]
        return np.array(vec).astype("float32")
    except KeyError:
        print("Embedding Error:", data)
        raise ValueError("Embedding failed. Check the API response.")




def main():
    files = os.listdir(DATA_DIR)
    vectors = []
    metas = []

    for fname in files:
        path = os.path.join(DATA_DIR, fname)

        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except:
            continue

        print(f"Embedding {fname}...")
        vec = embed_text(text)
        vectors.append(vec)
        metas.append({"source": fname})

    # FAISS index 作成
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    faiss.write_index(index, OUTPUT_INDEX)

    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print("FAISS index & meta.json created!")


if __name__ == "__main__":
    main()
