from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import requests
import json
import faiss
import numpy as np
from dotenv import load_dotenv

# --- パス設定 ---
BUCKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BUCKEND_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# --- 環境変数 ---
load_dotenv(os.path.join(BUCKEND_DIR, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- FastAPI ---
app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FAISS / metadata 読み込み ---
INDEX_PATH = os.path.join(BUCKEND_DIR, "index.faiss")
META_PATH = os.path.join(BUCKEND_DIR, "meta.json")
DATA_DIR = os.path.join(BUCKEND_DIR, "data")

if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        metas = json.load(f)
else:
    index = None
    metas = []
    print("Warning: index.faiss or meta.json not found.")


# --- Gemini Embedding API ---
EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent?key="
    + GEMINI_API_KEY
)

def embed_text(text: str):
    """Gemini API で埋め込み生成（超軽量）"""
    payload = {
        "model": "text-embedding-004",
        "content": {"parts": [{"text": text}]},
    }

    res = requests.post(EMBED_URL, json=payload)
    data = res.json()

    # 正しいキーは "values"
    try:
        vec = data["embedding"]["values"]
        return np.array(vec).astype("float32")
    except KeyError:
        print("Embedding Error in app.py:", data)
        # 検索時に完全に落ちないように、とりあえずゼロベクトル返す
        return np.zeros((768,), dtype="float32")



# --- ベクトル検索 ---
def retrieve(query, top_k=3):
    if index is None:
        return ""

    vec = embed_text(query).reshape(1, -1)
    D, I = index.search(vec, top_k)

    results = []
    for idx in I[0]:
        if 0 <= idx < len(metas):
            fname = metas[idx]["source"]
            try:
                with open(os.path.join(DATA_DIR, fname), encoding="utf-8") as f:
                    content = f.read()
                results.append(f"[{fname}]\n{content}")
            except:
                continue

    return "\n\n".join(results)


# --- リクエストモデル ---
class ChatRequest(BaseModel):
    message: str


# --- ルート（index.html） ---
@app.get("/")
def home():
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"error": "index.html not found"}


# --- 画像提供 ---
@app.get("/akiyama.jpg")
def get_image():
    img_path = os.path.join(FRONTEND_DIR, "akiyama.jpg")
    if os.path.exists(img_path):
        return FileResponse(img_path)
    return {"error": "Image not found"}


# --- チャットAPI ---
@app.post("/chat")
def chat(req: ChatRequest):
    query = req.message
    context = retrieve(query, top_k=5)

    prompt = f"""
あなたはスマートICTソリューション研究室の教授「秋山康智（あきやま こうじ）」です。
以下の参考情報を元に、学生にわかりやすく日本語で回答してください。

#会話ルール
- 敬語で話してください。
- 重要な単語は ** で囲んで強調してください（フロントエンドで太字になります）。
- 文脈を維持し、自然に会話してください。
- 挨拶＋名乗りは、ユーザーが挨拶してきた時だけ行う。

#回答ルール
- 研究室データ（参考情報）を優先。
- 不明な点は「不明です」と答える。
- 専門用語は簡単に説明。
- 本文は 300 字程度にまとめる。

#重要：学生が次に聞きそうな質問を3つ提案する
回答本文と質問候補は「---」で区切る。

#参考情報:
{context}

#質問:
{query}
"""

    # Gemini Chat API
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        res = requests.post(url, json=payload, timeout=20)

        if res.status_code != 200:
            print("Gemini API Error:", res.status_code, res.text)
            return {"answer": "AIサーバーとの通信でエラーが発生しました。"}

        data = res.json()
        answer = data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        print("Server Error:", e)
        return {"answer": "エラーが発生しました（サーバーログを確認してください）。"}

    return {"answer": answer}
