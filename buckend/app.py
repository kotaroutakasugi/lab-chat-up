from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import requests
import json
from dotenv import load_dotenv
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# --- 0. パス設定 ---
# この app.py がある backend フォルダ
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# プロジェクトルート lab-chatbot-up
ROOT_DIR = os.path.dirname(BACKEND_DIR)
# frontend フォルダ
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# --- 1. 環境変数読み込み ---
# ローカル開発用（backend/.env を読む）。Render では無視される
load_dotenv(os.path.join(BACKEND_DIR, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# --- 2. CORS設定 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. RAG準備 ---
INDEX_FILE = os.path.join(BACKEND_DIR, "index.faiss")
META_FILE = os.path.join(BACKEND_DIR, "meta.json")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, encoding="utf-8") as f:
        metas = json.load(f)
else:
    index = None
    metas = []
    print("Warning: index.faiss or meta.json not found.")

model = SentenceTransformer("all-MiniLM-L6-v2")


class ChatRequest(BaseModel):
    message: str


def retrieve(query, top_k=3):
    if index is None:
        return ""
    vec = model.encode([query])
    D, I = index.search(vec, top_k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(metas):
            fname = metas[idx]["source"]
            try:
                with open(os.path.join(DATA_DIR, fname), encoding="utf-8") as f:
                    content = f.read()
                results.append(f"[{fname}]\n{content}")
            except Exception as e:
                print(f"Error reading {fname}: {e}")
    return "\n\n".join(results)


# --- 4. ルーティング設定 ---

@app.get("/")
def home():
    """frontend/index.html を返す"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found"}


@app.get("/akiyama.jpg")
def get_image():
    """frontend/akiyama.jpg を返す"""
    img_path = os.path.join(FRONTEND_DIR, "akiyama.jpg")
    if os.path.exists(img_path):
        return FileResponse(img_path)
    else:
        return {"error": "Image not found"}


@app.post("/chat")
def chat(req: ChatRequest):
    user_message = req.message
    context = retrieve(user_message, top_k=5)

    # --- Geminiへのプロンプト ---
    prompt = f"""
あなたはスマートICTソリューション研究室の教授「秋山康智（あきやま こうじ）」です。
以下の参考情報を元に、学生にわかりやすく日本語で回答してください。

#会話ルール
- 敬語で話してください。
- 重要な単語は ** で囲んで強調してください（フロントエンドで太字に変換されます）。
- 文脈を維持し、自然に会話してください。
- 「挨拶＋名乗り＋何が気になるかを聞く」形式の挨拶は、ユーザーが挨拶したときだけ行ってください。それ以外は不要です。

#回答ルール
- 研究室データ（参考情報）を優先的に使用してください。
- 推測で誤った情報を話さないでください。不明な点は「不明です」と答えてください。
- 専門用語には簡単な説明を加えてください。
- 回答は300字程度にまとめてください。

#重要：質問候補の提案
回答の最後に、その回答に関連して学生が次に聞きそうな「おすすめの質問」を3つ提案してください。
回答本文とおすすめの質問は「---」という区切り線で分け、それぞて改行して書いてください。

形式例:
回答本文です。ここまでは普通に答えてください。
---
研究室の場所は？
コアタイムはありますか？
プログラミング言語は何を使いますか？

#参考情報:
{context}

#質問:
{user_message}
"""

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    )

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)

        if response.status_code != 200:
            print(f"Gemini API Error: {response.status_code} - {response.text}")
            answer = "申し訳ありません。AIサーバーとの通信でエラーが発生しました。"
        else:
            data = response.json()
            answer = data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        answer = "エラーが発生しました（サーバーログを確認してください）"
        print(f"Server Error Details: {e}")

    return {"answer": answer}
