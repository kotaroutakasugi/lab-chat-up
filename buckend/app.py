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
from functools import lru_cache

# --- パス設定 ---
BUCKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BUCKEND_DIR)
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# --- 環境変数 ---
load_dotenv(os.path.join(BUCKEND_DIR, ".env"))  # ローカル用
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- FastAPI 本体 ---
app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FAISS / メタデータ読み込み ---
INDEX_PATH = os.path.join(BUCKEND_DIR, "index.faiss")
META_PATH = os.path.join(BUCKEND_DIR, "meta.json")
DATA_DIR = os.path.join(BUCKEND_DIR, "data")

if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        metas = json.load(f)
    EMBED_DIM = index.d  # 埋め込み次元（ゼロベクトル用に保持）
else:
    index = None
    metas = []
    EMBED_DIM = 768  # 一応のデフォルト
    print("Warning: index.faiss or meta.json not found.")


# --- Gemini Embedding API 設定 ---
EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent"
    f"?key={GEMINI_API_KEY}"
)

@lru_cache(maxsize=512)
def embed_text_cached(text: str) -> np.ndarray:
    """
    質問文などの埋め込みを Gemini で取得。
    lru_cache でキャッシュして、同じ質問なら API を再呼び出ししない。
    """
    if not GEMINI_API_KEY:
        print("No GEMINI_API_KEY set for embeddings.")
        return np.zeros((EMBED_DIM,), dtype="float32")

    payload = {
        "model": "text-embedding-004",
        "content": {"parts": [{"text": text}]},
    }

    try:
        res = requests.post(EMBED_URL, json=payload, timeout=15)
        data = res.json()

        # 正常系：embedding.values
        if "embedding" in data and "values" in data["embedding"]:
            vec = data["embedding"]["values"]
            return np.array(vec, dtype="float32")

        # エラー系：ログに内容を出しておく
        print("Embedding API error response:", data)
        return np.zeros((EMBED_DIM,), dtype="float32")

    except Exception as e:
        print("Embedding API call failed:", e)
        return np.zeros((EMBED_DIM,), dtype="float32")


# --- ベクトル検索（RAG） ---
def retrieve(query: str, top_k: int = 3) -> str:
    """質問文から近いテキストを FAISS で検索"""
    if index is None:
        return ""

    vec = embed_text_cached(query).reshape(1, -1)
    D, I = index.search(vec, top_k)

    results = []
    for idx in I[0]:
        if 0 <= idx < len(metas):
            fname = metas[idx]["source"]
            path = os.path.join(DATA_DIR, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                results.append(f"[{fname}]\n{content}")
            except Exception as e:
                print(f"Error reading {fname}:", e)
                continue

    return "\n\n".join(results)


# --- リクエストモデル ---
class ChatRequest(BaseModel):
    message: str


# --- ルート（フロント配信） ---
@app.get("/")
def home():
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"error": "index.html not found"}


@app.get("/akiyama.jpg")
def get_image():
    img_path = os.path.join(FRONTEND_DIR, "akiyama.jpg")
    if os.path.exists(img_path):
        return FileResponse(img_path)
    return {"error": "Image not found"}


# --- Gemini 本文生成呼び出し ---
def call_gemini_chat(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY が設定されていないため、回答を生成できません。"

    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        res = requests.post(url, json=payload, timeout=20)

        # レート制限・課金関連のエラーを丁寧に処理
        if res.status_code == 429:
            print("Gemini API Error 429:", res.text)
            return (
                "現在、Gemini API の利用上限に達しています。\n"
                "しばらく時間をおいてから、もう一度お試しください。"
            )

        if res.status_code != 200:
            print("Gemini API Error:", res.status_code, res.text)
            return (
                "AIサーバーとの通信でエラーが発生しました。\n"
                "時間をおいて再度お試しください。"
            )

        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        print("Gemini chat call failed:", e)
        return "エラーが発生しました（サーバーログを確認してください）。"


# --- チャットAPI ---
@app.post("/chat")
def chat(req: ChatRequest):
    user_message = req.message
    context = retrieve(user_message, top_k=5)

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
{user_message}
"""

    answer = call_gemini_chat(prompt)
    return {"answer": answer}
