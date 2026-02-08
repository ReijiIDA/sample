
# LLMapi_openrouter.py
import os
import json
import requests
import time
from typing import Optional, Dict, Any

# ===== OpenRouter 設定 =====
# 環境変数: export OPENROUTER_API_KEY=...
DEFAULT_API_TOKEN = os.getenv("OPENROUTER_API_KEY")

# モデル: お好みで。自動選択なら "openrouter/auto"。
# 具体例: "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "qwen/qwen2.5-7b-instruct"
DEFAULT_MODEL = "openai/gpt-oss-120b:free"

# エンドポイント: OpenAI 互換 /chat/completions
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def default_headers(token: str,
                    referer: Optional[str] = None,
                    title: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    # OpenRouter 推奨（任意・ダッシュボードでの識別に便利）
    if referer:
        headers["HTTP-Referer"] = referer  # e.g., "https://your-app.example.com"
    if title:
        headers["X-Title"] = title         # e.g., "My RL Reward Designer"
    return headers


def call_llm(
    user_content: str,
    system_instruction: str = "あなたは強化学習の専門家です。報酬設計をしてください。",
    *,
    model: Optional[str] = None,
    api_token: Optional[str] = None,
    api_url: Optional[str] = None,
    max_tokens: int = 800,
    temperature: float = 0.7,
    top_p: float = 0.95,
    extra_payload: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
    http_referer: Optional[str] = None,
    x_title: Optional[str] = None,
    # ▼ 追加：デバッグと簡易リトライ
    debug: bool = False,
    retries: int = 2,
    backoff_sec: float = 1.0,
) -> str:
    """
    OpenRouter でチャット補完を行うラッパ。
    成功時: モデル出力テキスト（str）を返す。
    失敗時: RuntimeError を送出。
    """
    token = api_token or DEFAULT_API_TOKEN
    if not token:
        raise RuntimeError("環境変数 OPENROUTER_API_KEY が未設定です。api_token で渡すか、環境変数を設定してください。")

    url = api_url or DEFAULT_API_URL
    model_name = model or DEFAULT_MODEL

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        # 必要に応じて "stream": True にし、SSE受信処理を実装
    }
    if extra_payload:
        payload.update(extra_payload)

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                url,
                headers=default_headers(token, referer=http_referer, title=x_title),
                data=json.dumps(payload),
                timeout=timeout,
            )
            if resp.status_code != 200:
                # 429/5xx はリトライ
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    if debug:
                        print(f"[DEBUG] retryable status={resp.status_code}, attempt={attempt+1}")
                    time.sleep(backoff_sec * (2 ** attempt))
                    continue
                raise RuntimeError(f"APIエラー (status={resp.status_code}): {resp.text}")

            try:
                data = resp.json()
            except ValueError as e:
                raise RuntimeError(f"レスポンスのJSON解析に失敗しました: {e}\nraw={resp.text[:500]}")

            if debug:
                print("[DEBUG] raw response:", json.dumps(data, ensure_ascii=False)[:3000])

            # ===== 出力取り出し（フォールバック付き）=====
            msg = None
            try:
                msg = data["choices"][0]["message"]
            except Exception:
                msg = None

            content: Optional[str] = None
            if isinstance(msg, dict):
                # 1) 通常の content
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    content = c
                else:
                    # 2) content が空/None → reasoning をフォールバック
                    r = msg.get("reasoning")
                    if isinstance(r, str) and r.strip():
                        content = r
                    else:
                        # 3) 拒否理由の可能性
                        ref = msg.get("refusal")
                        if isinstance(ref, str) and ref.strip():
                            content = ref

            # 4) それでも空なら生JSONを返す（デバッグ用）
            if not content or not content.strip():
                content = json.dumps(data, ensure_ascii=False)

            return content

        except (requests.exceptions.RequestException, RuntimeError) as e:
            last_err = e
            if attempt < retries:
                if debug:
                    print(f"[DEBUG] error on attempt {attempt+1}/{retries+1}: {e}")
                time.sleep(backoff_sec * (2 ** attempt))
                continue
            raise RuntimeError(f"HTTP/API呼び出しに失敗しました: {e}") from e

    # ここには来ない想定
    raise RuntimeError(f"不明なエラー: {last_err}")


def main():
    token = DEFAULT_API_TOKEN
    if not token:
        print("ERROR: 環境変数 OPENROUTER_API_KEY が未設定です。先に設定してください。")
        return

    print("user: ")
    user_input = input().strip()
    system_instruction = "あなたは強化学習の専門家です。報酬設計をしてください。"

    try:
        content = call_llm(
            user_content=user_input,
            system_instruction=system_instruction,
            # model="openai/gpt-4o-mini",  # 例: 固定したい場合
            # http_referer="https://your-app.example.com",
            # x_title="My RL Reward Designer",
            debug=False,
        )
    except RuntimeError as e:
        print("ERROR:", e)
        return

    print("\nmodel:")
    print(content)


if __name__ == "__main__":
    main()