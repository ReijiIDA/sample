
import os
import json
import requests
from typing import Optional, Dict, Any

# 環境変数からのデフォルト
DEFAULT_API_TOKEN = os.getenv("HF_TOKEN")  # export HF_TOKEN=... を想定
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_API_URL = "https://router.huggingface.co/v1/chat/completions"

DEFAULT_HEADERS = lambda token: {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

def call_llm(
    user_content: str,
    system_instruction: str = "あなたは強化学習の専門家です。報酬設計をしてください。",
    *,
    model: Optional[str] = None,
    api_token: Optional[str] = None,
    api_url: Optional[str] = None,
    max_tokens: int = 600,
    temperature: float = 0.7,
    top_p: float = 0.95,
    extra_payload: Optional[Dict[str, Any]] = None,
    timeout: int = 120
) -> str:
    """
    他ファイルから呼び出すためのAPIラッパ。
    成功時: モデルの出力テキスト（str）を返す。
    失敗時: RuntimeError を送出。

    Parameters
    ----------
    user_content : str
        ユーザーからの入力テキスト
    system_instruction : str
        システムプロンプト
    model : Optional[str]
        使用モデル（未指定なら DEFAULT_MODEL）
    api_token : Optional[str]
        HFトークン（未指定なら DEFAULT_API_TOKEN）
    api_url : Optional[str]
        エンドポイントURL（未指定なら DEFAULT_API_URL）
    max_tokens, temperature, top_p : decoding パラメータ
    extra_payload : Optional[Dict[str, Any]]
        API仕様に応じて追加したいフィールド（e.g., "presence_penalty"など）
    timeout : int
        HTTPタイムアウト（秒）
    """
    token = api_token or DEFAULT_API_TOKEN
    if not token:
        raise RuntimeError("環境変数 HF_TOKEN が未設定です。api_token 引数で渡すか、環境変数を設定してください。")

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
        # 必要に応じて "stream": True にし、ストリーミング対応を別途実装
    }

    if extra_payload:
        # ユーザーの追加指定で上書き・追加
        payload.update(extra_payload)

    try:
        resp = requests.post(url, headers=DEFAULT_HEADERS(token), data=json.dumps(payload), timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTPリクエストに失敗しました: {e}")

    if resp.status_code != 200:
        # デバッグしやすいように本文も含める
        raise RuntimeError(f"APIエラー (status={resp.status_code}): {resp.text}")

    try:
        data = resp.json()
    except ValueError as e:
        raise RuntimeError(f"レスポンスのJSON解析に失敗しました: {e}\nraw={resp.text[:500]}")

    # OpenAI/HF Router 互換フォーマット: choices[0].message.content
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        # 仕様外のレスポンスのときは丸ごと返す（デバッグ用）
        content = json.dumps(data, ensure_ascii=False)

    return content


def main():
    token = DEFAULT_API_TOKEN
    if not token:
        print("ERROR: 環境変数 HF_TOKEN が未設定です。先に設定してください。")
        return

    print("user: ")
    user_input = input().strip()
    system_instruction = "あなたは強化学習の専門家です。報酬設計をしてください。"

    try:
        content = call_llm(
            user_content=user_input,
            system_instruction=system_instruction,
            # modelや各種パラメータは必要に応じて明示指定も可
            # model="Qwen/Qwen2.5-7B-Instruct",
            # temperature=0.7,
            # top_p=0.95,
            # max_tokens=300,
        )
    except RuntimeError as e:
        print("ERROR:", e)
        return

    print("\nmodel:")
    print(content)


if __name__ == "__main__":
    main()