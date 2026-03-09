#!/usr/bin/env python3
"""调用本地 llama-server 接口的示例脚本
USAGE:
    # 单次对话
    python llama_client.py

    # 流式输出
    python llama_client.py --stream 你好，请介绍一下自己
"""

import argparse
import json
import sys
from typing import Optional

import requests

# 配置
BASE_URL = "http://localhost:8001"
MODEL_NAME = "unsloth/Qwen3.5-4B"

_SESSION = requests.Session()


def _build_messages(prompt: str, system_prompt: Optional[str]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def chat(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    base_url: str = BASE_URL,
    model: str = MODEL_NAME,
    timeout: float = 600,
):
    """发送聊天请求"""
    messages = _build_messages(prompt=prompt, system_prompt=system_prompt)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = _SESSION.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def stream_chat(
    prompt: str,
    system_prompt: Optional[str] = None,
    base_url: str = BASE_URL,
    model: str = MODEL_NAME,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    timeout: float = 600,
):
    """流式聊天请求"""
    messages = _build_messages(prompt=prompt, system_prompt=system_prompt)

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = _SESSION.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()

    print("回复: ", end="")
    for line in response.iter_lines():
        if not line:
            continue

        decoded = line.decode("utf-8", errors="replace").strip()
        if not decoded.startswith("data:"):
            continue

        data = decoded[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        choice0 = (chunk.get("choices") or [{}])[0] or {}
        delta = choice0.get("delta") or {}
        content = delta.get("content") or ""
        if content:
            print(content, end="", flush=True)
    print()

def chat_once():
    """单次对话示例"""
    response = chat(
        system_prompt="你是一个有帮助的AI助手。",
        prompt="你好",
        temperature=0.6
    )

    print("回复:")
    print(response["choices"][0]["message"]["content"])


def _format_request_error(exc: requests.exceptions.RequestException) -> str:
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        body = (exc.response.text or "").strip()
        if body:
            body = body[:2000]
            return f"HTTP {status}: {body}"
        return f"HTTP {status}"
    return str(exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="调用本地 llama-server 的 OpenAI 兼容接口")
    parser.add_argument("prompt", nargs="*", help="单次/流式对话的用户输入")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--system", default=None, help="系统提示词")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--stream", action="store_true", help="流式输出")
    args = parser.parse_args()

    try:
        if args.stream:
            prompt = " ".join(args.prompt).strip() if args.prompt else "你好，请介绍一下自己"
            stream_chat(
                prompt=prompt,
                system_prompt=args.system,
                base_url=args.base_url,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            raise SystemExit(0)

        if args.prompt:
            response = chat(
                prompt=" ".join(args.prompt).strip(),
                system_prompt=args.system,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
            )
            print("回复:")
            print(response["choices"][0]["message"]["content"])
            raise SystemExit(0)

        chat_once()
    except requests.exceptions.RequestException as exc:
        print(f"请求失败: {_format_request_error(exc)}", file=sys.stderr)
        raise SystemExit(1)
