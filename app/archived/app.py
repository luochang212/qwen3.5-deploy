from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from typing import Dict, List

import requests

from utils.web_ui import create_ui, custom_css, theme

BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8001")
MODEL_NAME = os.environ.get("LLAMA_MODEL", "unsloth/Qwen3.5-4B")

_SESSION = requests.Session()


def error_summary(err: Exception, limit: int = 500) -> str:
    """总结 LLM 运行错误"""
    import traceback

    # 获取完整报错信息
    full_trace = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    full_trace = full_trace[-5000:]  # 避免报错信息过长

    # 简化报错摘要，不再尝试使用 llm.invoke
    summary = f"\n ⚠️ 发生错误，以下是原始日志：\n{full_trace[:limit]}"
    return summary


def _format_request_error(exc: requests.exceptions.RequestException) -> str:
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        body = (exc.response.text or "").strip()
        if body:
            body = body[:2000]
            return f"HTTP {status}: {body}"
        return f"HTTP {status}"
    return str(exc)


def _stream_chat(
    messages: list[dict[str, str]],
    base_url: str = BASE_URL,
    model: str = MODEL_NAME,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    timeout: float = 600,
):
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    with _SESSION.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()

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
                yield content


async def _astream_chat(
    messages: list[dict[str, str]],
    base_url: str = BASE_URL,
    model: str = MODEL_NAME,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    timeout: float = 600,
):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue()
    sentinel = object()

    def _worker():
        try:
            for delta_text in _stream_chat(
                messages=messages,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(delta_text), loop).result()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

    worker_task = asyncio.create_task(asyncio.to_thread(_worker))
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not worker_task.done():
            worker_task.cancel()
            with contextlib.suppress(Exception):
                await worker_task


async def generate_response(
    message: str,
    history: List[Dict[str, str]],
):
    """生成 LLM 的响应"""
    if not message:
        yield "", history
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    messages = history[:-1]

    # 避免 MCP 调用失败引发的退出
    try:
        # 使用优化显示
        async for delta_text in _astream_chat(messages=messages):
            history[-1]["content"] += delta_text
            yield "", history
    except requests.exceptions.RequestException as err:
        history[-1]["content"] += f"\n ⚠️ 请求失败：{_format_request_error(err)}"
        yield "", history
    except Exception as err:
        history[-1]["content"] += error_summary(err)
        yield "", history

    yield "", history


def get_greeting() -> str:
    return "你好！我可以通过本地 llama-server 与你对话。"


def main():
    """主函数"""
    # 配置网络参数
    # docker 预留操作入口，docker 的 host 一般设置为 0.0.0.0
    parser = argparse.ArgumentParser(description="Gradio Agent APP")
    parser.add_argument("--host", type=str, default="localhost", help="主机地址")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    args = parser.parse_args()

    app = create_ui(
        llm_func=generate_response,
        tab_name="Gradio APP - WebUI",
        main_title="Gradio Agent APP",
        initial_message=[{"role": "assistant", "content": get_greeting()}]
    )

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        theme=theme,
        css=custom_css
    )


if __name__ == "__main__":
    main()
