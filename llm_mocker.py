#!/usr/bin/env python3

import argparse
import asyncio
import json
import time
from hashlib import sha256
from typing import Dict, List

from aiohttp import web


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.strip().split())


def build_reply(messages: List[Dict[str, str]], session_id: str) -> str:
    parts = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
    joined = "\n".join(parts)
    digest = sha256(joined.encode("utf-8")).hexdigest()[:16]
    return f"[mock][session:{session_id or 'unknown'}][hash:{digest}]"


async def handle_chat_completion(request: web.Request) -> web.StreamResponse:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid_json"}, status=400)

    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return web.json_response({"error": "messages_must_be_list"}, status=400)

    session_id = ""
    if isinstance(payload.get("session_id"), str):
        session_id = payload["session_id"].strip()
    elif isinstance(payload.get("metadata"), dict):
        meta_sid = payload["metadata"].get("session_id")
        if isinstance(meta_sid, str):
            session_id = meta_sid.strip()

    reply = build_reply(messages, session_id)

    prompt_tokens = 0
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                prompt_tokens += count_tokens(content)
    completion_tokens = count_tokens(reply)

    created = int(time.time())
    model = payload.get("model", "mock-model")

    if payload.get("stream"):
        response = web.StreamResponse(
            status=200, 
            reason="OK", 
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        await response.prepare(request)

        chunk = {
            "id": f"mock-{created}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": reply},
                    "finish_reason": None
                }
            ]
        }
        await response.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
        await response.drain()

        final_chunk = {
            "id": f"mock-{created}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        await response.write(f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
        await response.drain()
        await response.write(b"data: [DONE]\n\n")
        await response.drain()
        await response.write_eof()
        return response

    body = {
        "id": f"mock-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    return web.json_response(body)


async def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat_completion)
    app.router.add_get("/health", lambda _: web.json_response({"status": "ok"}))
    return app


async def run_server(host: str, port: int):
    app = await create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    print(f"Mock server listening on http://{host}:{port}")
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


async def run_server_until_cancelled(host: str, port: int, shutdown_event: asyncio.Event):
    app = await create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    print(f"Mock server listening on http://{host}:{port}")
    try:
        await shutdown_event.wait()
    finally:
        await runner.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM mock server for dual_round_benchmarker")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (default 8001)")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        asyncio.run(run_server(args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
