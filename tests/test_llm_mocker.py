import asyncio
import json
import sys
import unittest
from pathlib import Path
from aiohttp import ClientSession

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_mocker import run_server_until_cancelled


class LlmMockerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.host = "127.0.0.1"
        self.port = 8765
        self.shutdown_event = asyncio.Event()
        self.server_task = asyncio.create_task(run_server_until_cancelled(self.host, self.port, self.shutdown_event))
        await asyncio.sleep(0.5)

    async def asyncTearDown(self):
        self.shutdown_event.set()
        await self.server_task

    async def test_stream_response(self):
        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        payload = {
            "model": "mock",
            "stream": True,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"}
            ]
        }
        async with ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                self.assertEqual(resp.status, 200)
                chunks = []
                async for line in resp.content:
                    text = line.decode().strip()
                    if text:
                        chunks.append(text)
                self.assertTrue(any("[DONE]" in chunk for chunk in chunks))

    async def test_non_stream_response(self):
        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        payload = {
            "model": "mock",
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        }
        async with ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                self.assertEqual(resp.status, 200)
                body = await resp.json()
                self.assertIn("choices", body)
                self.assertEqual(body["choices"][0]["message"]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
