import test.context
import httpx
import asyncio
import pytest

@pytest.mark.asyncio
async def test_stream_answer():
    query = "How to fix router?"
    url = 'http://127.0.0.1:8000/question'
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream('Post', url, json={"question": query}) as r:
            async for chunk in r.aiter_text():  # or, for line in r.iter_lines():
                print(chunk)
    