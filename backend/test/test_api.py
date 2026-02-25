import test.context
import httpx
import asyncio
import pytest

@pytest.mark.asyncio
async def test_stream_answer():
    query = "How to fix router?"
    url = 'http://127.0.0.1:8000/question'
    async with httpx.AsyncClient(timeout=None) as client:
        session_id = None
        async with client.stream('Post', url, json={"question": query}) as r:
            async for chunk in r.aiter_text():  # or, for line in r.iter_lines():
                print(chunk)

            session_id = r.headers["X-Session-ID"]

        print(session_id)
        
        follow_up_question = "What if I forget my login credentials to my wifi network?"
        async with client.stream('Post', url, json={"question": follow_up_question, "session_id": session_id}) as r:
            async for chunk in r.aiter_text():  # or, for line in r.iter_lines():
                print(chunk)



        

