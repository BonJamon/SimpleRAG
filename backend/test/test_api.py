import test.context
import httpx
import asyncio
import pytest
import json

@pytest.mark.asyncio
async def test_stream_answer():
    question = "How to fix router?"
    follow_up_question = "What if I forget my login credentials to my wifi network?"
    url = 'http://127.0.0.1:8000/question'
    #ask question + follow up
    async with httpx.AsyncClient(timeout=None) as client:
        session_id = None
        async with client.stream('Post', url, json={"question": question}) as r:
            async for chunk in r.aiter_text():  # or, for line in r.iter_lines():
                print(chunk)

            session_id = r.headers["X-Session-ID"]

        print(session_id)

        async with client.stream('Post', url, json={"question": follow_up_question, "session_id": session_id}) as r:
            async for chunk in r.aiter_text():  # or, for line in r.iter_lines():
                print(chunk)

            session_id_follow_up = r.headers["X-Session-ID"]


        assert session_id == session_id_follow_up


        #Read session data
        data_file = "./data/session_data/"+session_id+".json"

        with open(data_file, "r", encoding="utf-8") as f:
            messages = json.load(f)

        loaded_question = messages[0]["user"]
        loaded_follow_up_question = messages[2]["user"]

        assert loaded_question == question
        assert loaded_follow_up_question == follow_up_question

        






        

