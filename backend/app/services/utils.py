import os
import json

def load_conversation(session_id:str):
    session_file = os.path.join("./data/session_data", session_id+".json")
    if os.path.exists(session_file):
        with open(session_file, "r", encoding="utf-8") as f:
            messages = json.load(f)
    else:
        messages = []
    return messages


def save_conversation(question:str, answer:str, session_id:str):
    session_file = os.path.join("./data/session_data", session_id+".json")
    if os.path.exists(session_file):
        #Get conversation
        with open(session_file, "r", encoding="utf-8") as f:
            messages = json.load(f)
    else:
        messages = []
    messages.append({"user": question})
    messages.append({"assistant": answer})
    

    #Save conversation
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(messages, f)

