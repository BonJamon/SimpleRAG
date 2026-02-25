import React, {useState} from 'react';
import logo from './logo.svg';
import './App.css';
import ReactMarkdown from "react-markdown";


async function* streamAnswer(question:string) {
  const response = await fetch("http://localhost:8000/question",{
    method: "Post",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({question: question}),
  });

  console.log("body: ", response.body)

  if (!response.body) {
    throw new Error("Readable Stream not supported");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");

  let buffer="";

  while(true) {
    const {value, done} = await reader.read();
    if (done) break;

    buffer+=decoder.decode(value, {stream:true});

    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      if (part.startsWith("data:")){
        const data = part.replace("data:", "");
        yield data
      }
    }

  }
}

function App() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const handleSubmit = (e: React.SubmitEvent) => {
    e.preventDefault();
    console.log("submitted value: ", question);

    setAnswer("");
    async function run(){
      for await (const chunk of streamAnswer(question)) {
        console.log("juhu")
        console.log("Chunk: ", chunk);
        setAnswer(prev => prev+chunk);
      }
    }
    run();
  }
  const cleanMarkdown = (text: string) => {
    return text
      .replace(/\*\*\s+/g, "**")       // remove space after **
      .replace(/\s+\*\*/g, "**")       // remove space before **
      .replace(/(\d+)\s+\./g, "$1.")   // fix numbered lists
      .replace(/\s+([.,!?;:])/g, "$1"); // fix punctuation spacing
  };
  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value = {question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button type="submit">Submit</button>
      </form>
      <div style={{ marginTop: "20px" }}>
        <ReactMarkdown>
          {cleanMarkdown(answer)}
        </ReactMarkdown>
      </div>
    </div>

  );
}

export default App;
