from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings

from langchain_community.embeddings import HuggingFaceEmbeddings



class E5Embeddings(Embeddings):
    def __init__(self, model_name: str = "intfloat/e5-small", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        batch_dict = self.tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self._mean_pooling(outputs, batch_dict["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    # Required by LangChain
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = ["passage: " + text for text in texts]
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        text = "query: " + text
        return self._embed([text])[0]
    


if __name__=="__main__":
    docs = ["1", "2"]
    embedder = E5Embeddings()
    embeddings = embedder.embed_documents(docs)

    embedder_2 = HuggingFaceEmbeddings(
        model_name="intfloat/e5-small",
        model_kwargs={"device": "cpu"},  # or "cpu"
        encode_kwargs={"normalize_embeddings": True}
    )
    embeddings_2 = embedder_2.embed_documents(docs)
    print(torch.norm(torch.tensor(embeddings[0])-torch.tensor(embeddings_2[0])))
    #print(embeddings)
    #print(embeddings_2)