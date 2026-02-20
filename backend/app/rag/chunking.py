from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from app.rag.embeddings import E5Embeddings
from abc import ABC, abstractmethod

embedding = E5Embeddings()
tokenizer = embedding.tokenizer

def token_length(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


class chunker(ABC):
    def __init__(self, chunk_size=500, overlap=150):
        self.chunk_size=chunk_size
        self.overlap=overlap
    @abstractmethod
    def chunking(self, text:str):
        pass
        

class semantic_chunker(chunker):
    def chunking(self, text):
        """
        Perform semantic chunking on the input text using RecursiveSplitter.

        Args:
            text (str): The document to chunk.
            chunk_size (int): The maximum size of each chunk.
            overlap (int): The number of characters to overlap between chunks.

        Returns:
            List[str]: List of text chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n",". ", " ", ""],
            chunk_size=chunk_size, 
            chunk_overlap=overlap,
            length_function=token_length)
        return splitter.split_text(text)
    

class fixed_chunker(chunker):
    def chunking(self, text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap = overlap,
            length_function=token_length
        )
        return text_splitter.split_text(text)



if __name__=="__main__":
    with open("./data/processed/txt_files/project-assurance-data/Activating_a_new_modem_or_router.txt", "r") as f:
        document = f.read()


    chunk_size = 200
    overlap = 50

    chunker_s = semantic_chunker(chunk_size, overlap)
    chunker_f = fixed_chunker(chunk_size, overlap)


    chunks_fixed = chunker_f.chunking(document)
    lengths_fixed = [token_length(chunk) for chunk in chunks_fixed]

    chunks_semantic = chunker_s.chunking(document)
    lengths_semantic = [token_length(chunk) for chunk in chunks_semantic]

    print("Chunk sizes (fixed): "+ str(lengths_fixed))
    print("Chunk sizes (semantic): "+ str(lengths_semantic))