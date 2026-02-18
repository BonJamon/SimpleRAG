from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from embeddings import E5Embeddings

embedding = E5Embeddings()

tokenizer = embedding.tokenizer

def token_length(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def semantic_chunking(text, chunk_size=500, overlap=150):
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


def fixed_size_chunking(text, chunk_size=500, overlap=150):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap = overlap,
        length_function=token_length
    )
    return text_splitter.split_text(text)



if __name__=="__main__":
    with open("data/markdown_files/Activating_a_new_modem_or_router.txt", "r") as f:
        document = f.read()


    chunk_size = 200
    overlap = 50

    chunks_fixed = fixed_size_chunking(document, chunk_size, overlap)
    lengths_fixed = [len(chunk) for chunk in chunks_fixed]

    chunks_semantic = semantic_chunking(document, chunk_size, overlap)
    lengths_semantic = [token_length(chunk) for chunk in chunks_semantic]

    print("Chunk sizes (fixed): "+ str(lengths_fixed))
    print("Chunk sizes (semantic): "+ str(lengths_semantic))