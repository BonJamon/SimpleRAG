from bs4 import BeautifulSoup
from app.rag.chunking import semantic_chunker
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.rag.embeddings import E5Embeddings

class Ingestion:
    """
    Ingestion: Can update or create index based on folder with raw data
    """
    def __init__(self, 
                 raw_dir="./data/raw", 
                 txt_dir="./data/processed/txt_files", 
                 chunk_dir="./data/processed/chunks", 
                 vectorstore_path="./data/vectorstore/faiss_index",
                 chunker = semantic_chunker,
                 chunk_size=500,
                 overlap=100):
        self.raw_dir = raw_dir
        self.txt_dir = txt_dir
        self.chunk_dir = chunk_dir
        self.vectorstore_path = vectorstore_path
        self.chunker = chunker(chunk_size, overlap)

    def ingest(self, name):
        self._extract(name)
        self._chunk(name)
        self._update_index(name)

    def create_index(self, name):
        self._extract(name)
        self._chunk(name)
        self._create_index(name)

    def _extract(self,name):
        ### Extract Text from HTML
        raw_dir = os.path.join(self.raw_dir,name)
        txt_dir = os.path.join(self.txt_dir,name)
        if not os.path.exists(raw_dir):
            raise ValueError("Directory containing raw documents does not exist")
        if os.path.exists(txt_dir):
            #Already processed
            return
        os.makedirs(txt_dir)

        for filename in os.listdir(raw_dir):
            with open(os.path.join(raw_dir,filename), "rb") as file:
                html_content = file.read()
            soup = BeautifulSoup(html_content, "html.parser")
            txt= soup.get_text()
                
            new_filename = filename[:-4]+"txt"
            with open(os.path.join(txt_dir,new_filename), "w", encoding="utf-8") as file:
                file.write(txt)


    def _chunk(self, name):
        chunk_dir = os.path.join(self.chunk_dir,name)
        txt_dir = os.path.join(self.txt_dir, name)
        if os.path.exists(chunk_dir):
            return
        #### Chunk using semantic chunking
        os.makedirs(chunk_dir)
        for filename in os.listdir(txt_dir):
            #read file
            with open(os.path.join(txt_dir,filename), "r", encoding="utf-8") as file:
                text = file.read()

            #create chunk
            chunks = self.chunker.chunking(text)

            #save chunk
            for i,chunk in enumerate(chunks):
                new_filename = filename[:-4]+"_"+str(i)+".txt"

                with open(os.path.join(chunk_dir,new_filename), "w", encoding="utf-8") as file:
                    text = file.write(chunk)



    def _create_index(self,name):
        chunk_dir = os.path.join(self.chunk_dir,name)
        if os.path.exists(self.vectorstore_path):
            return
        embeddings = E5Embeddings()
        # Construct documents
        docs = []
        for chunk_file in os.listdir(chunk_dir):
            with open(os.path.join(chunk_dir, chunk_file), "r", encoding="utf-8") as f:
                chunk = f.read()
            docs.append(Document(chunk))
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(docs, embeddings)
        # Save
        os.makedirs(self.vectorstore_path)
        vectorstore.save_local(self.vectorstore_path)

    
    def _update_index(self,name):
        chunk_dir = os.path.join(self.chunk_dir,name)
        if not os.path.exists(self.vectorstore_path):
            raise AttributeError("The index does not exist. Use create_index() instead")

        embeddings = E5Embeddings()
        # Construct documents
        docs = []
        for chunk_file in os.listdir(chunk_dir):
            with open(os.path.join(chunk_dir, chunk_file), "r", encoding="utf-8") as f:
                chunk = f.read()
            docs.append(Document(chunk))
        
        vectorstore = FAISS.load_local(self.vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(docs)



    