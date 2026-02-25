import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.ingestion import Ingestion

name = "project-assurance-data"

ingest = Ingestion(chunk_size=500, overlap=100)
ingest.create_index(name)





