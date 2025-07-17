from typing import List, Annotated

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.base import RetrievalChain

class PDFRetrievalChain(RetrievalChain):
    # def __init__(self, source_uri: Annotated[str, "Source URI"]):
    def __init__(self):
        # self.source_uri = source_uri
        self.k = 1
        self.model_name = "gemma3:12b"
        # self.ollama_url = "http://192.168.50.70:11434"
        self.ollama_url = "http://localhost:11434"
        self.model_path = "/Users/netager/Docker_Data/openwebui-dify/rag_data/HUGGING_FACE_MODEL/BAAI_bge-m3"
        self.chroma_path = "/Users/netager/Docker_Data/openwebui-dify/rag_data/Chroma_DB_500_50/chroma_bank_law_db"
        self.prompt_path = "/Users/netager/Docker_Data/openwebui-dify/rag_data/prompts/law.yaml"


    def load_documents(self, source_uris: List[str]):
        docs = []
        for source_uri in source_uris:
            loader = PDFPlumberLoader(source_uri)
            docs.extend(loader.load())

        return docs

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
