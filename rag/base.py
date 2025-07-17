from abc import ABC, abstractmethod
from operator import itemgetter

from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import load_prompt
from urllib.parse import urlencode



class RetrievalChain(ABC):
    def __init__(self):
        # self.source_uri = None
        self.k = 1
        # self.chroma_path = "/Users/netager/Docker_Data/openwebui-dify/rag_data/Chroma_DB/chroma_bank_law_db"
        # self.model_path = "/Users/netager/Docker_Data/openwebui-dify/rag_data/HUGGING_FACE_MODEL/BAAI_bge-m3"
        # self.prompt_path = "/Users/netager/Docker_Data/openwebui-dify/rag_data/prompts/law.yaml"


    @abstractmethod
    def load_documents(self, source_uris):
        """loader를 사용하여 문서를 로드합니다."""
        pass

    @abstractmethod
    def create_text_splitter(self):
        """text splitter를 생성합니다."""
        pass

    def split_documents(self, docs, text_splitter):
        """text splitter를 사용하여 문서를 분할합니다."""
        return text_splitter.split_documents(docs)

    def create_embedding(self):
        return HuggingFaceEmbeddings(
            model_name=self.model_path,
            model_kwargs={"device": "mps"},  # cpu : 'cpu', macOS: 'mps', CUDA: 'cuda'
            encode_kwargs={"normalize_embeddings": True},
        )

    def create_vectorstore(self):
        return Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.create_embedding(),
                collection_name="bank_law_case",
            )

    def create_retriever(self, vectorstore):
        return vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": self.k},
            # Add mmr search for diversity
            # search_type="mmr",
            # search_kwargs={"k": 1, "fetch_k": 3, "lambda_mult": 0.5}
        )

    def create_model(self):
        # return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        return ChatOllama(
                # base_url='http://host.docker.internal:11434',
                base_url=self.ollama_url,
                model=self.model_name,
                temperature=0,
                context_window=8192,
            )
    
    def create_prompt(self):
        # return hub.pull("teddynote/rag-prompt-chat-history")
        return load_prompt(self.prompt_path)
    

    @staticmethod
    def format_docs(docs):
        return "\n".join(docs)

    @staticmethod
    def content_for_documents(document_list):
        """Combine document content with optimized string concatenation"""
        if not document_list:
            return ""
        return "\n\n".join(doc.page_content for doc in document_list)


    # @lru_cache(maxsize=32)  # Cache recent similarity search results
    @staticmethod
    def similarity_search(self, user_input):
        """Perform similarity search with caching for repeat queries"""
        
        search_results = self.vectorstore.similarity_search_with_score(
            query=user_input, 
            k=5,
        )
        
        return search_results


    def process_search_results(self, user_message: str) -> str:
        """Process search results asynchronously"""
        search_results = self.similarity_search(self, user_message)
        if not search_results:
            return ""
            
        linked_docs = []
        base_url = "https://jabis.jbbank.co.kr/jabis_pdf_view"
        
        for search_result in search_results:
            if search_result[1] < 0.8:  # relevance threshold
                params = {
                    "source": search_result[0].metadata["source"], 
                    "title": search_result[0].metadata["title"],
                    "page": search_result[0].metadata["page"] + 1,
                }
                url_with_params = base_url + "?" + urlencode(params)
                
                linked_docs.append(
                    f"👉 [{params['title']}]({url_with_params}) [pages]: {params['page']} [{round(search_result[1],3)}]"
                )
        
        if not linked_docs:
            return ""
            
        return "\n\n 📖 관련 문서 보기\n\n" + "\n\n".join(linked_docs)


    def create_chain(self):
        # docs = self.load_documents(self.source_uri)
        # text_splitter = self.create_text_splitter()
        # split_docs = self.split_documents(docs, text_splitter)

        self.vectorstore = self.create_vectorstore()
        self.retriever = self.create_retriever(self.vectorstore)
        model = self.create_model()
        prompt = self.create_prompt()
        
        # print(f"[base.py] prompt: {prompt}")

        # self.chain = (
        #     {
        #         "question": itemgetter("question"),
        #         "context": itemgetter("context"),
        #         "chat_history": itemgetter("chat_history"),
        #     }
        #     | prompt
        #     | model
        #     | StrOutputParser()
        # )

        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("question") | self.retriever | self.content_for_documents,
                "chat_history": itemgetter("chat_history"),
            }
            | prompt
            | model
            | StrOutputParser()
        )

        return self
