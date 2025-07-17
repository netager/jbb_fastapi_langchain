from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.pdf import PDFRetrievalChain

import asyncio
from functools import lru_cache
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# PDF 체인을 전역 변수로 초기화
pdf = PDFRetrievalChain().create_chain()
pdf_retriever = pdf.retriever
pdf_chain = pdf.chain

# 검색 결과 캐싱
@lru_cache(maxsize=100)
def cache_search_results(question: str) -> str:
    return pdf.process_search_results(question)

class ChatRequest(BaseModel):
    question: str
    chat_history: list

@app.post("/streaming_async/chat")
async def streaming_async(request: ChatRequest):
    CHUNK_SIZE = 3  # 작은 청크 사이즈로 변경
    
    async def event_stream():
        try:
            # 검색 결과를 백그라운드에서 미리 가져오기
            search_task = asyncio.create_task(
                asyncio.to_thread(cache_search_results, request.question)
            )

            buffer = []
            async for chunk in pdf_chain.astream({
                "question": request.question,
                "chat_history": request.chat_history or []
            }):
                if chunk:
                    buffer.append(chunk)
                    if len(buffer) >= CHUNK_SIZE:
                        yield str(''.join(buffer))
                        buffer = []
                        await asyncio.sleep(0)  # 다른 코루틴에게 실행 기회 제공
            
            # 남은 버퍼 전송
            if buffer:
                yield str(''.join(buffer))

            # 검색 결과 가져오기
            search_results = await search_task
            if search_results:
                yield f"\n\n{search_results}"

        except Exception as e:
            print(f"Error in streaming: {str(e)}")
            yield str({"error": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx 프록시 버퍼링 비활성화
        }
    )

# 서버 상태 모니터링 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
