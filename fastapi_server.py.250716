from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from pydantic import BaseModel
# from langchain_jb import pdf_chain, pdf_retriever
from rag.pdf import PDFRetrievalChain
import asyncio
import os

# tokenizers 병렬 처리 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
app = FastAPI()

# PDF 체인을 전역 변수로 초기화
pdf = PDFRetrievalChain().create_chain()
pdf_retriever = pdf.retriever
pdf_chain = pdf.chain

class ChatRequest(BaseModel):
    question: str
    chat_history: list

@app.post("/streaming_async/chat")
async def streaming_async(request: ChatRequest):
    async def event_stream():
        print(f"query: {request.question}")

        try:
            # PDF 검색을 비동기로 실행
            # search_results = await asyncio.to_thread(pdf_retriever.invoke, request.question)
            # print(f"search_results: {search_results}")

            search_results = pdf.process_search_results(request.question)
            # print(f"search_results: {search_results}")

            # 청크 크기를 조절하여 응답 속도 개선
            buffer = []
            async for chunk in pdf_chain.astream({
                "question": request.question, 
                # "context": search_results, 
                "chat_history": request.chat_history,
            }):
                if len(chunk) > 0:
                    buffer.append(chunk)
                    # 버퍼가 일정 크기가 되면 한 번에 전송
                    if len(buffer) >= 5:
                        yield str(''.join(buffer))
                        buffer = []
            
            # 남은 버퍼 전송
            if buffer:
                yield str(''.join(buffer))

            yield f"\n\n\n{search_results}"            

            # ai_answer = "[감사합니다.](http://jabis.jbbank.co.kr)"
            # yield f"\n\n\n{ai_answer}"
                
        except Exception as e:
            print(f"Error: {e}")
            yield str(e)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# 비동기적 호출: ainvoke
# ----------------------
# @app.get("/async/chat")
# async def async_chat(query: str = Query(None, min_length=3, max_length=50)):
#     response = await chain.ainvoke({"query": query})
#     return response

# 스트리밍 방식의 동기적 호출: stream
# -----------------------------------
# @app.get("/stream/chat")
# @app.post("/stream/chat")
# def stream_chat(query: str = Query(None, min_length=3, max_length=50)):
#     def event_stream():
#         print(f"query: {query}")
#         for chunk in chain.stream({"query": query}):
#             yield chunk
#     return StreamingResponse(event_stream(), media_type="text/event-stream")

# @app.get("/streaming_async/chat")
# @app.post("/streaming_async/chat")
# async def streaming_async(request: ChatRequest):
#     async def event_stream():
#         print(f"query: {request.query}")
#         try:
#             async for chunk in chain.astream({"query": request.query}):
#                 if len(chunk) > 0:
#                     yield f"data: {chunk}\n\n"
#         except Exception as e:
#             yield f"data: {str(e)}\n\n"

#     return StreamingResponse(event_stream(), media_type="text/event-stream")
