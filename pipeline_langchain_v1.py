import requests
import json
from typing import List, Union, Generator, Iterator

# try:
#     from pydantic.v1 import BaseModel
# except Exception:
#     from pydantic import BaseModel
from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        pass

    def __init__(self):
        self.id = "LangChain Agent(Stream)"
        self.name = "Langchain Agent(Stream)"

    async def on_startup(self):
        print(f"on_startup: {__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
            ) -> Union[str, Generator, Iterator]:

        url = 'http://host.docker.internal:8003/streaming_async/chat'
        headers = {
            'accept': 'text/event-stream',  # SSE 형식으로 변경
            # 'accept': 'application/json',
            'Content-Type': 'application/json'
            # 'Content-Type': 'text/event-stream'

        }

        data = {
            "question": user_message,
            "chat_history": messages}
        
        print(f"data: {data}")
        
        try:
            response = requests.post(url, json=data, headers=headers, stream=True)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("HTTP error occurred:", e)
        except requests.exceptions.RequestException as e:
            print("A request error occurred:", e)

        def process_stream():
            for line in response.iter_lines():
                if line:
                    try:
                        # JSON 응답을 파싱하고 줄바꿈 추가
                        content = line.decode('utf-8')
                        yield content + '\n'
                    except Exception as e:
                        print(f"Error processing line: {e}")
                        continue

                    
                    #     decoded_line = line.decode('utf-8')
                    #     print(f"decoded_line: {decoded_line}")
                    #     print(f"decoded_line.startswith('data: '): {decoded_line.startswith('data: ')}")
                    #     # if decoded_line.startswith('data: '):
                    #         # content = decoded_line[6:]  # 'data: ' 부분을 제거
                    #         # yield content
                    #     content = decoded_line
                    #     yield content

                    # except Exception as e:
                    #     print(f"Error processing line: {e}")
                    #     continue
        
        return process_stream()