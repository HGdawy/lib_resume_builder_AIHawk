from typing import List, Dict
from langchain_openai import ChatOpenAI
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

class BaseModelWrapper:
    def __call__(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

class OpenAIChatModel(BaseModelWrapper):
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.4):
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
            temperature=temperature
        )
    
    def __call__(self, messages: List[Dict[str, str]]) -> str:
        result = self.llm.invoke(messages)
        return result.content

class GeminiChatModel(BaseModelWrapper):
    def __init__(self,model_name: str, api_key: str, temperature: float = 0.4):
        genai.configure(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
        )
    
    def __call__(self, messages: List[Dict[str, str]]) -> str:
        result = self.llm.invoke(messages)
        return result.content