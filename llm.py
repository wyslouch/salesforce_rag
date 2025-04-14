from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
import os


class LLMClient:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        model_provider: str = "openai",
        temperature: float = 0.2,
        openai_api_key: Optional[str] = None,
    ):
        self.llm = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            temperature=temperature,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        )

        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant answering questions about documents in a knowledge base.
        Use the following pieces of retrieved context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        When answering, consider both the previous conversation and the retrieved context.
        Make sure your answer is relevant to the current question while maintaining consistency
        with previous responses when appropriate.
        """)

        self.context_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant answering questions about documents in a knowledge base.
        Use the following pieces of retrieved context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Previous conversation:
        {conversation_context}
        
        Context:
        {context}
        
        Question: {question}
        
        When answering, consider both the previous conversation and the retrieved context.
        Make sure your answer is relevant to the current question while maintaining consistency
        with previous responses when appropriate.
        """)

    def generate_answer(
        self,
        question: str,
        context: str,
        conversation_context: Optional[str] = None,
    ) -> str:
        if conversation_context:
            messages = self.context_prompt.invoke(
                {
                    "question": question,
                    "context": context,
                    "conversation_context": conversation_context,
                }
            )
        else:
            messages = self.prompt.invoke({"question": question, "context": context})
        response = self.llm.invoke(messages)
        return response.content.strip()
