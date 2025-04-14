from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import os
from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel


class QuestionType(Enum):
    QA = "question_answering"
    SUMMARY = "summarization"
    METADATA = "metadata_analytics"
    UNKNOWN = "unknown"


class QuestionClassifier:
    """Classify questions into categories for better handling."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        model_provider: str = "openai",
        temperature: float = 0.2,
        openai_api_key: str = None,
    ):
        self.llm = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            temperature=temperature,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        )

        self.prompt = ChatPromptTemplate.from_template("""
        You are classifying questions about Salesforce earnings call transcripts.
        Classify the following question into one of these categories:
        - question_answering: Any factual questions about Salesforce's business, risks, performance, or statements from earnings calls
        - summarization: Requests to synthesize, condense, or summarize content from earnings calls
        - metadata_analytics: Questions about the documents themselves (count, dates, pages, etc.)

        Question: {question}
        
        Return only the category name as a single word without explanation. If you're in doubt, classify it as "question_answering".
        """)

    def classify(self, question: str) -> Dict[str, Any]:
        messages = self.prompt.invoke({"question": question})
        response = self.llm.invoke(messages)
        question_type = response.content.strip().lower()

        # Map to enum if possible, default to QA
        try:
            return {"type": QuestionType(question_type), "question": question}
        except ValueError:
            return {"type": QuestionType.QA, "question": question}
