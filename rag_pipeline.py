from typing import List
from langchain.schema import Document
from processor import PdfProcessor
from llm import LLMClient
from question_classifier import QuestionClassifier, QuestionType
from langchain_core.prompts import ChatPromptTemplate
from time_constraint_extractor import TimeConstraintExtractor
from typing import Optional


class RAGPipeline:
    """
    Connects the vector store retriever with an LLM for question answering.
    """

    def __init__(
        self,
        pdf_processor: PdfProcessor,
        question_classifier: QuestionClassifier,
        llm_client: LLMClient,
        time_constraint_extractor: TimeConstraintExtractor,
    ):
        self.pdf_processor = pdf_processor
        self.llm_client = llm_client
        self.question_classifier = question_classifier
        self.time_constraint_extractor = time_constraint_extractor

        self.summary_prompt = ChatPromptTemplate.from_template("""
        You are an expert financial analyst specializing in Salesforce earnings calls.
        Provide a clear, concise summary based on the following transcript excerpts.
        
        Focus on:
        - Key financial metrics and performance indicators
        - Strategic initiatives and priorities
        - Market outlook and challenges
        - Product announcements or developments
        - Executive commentary on business direction
        
        Transcript Context:
        {context}
        
        Question: {question}
        
        Create a well-structured summary that addresses the question directly.
        """)

        self.metadata_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant answering questions about Salesforce earnings call transcripts.
        
        Now, use the following content from the transcripts with their metadata to answer the user's question:
        {context}
        
        Question: {question}
        
        If the question is about document metadata (dates, page counts, etc.), focus on the metadata section.
        If the question is about content, focus on the transcript content.
        """)

    def run(
        self,
        question: str,
        top_k: int = 5,
        conversation_context: Optional[str] = None,
    ) -> str:
        time_constraints = self.time_constraint_extractor.detect_time_constraints(
            question
        )
        filters = self.time_constraint_extractor.convert_to_filters(time_constraints)

        question_classification = self.question_classifier.classify(question=question)
        # if question_classification["type"] == QuestionType.UNKNOWN:
        #     return "This question is out of scope. Please ask a specific question about the documents."
        if question_classification["type"] == QuestionType.SUMMARY:
            return self.run_summary(
                question=question,
                top_k=10,
                filters=filters,
                conversation_context=conversation_context,
            )
        elif question_classification["type"] == QuestionType.METADATA:
            return self.run_metadata(
                question=question,
                top_k=10,
                filters=filters,
                conversation_context=conversation_context,
            )
        elif question_classification["type"] == QuestionType.QA:
            return self.run_qa(
                question=question,
                top_k=top_k,
                filters=filters,
                conversation_context=conversation_context,
            )

    def run_qa(
        self,
        question: str,
        top_k: int = 5,
        filters: list = [],
        conversation_context: Optional[str] = None,
    ) -> str:
        documents: List[Document] = self.pdf_processor.retrieve(
            query=question, top_k=top_k, filters=filters
        )
        if not documents:
            return "No relevant documents found."
        documents = sorted(
            documents, key=lambda d: d.metadata.get("score", 0), reverse=True
        )

        context = "\n\n".join([doc.page_content for doc in documents])
        answer = self.llm_client.generate_answer(
            question=question,
            context=context,
            conversation_context=conversation_context,
        )

        return answer

    def run_summary(
        self,
        question: str,
        top_k: int = 10,
        filters: list = [],
        conversation_context: Optional[str] = None,
    ) -> str:
        # use higher top_k for summary
        documents: List[Document] = self.pdf_processor.retrieve(
            query=question, top_k=top_k, filters=filters
        )
        if not documents:
            return "No relevant documents found."
        documents = sorted(
            documents, key=lambda d: d.metadata.get("score", 0), reverse=True
        )

        context = "\n\n".join(
            [
                f"DOCUMENT {i + 1} ({doc.metadata.get('source', 'Unknown')}): {doc.page_content}"
                for i, doc in enumerate(documents)
            ]
        )
        answer = self.llm_client.generate_answer(
            question=question,
            context=context,
            conversation_context=conversation_context,
        )

        return answer

    def run_metadata(
        self,
        question: str,
        top_k: int = 10,
        filters: list = [],
        conversation_context: Optional[str] = None,
    ) -> str:
        documents: List[Document] = self.pdf_processor.retrieve(
            query=question, top_k=top_k, filters=filters
        )
        if not documents:
            return "No relevant documents found."
        vector_store_stats = self.pdf_processor.vector_store.get_stats()
        documents = sorted(
            documents, key=lambda d: d.metadata.get("score", 0), reverse=True
        )

        context = "\n\n".join(
            [
                f"DATABASE STATISTICS: {vector_store_stats}.\n\nDOCUMENT {i + 1}.\n\nMETADATA: {doc.metadata}.\n\nContent: {doc.page_content}"
                for i, doc in enumerate(documents)
            ]
        )
        answer = self.llm_client.generate_answer(
            question=question,
            context=context,
            conversation_context=conversation_context,
        )

        return answer
