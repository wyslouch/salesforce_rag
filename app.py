import streamlit as st
from rag_pipeline import RAGPipeline
from db import VectorStore, get_vector_store
from processor import PdfProcessor
from llm import LLMClient
from dotenv import load_dotenv
import os
from question_classifier import QuestionClassifier
from time_constraint_extractor import TimeConstraintExtractor


def get_embedding_dimension(model_name):
    dimensions = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    return dimensions.get(model_name, 1536)


@st.cache_resource
def initialize_components():
    load_dotenv(override=True)

    vector_store_type = os.getenv("VECTOR_DB", "qdrant")
    collection_name = os.getenv("COLLECTION_NAME", "pdf_docs")
    use_disk = os.getenv("USE_DISK", "False").lower() == "true"
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    model_provider = os.getenv("MODEL_PROVIDER", "openai")
    temparature = float(os.getenv("TEMPERATURE", 0.2))
    pdf_directory = os.getenv("PDF_DIRECTORY", "transcripts")
    classification_model = os.getenv("CLASSIFICATION_MODEL", "gpt-3.5-turbo")
    classification_model_provider = os.getenv("CLASSIFICATION_MODEL_PROVIDER", "openai")
    classification_model_temperature = float(
        os.getenv("CLASSIFICATION_MODEL_TEMPERATURE", 0.2)
    )
    filtering_model = os.getenv("FILTERING_MODEL", "gpt-3.5-turbo")
    filtering_model_provider = os.getenv("FILTERING_MODEL_PROVIDER", "openai")
    filtering_model_temperature = float(os.getenv("FILTERING_MODEL_TEMPERATURE", 0.2))

    vector_size = get_embedding_dimension(embedding_model)
    if not openai_api_key:
        return {
            "error": "OpenAI API key is required. Please set the OPENAI_API_KEY environment variable."
        }
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        return {
            "error": f"The directory '{pdf_directory}' was created but is empty. Please add PDF files."
        }

    # Check if there are PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return {
            "error": f"No PDF files found in '{pdf_directory}'. Please add PDF files."
        }
    # Initialize the vector store
    vector_store: VectorStore = get_vector_store(
        vector_store_type=vector_store_type,
        collection_name=collection_name,
        use_disk=use_disk,
        vector_size=vector_size,
    )

    # Initialize the PDF processor
    pdf_processor = PdfProcessor(
        pdf_directory=pdf_directory,
        vector_store=vector_store,
        collection_name=collection_name,
        embedding_model=embedding_model,
        openai_api_key=openai_api_key,
    )
    pdf_processor.process_pdfs()

    # Initialize the filtering model
    time_constraint_extractor = TimeConstraintExtractor(
        model_name=filtering_model,
        model_provider=filtering_model_provider,
        temperature=filtering_model_temperature,
        openai_api_key=openai_api_key,
        vector_store_type=vector_store_type,
    )
    # Initialize the question classifier
    question_classifier = QuestionClassifier(
        model_name=classification_model,
        model_provider=classification_model_provider,
        temperature=classification_model_temperature,
        openai_api_key=openai_api_key,
    )

    # Initialize the LLM client
    llm_client = LLMClient(
        model_name=model_name,
        model_provider=model_provider,
        temperature=temparature,
        openai_api_key=openai_api_key,
    )

    # Initialize the RAG pipeline
    rag = RAGPipeline(
        pdf_processor=pdf_processor,
        llm_client=llm_client,
        question_classifier=question_classifier,
        time_constraint_extractor=time_constraint_extractor,
    )
    return rag


def main():
    # Set up the Streamlit app
    st.set_page_config(page_title="RAG PDF Search", page_icon=":book:", layout="wide")

    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = True
        st.session_state.rag = initialize_components()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    if "use_context" not in st.session_state:
        st.session_state.use_context = True

    if isinstance(st.session_state.rag, dict) and "error" in st.session_state.rag:
        error_message = st.session_state.rag["error"]
        st.error(f"⚠️ {error_message}")
        return

    with st.sidebar:
        st.title("Settings")
        st.session_state.use_context = st.checkbox(
            "Use conversation context", value=True
        )
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

    # Show spinner only when processing
    if st.session_state.is_processing:
        with st.chat_message("assistant"):
            st.spinner("Thinking...")
        st.chat_input("Processing your request...", disabled=True)
    else:
        user_question = st.chat_input("Type your message here...")

        if user_question:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )
            st.session_state.is_processing = True
            st.rerun()

    if st.session_state.is_processing and st.session_state.chat_history:
        # Get the last user message
        last_user_message = next(
            (
                msg["content"]
                for msg in reversed(st.session_state.chat_history)
                if msg["role"] == "user"
            ),
            None,
        )

        if last_user_message:
            conversation_context = ""
            if st.session_state.use_context:
                recent_messages = st.session_state.chat_history[
                    -6:
                ]  # Last 3 exchanges (3 user + 3 assistant messages)
                if len(recent_messages) > 1:  # Ensure there are previous messages
                    conversation_context = "\n".join(
                        [
                            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                            for msg in recent_messages[:-1]
                        ]
                    )
            answer = st.session_state.rag.run(
                last_user_message,
                conversation_context=conversation_context
                if st.session_state.use_context
                else None,
            )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
            st.session_state.is_processing = False
            st.rerun()


if __name__ == "__main__":
    main()
