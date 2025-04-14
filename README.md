# Salesforce Earnings Call RAG System
The objective is to build a solution that provides Question & Answer
(Q&A) and summarization capabilities based on Salesforce's quarterly earnings presentation
transcripts.

Question Answering: Get accurate answers to specific questions about Salesforce's business, risks, and performance
Summarization: Generate concise summaries of key points, specific topics, or trends
Metadata Analysis: Answer questions about document metadata (dates, page counts, etc.)
Database Statistics: Retrieve information about the document collection
Conversation Context: Maintain context across multiple questions for a more natural interaction
Time-Based Filtering: Filter information by specific fiscal years and quarters

## Requirements

- Python 3.9+
- Make
- OpenAI API key
- PDF transcripts in the `transcripts` directory

## Setup and Installation
1. Clone the repository
2. Set up the environment:
   ```
   bash setup.sh
   ```
3. Configure `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY = "your-api-key-here"
   ```

## Usage

1. Start the application:
   ```
   make run
   ```

2. Access the UI at http://localhost:8501

## Features

- Question answering based on transcript content
- Summarization of key points and trends
- Metadata and document statistics
- Time-based filtering by fiscal years/quarters
- Conversation context for follow-up questions

## Vector Database
Qdrant is used as the vector database and runs locally in one of two modes:

- In-memory mode (default): Faster but data is lost when the application stops
- Disk mode: Persistent storage that preserves embeddings between sessions

To enable disk storage, set USE_DISK = "true" in your .env file. This will store the vector database in a ./qdrant_data directory.