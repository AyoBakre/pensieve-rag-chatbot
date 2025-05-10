# Pensieve RAG Chatbot

Pensieve is an intelligent RAG (Retrieval-Augmented Generation) chatbot that helps users find and understand information in their document collections. It provides accurate, helpful, and concise answers based on the provided context from various document sources.

## Features

- **Hybrid Retrieval**: Combines vector similarity and keyword-based search
- **Adaptive Context Relevance**: Evaluates and reranks document relevance
- **Query Refinement**: Automatically improves vague or ambiguous queries
- **Smart Model Selection**: Chooses between more capable models based on query complexity
- **Streaming Responses**: Real-time response generation for better user experience
- **Suggested Questions**: Provides contextually relevant initial questions
- **Follow-up Questions**: Generates intelligent follow-up questions based on previous responses
- **Source Citations**: Clearly attributes information to source documents
- **Feedback System**: Allows users to rate response quality

## Document Collections

The system is designed to work with multiple document collections stored in Pinecone vector databases:

- **Legal Documents**: Contracts, loan agreements, and other legal files
- **Emails**: Email communications related to the project
- **Investment Committee Memos**: Financial documents and memos
- **Slack Messages**: Communications from Slack channels
- **Loan Security Agreements**: Additional legal documents

## Setup

### Prerequisites

- Python 3.8+
- Pinecone account (for vector database)
- OpenAI API key (for embeddings and completion)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/AyoBakre/pensieve-rag-chatbot.git
   cd pensieve-rag-chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   ```

4. Create and configure your Pinecone indexes

### Running the application

```
streamlit run rag_chatbot.py
```

## Usage

1. Start with a suggested question or ask your own
2. View the response with source citations
3. Follow up with suggested questions or ask new ones
4. Provide feedback on response quality with thumbs up/down buttons

## Architecture

- **Frontend**: Streamlit web interface
- **Embedding**: OpenAI's text-embedding-3-small model
- **Vector Database**: Pinecone indexes
- **Response Generation**: OpenAI GPT-4o and GPT-4o-mini

## Configuration

Advanced settings are available in the application sidebar:
- Results per index
- Auto-refine queries option
- Response model selection
- Debug information display

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 