# University of Chicago MS in Applied Data Science RAG System Chatbot

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system for the Master's in Applied Data Science program at the University of Chicago. The system provides accurate and contextually relevant answers to user queries about the program, combining efficient information retrieval with advanced natural language generation capabilities.

## Key Features

- Web scraping of program-related content
- Custom embedding model using fine-tuned BAAI/bge-large-en-v1.5
- Vector store and query engine implementation with LlamaIndex
- Custom dataset generation for model fine-tuning
- Advanced fine-tuning techniques (data augmentation, Matryoshka representation)
- Comprehensive evaluation metrics

## System Architecture

### 1. Data Preparation and Embedding
- Process textual content from program documents
- Utilize custom embedding function based on BAAI/bge-large-en-v1.5

### 2. Vector Store and Query Engine
- Create vector store index using LlamaIndex
- Enable efficient retrieval of relevant information

### 3. RAG Implementation
- **Indexing:** Prepare and embed program content
- **Retrieval:** Fetch relevant information based on user queries
- **Generation:** Synthesize coherent responses using a language model

## Performance Highlights

- NDCG@10 scores improved from 0.456 to 0.615 after fine-tuning
- Maintained high relevance scores (0.99) while reducing hallucination
- Average response time under 1.5 seconds for most queries

## Future Directions

- [ ] Implement multilingual support
- [ ] Develop feedback loops for continuous improvement
- [ ] Enhance deep content understanding for complex queries

---

This RAG system demonstrates the potential of AI-powered information retrieval and generation in educational contexts, providing accurate and tailored responses to program-specific queries.
