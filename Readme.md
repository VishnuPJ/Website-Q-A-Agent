# AI-Powered Documentation Assistant Agent

A sophisticated question-answering system utilizing multiple AI agents to process queries and retrieve accurate information from documentation. The system features a multi-agent architecture for efficient query routing, information retrieval, and response validation.

---

## Technical Architecture Overview

### System Components

1. **User Interface Layer**
   - Manages user query inputs and presents final responses.
   - Provides an intuitive interface for submitting queries and viewing results.
   - Displays system responses, warnings, and clarifications.

2. **Web Crawler Component**
   - **AsyncWebCrawler**: Asynchronously crawls specified web content for improved performance.
   - **Markdown File Processing**: Extracts and processes content into markdown format for structured storage.
   - **Knowledge Graph**:
     - Indexes processed content for efficient retrieval.
     - Maps relationships between different pieces of information.
     - Feeds into the RAG (Retrieval Augmented Generation) system for advanced querying.

3. **LLM Provider**
   - **Ollama Integration**:
     - Utilizes the Qwen 2.5 7B model for high-performance inference.
     - Supports local deployment for reduced latency and enhanced privacy.
     - Handles natural language understanding and generation across subsystems.

4. **Multi-Agent System**
   - **Router Agent**:
     - Routes queries based on their characteristics and processing requirements.
     - Triggers warnings or clarification requests when needed.
   - **Retriever Agent**:
     - Conducts vector search operations on the knowledge base.
     - Integrates with the RAG system for context-aware retrieval.
     - Processes and prepares retrieved information for validation.
   - **Validation Agents**:
     - **Grader Agent**: Performs an initial validation of retrieved information.
     - **Hallucination Checker**: Ensures factual consistency and prevents inaccuracies.
     - **Answer Grader**: Conducts a final quality assessment before delivering the response.

---

## Data Flow

1. **Input Processing**
   - The user query enters through the UI.
   - The Router Agent evaluates the query and directs it to the appropriate subsystem.

2. **Information Retrieval**
   - Vector search is performed if necessary.
   - The knowledge graph is consulted to find relevant information.
   - The RAG system combines retrieved data with LLM-generated insights.

3. **Validation Pipeline**
   - Retrieved information undergoes multiple validation stages:
     - Grader Agent validates relevance.
     - Hallucination Checker ensures accuracy.
     - Answer Grader finalizes quality checks.

4. **Response Generation**
   - The validated response is formatted and delivered to the user interface.
   - Any warnings or clarifications are included.

---

## Technical Specifications

### Web Crawler
- Asynchronous operation for enhanced throughput.
- Markdown-based storage for simplicity and structure.
- Knowledge graph integration to map and index relationships efficiently.

### LLM Integration
- Model: Qwen 2.5 7B
- Deployment: Local via Ollama for privacy and low latency.
- Supports multi-agent parallel processing for faster results.

### Vector Search
- Optimized for rapid and context-aware retrieval.
- Integrated with RAG to merge search results with LLM capabilities.

### Validation System
- Multi-stage verification ensures high-quality responses.
- Hallucination detection prevents misinformation.
- Scoring mechanisms evaluate the final output's relevance and accuracy.

---

## System Requirements

- Hardware capable of running the Qwen 2.5 7B model.
- Storage for knowledge graph and markdown files.
- Network connectivity for web crawling.
- Adequate memory for vector operations and LLM inference.

---

## Setup Instructions

1. **Environment Setup**
   ```bash
   python -m venv venv
   source ./venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install crewai 'crewai[tools]' crawl4ai pydantic keyboard
   ```

   - [https://github.com/unclecode/crawl4ai](https://github.com/unclecode/crawl4ai)
   - [https://github.com/crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)

3. **LLM Setup**
   - Install Ollama from [https://ollama.ai/](https://ollama.ai/).
   - Pull the required model:
     ```bash
     ollama pull qwen2.5
     ```

---

## Dependencies

### Core Dependencies
- Python 3.8+
- CrewAI
- CrewAI tools
- Crawl4AI
- Pydantic
- Keyboard

### LLM Dependencies
- Ollama (local deployment)
- Qwen 2.5 model

### Optional Dependencies
- Huggingface Hub (for embeddings)
- BAAI/bge-small-en-v1.5 (embedding model)

---

## Usage Examples

1. **Basic Usage**
   ```bash
   python app.py --url "https://your-documentation-url.com"
   ```

2. **Interactive Session**
   - Run the script and enter your query, e.g., "What is Slack?".
   - The system will process your query through its multi-agent pipeline and provide a response.

3. **Exit Application**
   - Press the `ESC` key to exit.

---

## Design Decisions

### Multi-Agent Architecture
- Modular design with distinct agents for routing, retrieval, validation, and quality control:
  - Router Agent evaluates query routing.
  - Retriever Agent performs document search and RAG integration.
  - Grader Agent validates retrieved information.
  - Hallucination Checker ensures response authenticity.
  - Answer Grader assesses the overall response quality.

### Flow Control
- CrewAI’s Flow system ensures sequential processing and validation at each stage.
- Error handling and logging are implemented at every step.

### Web Crawling
- Asynchronous crawling improves speed and efficiency.
- Depth-limited crawling avoids unnecessary recursion.
- LLM-based strategies enhance content extraction.

### Storage
- Markdown files are used for structured content storage.
- The vector store maintains a hierarchy for context-aware querying.

---

## Known Limitations

1. **Crawling Limitations**
   - Prolonged extraction times for large datasets.
   - Needs optimization for faster data processing.

2. **LLM Limitations**
   - Limited by the hardware running the local Ollama server.
   - Quantized models may reduce response quality.

3. **Performance Considerations**
   - Sequential processing increases response time.
   - Memory usage grows with deeper crawling operations.

4. **Content Processing**
   - Complex formatting may not be fully preserved.
   - Multimedia content handling is limited.

---

## Future Improvements

- Parallelize components for faster response generation.
- Integrate multiple LLM providers for enhanced versatility.
- Improve multimedia content handling capabilities.
- Support diverse documentation formats.
- Add caching mechanisms for frequently accessed data.
- Implement advanced content extraction strategies.

