# LangChain Custom Chatbot with Flask API

This repository contains a custom chatbot built using LangChain and Flask. The chatbot extracts data from external websites, creates embeddings, stores them in a vector store, and offers a RESTful API for handling conversations.

## Features

1. **Data Extraction**: Uses LangChain's URL loaders to extract data from [Brainlox Technical Courses](https://brainlox.com/courses/category/technical).
2. **Embeddings & Vector Store**: Embeds the extracted data using HuggingFace sentence transformers (`sentence-transformers/all-MiniLM-L6-v2`) and stores it in a FAISS vector store for semantic retrieval.
3. **Conversational AI**: Interacts with users using the LangChain conversational retrieval chain and Ollama's language model.
4. **Flask RESTful API**: Provides a simple endpoint to handle conversations.

## Project Structure

- `app.py`: The main Flask application that runs the chatbot, handles requests and responds with conversational AI results.
- `requirements.txt`: List of dependencies required for the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mishajain1110/chatbot-with-langchain-flask.git
   cd chatbot-with-langchain-flask
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Flask application:
   ```bash
   python app.py

## Usage via Postman

1. Open Postman and create a new `POST` request.
2. Set the URL to: http://127.0.0.1:5000/chat
3. In the `Body` tab, select `raw` and choose `JSON`.
4. Use the following JSON format:
```json
{
    "user_input": "Your question here"
}
5. Hit Send and the chatbot will respond with the answer.
