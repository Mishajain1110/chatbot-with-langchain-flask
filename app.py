from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader

# Part 1
app = Flask(__name__)
data_sources = [
    "https://brainlox.com/courses/category/technical"
]

def load_split_data(urls):
    urlloader = SeleniumURLLoader(urls=urls)
    loaded_document = urlloader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(loaded_document)

text_chunks = load_split_data(data_sources)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

#Part 2
vector_db = FAISS.from_documents(text_chunks, embedding_model)
language_model = Ollama(model="llama3")
conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

retrieval_chain = ConversationalRetrievalChain.from_llm(llm = language_model, chain_type="stuff", retriever=vector_db.as_retriever(search_kwargs={"k" : 2}), memory= conversation_memory)

# Part 3
@app.route("/chat", methods=["POST"])
def chat_with_model():
    user_input = request.json.get("user_input")
    if not user_input:
        return jsonify({"error": "No user input provided"}), 400
    
    result = retrieval_chain({"question": user_input, "chat_history": []})
    return jsonify({"answer": result["answer"]})


if __name__ == "__main__":
    app.run(debug=True)