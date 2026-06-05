from flask import Flask, render_template, request, jsonify 
from src.helper import download_hugging_face_embeddings 
from langchain_pinecone import PineconeVectorStore 
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
from dotenv import load_dotenv 
from src.prompt import * 
import os

app = Flask(__name__) 
load_dotenv() 

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY 
os.environ["GROQ_API_KEY"] = GROQ_API_KEY  
 
embeddings = download_hugging_face_embeddings() 
index_name = "medicalbot" 

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
) 

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3}) 

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.4,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)  

@app.route("/") 
def index():
    return render_template("chat.html") 

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.json.get("msg")
        print(f"User message: {msg}")
        response = rag_chain.invoke({"input": msg})
        print("Response:", response["answer"])
        return jsonify({"response": response["answer"]})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        error_msg = "Sorry, something went wrong on the server."
        if "rate_limit" in str(e).lower() or "429" in str(e):
            error_msg = "Rate limit exceeded: Too many requests. Please wait a moment and try again."
        elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
            error_msg = "API Key Error: Please check if your GROQ_API_KEY is correctly set in the .env file."
        return jsonify({"response": error_msg}), 200

if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=8080, debug=True)