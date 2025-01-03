import os
import awsgi
import threading
from pathlib import Path
from dotenv import load_dotenv 
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, render_template, request, render_template_string
from langchain.chains.combine_documents import create_stuff_documents_chain

app = Flask(__name__)
retrieval_chain = None
lock = threading.Lock()

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_vector_db(persist_directory: Path) -> Chroma:
    db = Chroma(
        persist_directory=str(persist_directory), 
        embedding_function=OpenAIEmbeddings()
    )
    return db

def create_prompt():
    prompt = ChatPromptTemplate.from_template("""
    You are a Jane Austen literary expert with deep knowledge of all her works.
    Your task is to answer questions based only on the provided context from Jane Austen's novels.
    Consider the historical context of Regency-era England and Austen's writing style when answering.
    Format the output in proper HTML. Avoid using markdown.
    
    When discussing characters or plot points, mention which novel they're from.
    If asked about themes, connect them across multiple novels if relevant.
    
    Below are multiple pieces of context that might be relevant:                                         
    <context>
    {context}
    </context>
    
    Question: 
    <question>                                         
    {input}
    </question>                                        
    """)
    return prompt

def initialize_retrieval_chain():
    global retrieval_chain
    if retrieval_chain is None:
        current_path = Path(__file__)
        root_path = current_path.parent
        vector_db_data_path = root_path / 'data' / 'vector_db'

        # Initialize GPT model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  
            temperature=0.7
        )

        prompt = create_prompt()
        db = load_vector_db(persist_directory=vector_db_data_path)
        retriever = db.as_retriever(
            search_kwargs={"k": 8}  # Number of relevant chunks to retrieve
        )

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ''
    if request.method == 'POST':
        try:
            initialize_retrieval_chain()
            user_input = request.form.get('input_text')
            if user_input:
                with lock:
                    response = retrieval_chain.invoke({"input": user_input})
                answer = response.get('answer', 'I apologize, but I cannot find relevant information in Jane Austen\'s works to answer this question.')
            else:
                answer = 'Please enter a question about Jane Austen\'s works.'
        except Exception as e:
            answer = f'An error occurred: {str(e)}. Please check your OpenAI API key and try again.'
    return render_template('index.html', answer=answer)

def handler(event, context):
    return awsgi.response(app, event, context)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)