import os
from pathlib import Path
from langchain_chroma import Chroma  
from langchain_ollama import OllamaLLM  #
from langchain_ollama import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def load_vector_db(persist_directory: Path) -> Chroma:
    db = Chroma(
        persist_directory=str(persist_directory), 
        embedding_function=OllamaEmbeddings(model="llama3.2")
    )
    return db

def create_prompt():
    prompt = ChatPromptTemplate.from_template("""
    You are a Jane Austen literary expert. 
    Your task is to answer questions based only on the provided context from Jane Austen's novels.
    Make the answer concise and to the point.
    Context: {context}
    Question: {input}
    """)
    return prompt

def test_query(question: str):
    # Set up paths
    current_path = Path(__file__)
    root_path = current_path.parent
    vector_db_data_path = root_path / 'data' / 'vector_db'

    # Initialize components
    llm = OllamaLLM(model="llama3.2")  # Updated to OllamaLLM
    prompt = create_prompt()
    db = load_vector_db(persist_directory=vector_db_data_path)
    retriever = db.as_retriever()
    
    # Create chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Make query
    response = retrieval_chain.invoke({"input": question})
    return response['answer']

if __name__ == "__main__":
    # Make sure Ollama is running first (ollama serve)
    question = "Who is Elizabeth Bennet's cousin?"
    print("\nAsking:", question)
    answer = test_query(question)
    print("\nAnswer:", answer)