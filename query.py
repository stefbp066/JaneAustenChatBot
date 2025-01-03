import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

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
    You are a Jane Austen literary expert. 
    Your task is to answer questions based only on the provided context from Jane Austen's novels.
    Context: {context}
    Question: {input}
    """)
    return prompt

def test_query(question: str):
    # Set up paths
    current_path = Path(__file__)
    root_path = current_path.parent
    vector_db_data_path = root_path / 'data' / 'vector_db' / 'pride_and_prejudice'
    print(vector_db_data_path)

    # Initialize components
    llm = ChatOpenAI(
        model="gpt-4", 
        temperature=0.7
    )
    prompt = create_prompt()
    db = load_vector_db(persist_directory=vector_db_data_path)
    retriever = db.as_retriever()
    
    # Create chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Make query
    try:
        response = retrieval_chain.invoke({"input": question})
        return response['answer']
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    question = "Why did Elizabeth Bennet reject Darcy's first proposal?"
    print("\nAsking:", question)
    answer = test_query(question)
    print(answer)