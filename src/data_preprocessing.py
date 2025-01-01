import os
import sys
import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

def making_vector_db(documents: list, persist_directory: Path) -> Chroma:
    """
    Create a vector database from documents using OpenAI embeddings
    
    Args:
        documents (list): List of processed documents
        persist_directory (Path): Directory to save the vector database
    Returns:
        Chroma: The vector database instance
    """
    embedding = OpenAIEmbeddings()
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=str(persist_directory)
    )
    db.persist()  # Make sure to persist the database
    return db

def main():
    # Set up paths
    current_path = Path(__file__)
    root_path = current_path.parent.parent
    processed_data_path = root_path / 'data' / 'processed' / 'documents.pkl'
    vector_db_data_path = root_path / 'data' / 'vector_db'
    
    # Create vector db directory if it doesn't exist
    vector_db_data_path.mkdir(exist_ok=True, parents=True)
    
    # Load processed documents
    print("Loading processed documents...")
    try:
        with open(processed_data_path, 'rb') as file:
            documents = pickle.load(file)
        print(f"Loaded {len(documents)} documents")
    except FileNotFoundError:
        print(f"Error: Could not find processed documents at {processed_data_path}")
        print("Please run data_ingestion.py first")
        return
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return
    
    # Create vector database
    print("Creating vector database...")
    try:
        db = making_vector_db(documents, vector_db_data_path)
        print("Vector store has been successfully created and persisted at:", vector_db_data_path)
    except Exception as e:
        print(f"Error creating vector database: {str(e)}")
        return

if __name__ == "__main__":
    main()