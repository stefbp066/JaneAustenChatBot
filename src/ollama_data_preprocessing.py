import os
from pathlib import Path
import pickle
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

def making_vector_db(documents: list, persist_directory: Path) -> Chroma:
    try:
        embedding = OllamaEmbeddings(model="llama2")
        db = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=str(persist_directory)
        )
        return db
    except Exception as e:
        print(f"Error: Make sure Ollama is running (ollama serve) and llama2 is pulled (ollama pull llama2)")
        raise e

def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent
    processed_data_path = root_path / 'data' / 'processed' / 'documents.pkl'
    vector_db_data_path = root_path / 'data' / 'vector_db'
    vector_db_data_path.mkdir(exist_ok=True)
    
    print("Loading documents...")
    with open(processed_data_path, 'rb') as file:
        documents = pickle.load(file)
    
    print("Creating vector database...")
    db = making_vector_db(documents, vector_db_data_path)
    print("Vector store has been successfully created and persisted at:", vector_db_data_path)

if __name__ == "__main__":
    main()