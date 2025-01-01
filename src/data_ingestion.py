import os
import sys
import pickle
from pathlib import Path
from base import SaveObjects
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

def load_data(raw_data_path: Path) -> list:
    loader = TextLoader(raw_data_path, encoding="utf-8")
    doc = loader.load()
    return doc

def chunking(doc: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(doc)
    return documents

def main():
    # Get the project root directory
    current_path = Path(__file__)
    root_path = current_path.parent.parent
    
    # Set up paths for raw and processed data
    raw_data_path = root_path / 'data' / 'raw'
    processed_data_path = root_path / 'data' / 'processed'
    processed_data_path.mkdir(exist_ok=True)
    
    all_documents = []
    
    # Process each .txt file in the raw data directory
    for book_file in raw_data_path.glob('*.txt'):
        print(f"Processing {book_file.name}...")
        doc = load_data(book_file)
        documents = chunking(doc)
        all_documents.extend(documents)
        print(f"Finished processing {book_file.name}")
    
    # Save all processed documents
    SaveObjects(path=processed_data_path / 'documents.pkl', object=all_documents)
    print("All documents processed and saved!")

if __name__ == "__main__":
    main()