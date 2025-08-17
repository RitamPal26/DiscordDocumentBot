import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

# Load environment variables (needed for the embedding model)
load_dotenv()

# --- Configuration ---
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
DOCUMENTS_TO_SHOW = 5 # How many documents to display

def inspect_database():
    """Loads and inspects the contents of the Chroma vector store."""
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ Database not found at '{CHROMA_PATH}'.")
        return

    print("🔎 Loading existing vector store...")
    try:
        # Initialize the same embedding function used to create the DB
        embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Load the persistent database from disk
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function
        )
        
        # Get the total count
        count = db._collection.count()
        print(f"✅ Database loaded successfully with {count} documents.\n")

        # Fetch the first few documents and their metadata
        print(f"--- Displaying the first {DOCUMENTS_TO_SHOW} documents ---")
        results = db.get(
            limit=DOCUMENTS_TO_SHOW,
            include=["metadatas", "documents"] # We want both content and metadata
        )
        
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])

        if not documents:
            print("No documents found in the database.")
            return

        # Print the results
        for i, (meta, doc) in enumerate(zip(metadatas, documents)):
            print(f"\n--- Document #{i + 1} ---")
            print(f"Metadata: {meta}")
            print(f"Content Preview: {doc[:250].strip()}...")

    except Exception as e:
        print(f"An error occurred while inspecting the database: {e}")
        print("Please ensure Ollama is running.")

if __name__ == "__main__":
    inspect_database()