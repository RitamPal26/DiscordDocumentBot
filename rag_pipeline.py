# rag_pipeline.py

import os
import shutil
import time
import logging
import argparse # <<< IMPROVEMENT: Added for command-line arguments
from typing import List, Dict, Any
from urllib.parse import urlparse
from pathlib import Path
import re

# Firecrawl import
from firecrawl import FirecrawlApp

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from firecrawl import FirecrawlApp, ScrapeOptions

# --- Configuration ---
class Config:
    CHROMA_PATH = "chroma_db"
    MAX_PAGES = 100
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    MIN_CONTENT_LENGTH = 50

    ENHANCED_PROMPT_TEMPLATE = """
Answer the user's question based on the following documentation context:

{context}

---

IMPORTANT INSTRUCTIONS:
- When mentioning setup, installation, or configuration, include relevant documentation URLs.
- If discussing downloadable software (CLI, desktop app, etc.), mention where to get it.
- For specific features or commands, reference the appropriate documentation page.
- Always provide actionable next steps with links when applicable.

Available Documentation Pages:
{source_urls}

---

Question: {input}

Provide a comprehensive answer that includes:
1. Direct answer to the question.
2. Step-by-step instructions when applicable.
3. Relevant documentation links for further reading.
4. Download links or installation commands when needed.
"""

# --- Logging Setup ---
def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --- Utility Functions ---
def is_valid_url(url: str) -> bool:
    """Validate if a URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

class FirecrawlDocumentationScraper:
    """Fast documentation scraper using Firecrawl API."""

    def __init__(self, config: Config):
        self.config = config
        self.api_key = os.getenv('FIRECRAWL_API_KEY')
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable not set. Get your key from https://firecrawl.dev")
        self.app = FirecrawlApp(api_key=self.api_key)
        logger.info("Firecrawl initialized successfully")

    def crawl_documentation(self, base_url: str) -> List[Dict[str, Any]]:
        """Crawl documentation using Firecrawl API."""
        try:
            logger.info(f"Starting Firecrawl crawl from: {base_url}")
        
            # Crawl with minimal parameters to avoid parameter errors
            crawl_result = self.app.crawl_url(
                base_url,
                limit=self.config.MAX_PAGES
            )
        
            # crawl_result is a CrawlStatusResponse object
            # Access the 'data' attribute to get the list of crawled pages
            if not crawl_result or not hasattr(crawl_result, 'data'):
                logger.error("Firecrawl returned no result or invalid format.")
                return []
        
            # Get the actual crawled pages from the data attribute
            crawled_pages = crawl_result.data
            logger.info(f"Successfully crawled {len(crawled_pages)} pages")
        
            return crawled_pages

        except Exception as e:
            logger.error(f"Firecrawl crawling error: {e}", exc_info=True)
            return []

    def convert_to_langchain_documents(self, pages_data: List[Any]) -> List[Document]:
        """Convert Firecrawl results to LangChain Documents."""
        documents = []
        # 'pages_data' is a list of FirecrawlDocument objects
        for page in pages_data:
            try:
                # --- FIX: Changed from .get() to attribute access ---
                # Access 'markdown' or 'content' directly from the page object
                content = page.markdown if hasattr(page, 'markdown') else page.content
                
                # Access metadata attributes directly
                metadata = page.metadata if hasattr(page, 'metadata') else {}
                source = metadata.get('sourceURL', 'Unknown') # Metadata is still a dict
                title = metadata.get('title', '')
                # --- END FIX ---
                
                if len(content.strip()) < self.config.MIN_CONTENT_LENGTH:
                    continue

                doc = Document(
                    page_content=content.strip(),
                    metadata={
                        'source': source,
                        'title': title,
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to process page data. Error: {e}")
                continue
        logger.info(f"Converted {len(documents)} pages to LangChain documents")
        return documents

class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with persistent vector store."""
    def __init__(self, config: Config):
        self.config = config
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self._initialize_models()
        self._load_existing_vector_store()
        
    def generate_guidance(self, task_description: str, context: str) -> str:
        """Generates technical guidance for a specific task using RAG context."""
        
        prompt_template = """
You are "AI Tech Lead," an expert software architect and mentor. Your task is to provide detailed, actionable guidance for a specific development task.

**Instructions:**
1.  Explain the best way to approach the task.
2.  Provide a concise code skeleton or example in the appropriate language.
3.  If the context mentions internal best practices, emphasize them.
4.  Keep the explanation clear and to the point.

---
**Internal Documentation Context:**
{context}

---
**Development Task:**
"{task}"

---
**Your Guidance:**
"""
        
        prompt = prompt_template.format(context=context, task=task_description)
        
        logger.info(f"Generating guidance for: '{task_description}'")
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate guidance from LLM: {e}", exc_info=True)
            return "Sorry, I encountered an error while generating guidance."
        
    def generate_plan(self, feature_request: str) -> str:
        """Generates a technical plan using the LLM without RAG context."""
    
        # <<< FIX: Added a clear example to the prompt for better formatting >>>
        prompt_template = """
You are "AI Tech Lead," an expert software architect. Your task is to decompose a high-level feature request into a detailed, actionable technical plan.

**Output Format Rules:**
- Use GitHub Flavored Markdown.
- Start with a high-level "Architecture Overview."
- Create two main sections: "## 📝 Backend Tasks" and "## 🎨 Frontend Tasks."
- **Crucially, every task MUST be a checklist item with a unique ID in the format (TID-B#) for backend or (TID-F#) for frontend.**

**EXAMPLE OUTPUT:**
## 📝 Backend Tasks
- [ ] (TID-B1) Set up a new database schema for user profiles.
- [ ] (TID-B2) Create API endpoint GET /api/users/{{id}}.
## 🎨 Frontend Tasks
- [ ] (TID-F1) Design the main profile page component in React.
- [ ] (TID-F2) Implement state management for user data.

---

**Feature Request:** "{request}"
"""
    
        prompt = prompt_template.format(request=feature_request)
    
        logger.info(f"Generating plan for: '{feature_request}'")
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate plan from LLM: {e}", exc_info=True)
            return "Sorry, I encountered an error while generating the plan."
        
    def add_to_vector_store(self, base_url: str) -> bool:
        """
        Crawls a URL and adds the documents to the existing vector store
        without deleting previous content.
        """
        if not self.vector_store:
            # If no database exists, this command should just create one.
            logger.info("No existing vector store found. Calling create_vector_store instead.")
            return self.create_vector_store(base_url)
        
        try:
            logger.info(f"Adding new documents from URL: {base_url}")
            start_time = time.time()

            # 1. Crawl and process the new documents
            scraper = FirecrawlDocumentationScraper(self.config)
            pages_data = scraper.crawl_documentation(base_url)
            if not pages_data:
                return False

            documents = scraper.convert_to_langchain_documents(pages_data)
            if not documents:
                return False

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)
            if not splits:
                logger.error("No text chunks created after splitting.")
                return False
            
            # 2. Add the new document splits to the existing database
            logger.info(f"Adding {len(splits)} new text chunks to the vector store.")
            self.vector_store.add_documents(documents=splits)
            
            total_time = time.time() - start_time
            logger.info(f"Successfully added new documents in {total_time:.2f} seconds.")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}", exc_info=True)
            return False

    def _initialize_models(self):
        """Initialize LLM and embedding models with error handling."""
        try:
            # 1. Set up the API key for CloudRift
            rift_api_key = os.getenv("RIFT_API_KEY")
            if not rift_api_key:
                raise ValueError("RIFT_API_KEY not found in environment variables.")

            # 2. Set up the LLM to use the CloudRift API
            self.llm = ChatOpenAI(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-FP8",
                api_key=rift_api_key,
                base_url="https://inference.cloudrift.ai/v1"
         )
        
            # 3. Set up the embedding model to ALSO use the CloudRift API
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # A popular, efficient model for text embeddings
                 model_kwargs={"device": "cpu"}  # Change to "cuda" if you have a GPU for faster performance
        )

            # 4. Test that the LLM is accessible
            self.llm.invoke("test") # Test connection to CloudRift
            logger.info("CloudRift LLM and Embedding models initialized successfully")
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            # Updated error message no longer mentions Ollama
            raise RuntimeError(f"Model initialization failed. Check your RIFT_API_KEY and network connection.")

    def _load_existing_vector_store(self):
        """Load existing vector store from disk if available."""
        if Path(self.config.CHROMA_PATH).exists() and os.listdir(self.config.CHROMA_PATH):
            try:
                self.vector_store = Chroma(
                    persist_directory=self.config.CHROMA_PATH,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vector store from {self.config.CHROMA_PATH}")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                self.vector_store = None
        else:
            logger.info("No existing vector store found.")
            self.vector_store = None

    def has_knowledge_base(self) -> bool:
        """Check if a knowledge base is currently loaded."""
        return self.vector_store is not None

    def get_knowledge_base_info(self) -> dict:
        """
        Get detailed information about the current knowledge base, including
        document count and a list of unique sources.
        """
        if not self.vector_store:
            return {"count": 0, "sources": []}
        
        try:
            # 1. Get all documents' metadata from the vector store
            # This is efficient as it doesn't pull the full document content
            metadata = self.vector_store.get(include=["metadatas"])['metadatas']
            count = len(metadata)
            
            # 2. Extract and process the source URLs to get unique, clean names
            unique_sources = set()
            if metadata:
                for meta in metadata:
                    source_url = meta.get('source')
                    if source_url and source_url != 'Unknown':
                        # Parse the domain name from the URL
                        domain = urlparse(source_url).netloc
                        # Clean up the domain to get a readable name (e.g., "docker" from "docs.docker.com")
                        parts = domain.split('.')
                        if len(parts) > 1:
                            # Typically the second-to-last part is the main name
                            clean_name = parts[-2]
                            unique_sources.add(clean_name.capitalize())

            return {"count": count, "sources": sorted(list(unique_sources))}
            
        except Exception as e:
            logger.error(f"Could not retrieve knowledge base details: {e}")
            # Fallback to a simpler message if details can't be fetched
            try:
                count = self.vector_store._collection.count()
                return {"count": count, "sources": ["various sources"]}
            except:
                return {"count": 0, "sources": []}

    def create_vector_store(self, base_url: str) -> bool:
        """Create vector store using Firecrawl for fast crawling."""
        try:
            start_time = time.time()
            if Path(self.config.CHROMA_PATH).exists():
                logger.info("Clearing existing database...")
                shutil.rmtree(self.config.CHROMA_PATH)

            scraper = FirecrawlDocumentationScraper(self.config)
            pages_data = scraper.crawl_documentation(base_url)
            if not pages_data:
                return False

            documents = scraper.convert_to_langchain_documents(pages_data)
            if not documents:
                return False

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)
            if not splits:
                logger.error("No text chunks created after splitting")
                return False
            
            logger.info(f"Created {len(splits)} text chunks for embedding.")
            
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.config.CHROMA_PATH
            )
            total_time = time.time() - start_time
            logger.info(f"Vector store created in {total_time:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}", exc_info=True)
            return False

    def extract_source_urls(self, docs: List[Document]) -> str:
        """Extract unique source URLs from retrieved documents."""
        source_urls = {doc.metadata.get('source', 'Unknown') for doc in docs}
        return '\n'.join(f"- {url}" for url in sorted(source_urls))

    def query_rag(self, question: str) -> str:
        """Enhanced query using persistent vector store."""
        if not self.vector_store:
            return "No knowledge base is loaded. Please build one first."

        try:
            logger.info(f"Processing query: '{question[:100]}...'")
            
            # <<< FIX: Streamlined RAG chain logic to avoid double retrieval
            # 1. Retrieve relevant documents from the vector store.
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(question)

            if not docs:
                return "No relevant information found in the documentation."

            # 2. Extract source URLs and get the base domain for recommendations.
            source_urls = self.extract_source_urls(docs)
            # <<< FIX: Correctly access metadata from the first document in the list.
            base_domain = ""
            if docs and docs[0].metadata.get('source'):
                base_domain = urlparse(docs[0].metadata['source']).netloc

            # 3. Create a chain that "stuffs" the retrieved documents into the prompt.
            prompt = ChatPromptTemplate.from_template(self.config.ENHANCED_PROMPT_TEMPLATE)
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            
            # 4. Invoke the chain with the retrieved documents.
            response = question_answer_chain.invoke({
                "input": question,
                "context": docs, # Pass the retrieved documents directly
                "source_urls": source_urls
            })
            
            # Add smart recommendations (optional feature, can be expanded)
            # enhanced_answer = self.add_smart_recommendations(response, question, base_domain)
            
            return response # `response` is the final answer string

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return f"An error occurred while processing your question: {e}"

# --- Main Execution ---
def main():
    """Main function to run the RAG pipeline from the command line."""
    # <<< IMPROVEMENT: Added robust command-line argument handling
    parser = argparse.ArgumentParser(description="A RAG pipeline for querying documentation.")
    parser.add_argument("--build", type=str, help="URL to crawl and build the knowledge base.")
    parser.add_argument("--query", type=str, help="A question to ask the pipeline.")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive Q&A mode.")
    
    args = parser.parse_args()

    config = Config()
    try:
        pipeline = EnhancedRAGPipeline(config)

        if args.build:
            if not is_valid_url(args.build):
                print("❌ Error: Please provide a valid URL.")
                return
            print(f"Building knowledge base from: {args.build}...")
            if pipeline.create_vector_store(args.build):
                print("✅ Knowledge base created successfully!")
            else:
                print("❌ Failed to create knowledge base. See rag_pipeline.log for details.")
            return

        # Check for knowledge base before querying
        if not pipeline.has_knowledge_base():
            print("📚 No knowledge base found. Build one first using the --build flag.")
            print("   Example: python rag_pipeline.py --build https://docs.example.com")
            return
            
        print(f"📚 {pipeline.get_knowledge_base_info()}")
        
        if args.query:
            answer = pipeline.query_rag(args.query)
            print(f"\n📖 Answer:\n{answer}")
        elif args.interactive:
            print("\n=== Interactive Mode (type 'quit' to exit) ===")
            while True:
                question = input("\n> ").strip()
                if question.lower() in ['quit', 'exit']:
                    break
                if question:
                    answer = pipeline.query_rag(question)
                    print(f"\n📖 Answer:\n{answer}")
        else:
            print("\nNo action specified. Use --query, --build, or --interactive.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()