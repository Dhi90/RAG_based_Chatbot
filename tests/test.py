from chatbot import RAGModel
from data import DataAPI
from database import Database
from ragprocess import RAGProcessor
from interface import UserInterface
import time

def test_functionality():
    print("Testing functionality...")
    # Initialize database
    database = Database()

    # Initialize RAG model
    rag_model = RAGModel(database)

    # Define the WordPress API class
    class WordPressAPI:
        def __init__(self, api_url, model):
            self.api_url = api_url
            self.model = model

        def generate_embeddings(self):
            # Implement the logic to generate embeddings
            pass

    # Initialize WordPress API
    wordpress_api = WordPressAPI(api_url="YOUR_WORDPRESS_API_ENDPOINT", model=rag_model)

    # Generate embeddings and add to database
    embeddings = wordpress_api.generate_embeddings()
    rag_model.add_vectors_to_database(embeddings)

    # Initialize RAG processor
    rag_processor = RAGProcessor(rag_model)

    # Start chat interface
    ui = UserInterface(rag_processor)
    ui.start_chat()

def test_performance():
    print("\nTesting performance...")
    start_time = time.time()
    # Initialize database
    database = Database()

    # Initialize RAG model
    rag_model = RAGModel(database)

    # Initialize WordPress API
    wordpress_api = DataAPI(api_url="YOUR_WORDPRESS_API_ENDPOINT", model=rag_model)

    # Generate embeddings and add to database
    embeddings = wordpress_api.generate_embeddings()
    rag_model.add_vectors_to_database(embeddings)

    # Initialize RAG processor
    rag_processor = RAGProcessor(rag_model)

    # Start chat interface
    ui = UserInterface(rag_processor)
    ui.start_chat()
    end_time = time.time()
    print("Time taken:", end_time - start_time, "seconds")

def test_chain_of_thought():
    print("\nTesting Chain of Thought module...")
    # Initialize database
    database = Database()

    # Initialize RAG model
    rag_model = RAGModel(database)

    # Initialize WordPress API
    wordpress_api = DataAPI(api_url="YOUR_WORDPRESS_API_ENDPOINT", model=rag_model)

    # Generate embeddings and add to database
    embeddings = wordpress_api.generate_embeddings()
    rag_model.add_vectors_to_database(embeddings)

    # Initialize RAG processor
    rag_processor = RAGProcessor(rag_model)

    # Start chat interface
    ui = UserInterface(rag_processor)
    ui.start_chat()

if __name__ == "__main__":
    test_functionality()
    test_performance()
    test_chain_of_thought()
