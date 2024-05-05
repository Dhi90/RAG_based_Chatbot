import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from chatbot import Model

class DataAPI:
    """
    Class for handling data API and model.
    
    Attributes:
        api_url (str): The URL of the data API.
        sbert_model (SentenceTransformer): A pre-trained Sentence-BERT model for semantic similarity.
        model (Model): The machine learning model used for predictions.
    
    Methods:
        __init__(self, api_url: str, model: Model) -> None:
            Initialize the dataAPI object with the provided API URL and model.
    
            Args:
                api_url (str): The URL of the data API.
                model (Model): The machine learning model used for predictions.
    
            Returns:
                None
    
        get_data(self) -> dict:
            Get data from the provided API URL.
    
            Args:
                None
    
            Returns:
                dict: The data retrieved from the API.
    
        generate_embeddings(self) -> np.ndarray:
            Generate vector embeddings for the textual content in the fetched data using Sentence-BERT.
    
            Args:
                None
    
            Returns:
                np.ndarray: The vector embeddings of the textual content.
    """
    def __init__(self, api_url: str, model: Model) -> None:
        """
        Initialize the dataAPI object with the provided API URL and model.

        Args:
            api_url (str): The URL of the data API.
            model (Model): The machine learning model used for predictions.

        Returns:
            None
        """
        self.api_url = api_url
        self.sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.model = model

    def get_data(self) -> dict:
        """
        Get data from the provided API URL.

        Args:
            None

        Returns:
            dict: The data retrieved from the API.
        """
        response = requests.get(self.api_url)
        data = response.json()
        return data

    def generate_embeddings(self) -> np.ndarray:
        """
        Generate vector embeddings for the textual content in the fetched data using Sentence-BERT.

        Args:
            None

        Returns:
            np.ndarray: The vector embeddings of the textual content.
        """
        # Fetch data from API
        data = self.get_data()

        # Convert textual content into vector embeddings using Sentence-BERT
        text = data['text']
        text_embeddings = self.sbert_model.encode(text)

        # Add vectors to database
        self.model.add_vectors_to_database(text_embeddings)
        return text_embeddings
