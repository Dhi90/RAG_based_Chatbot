# we transformer for implement rag model for chatbot purposes
from transformers import RagRetriever, RagTokenizer, RagTokenForGeneration
from data import Data

class RagModel:
    """
    RAG model class for text generation with a database.

    Args:
        database (Data): The database containing the text passages.

    Attributes:
        tokenizer (RagTokenizer): Hugging Face's RAG tokenizer for text tokenization.
        retriever (RagRetriever): Hugging Face's RAG retriever for retrieving relevant passages from the database.
        model (RagTokenForGeneration): Hugging Face's RAG model for text generation.
        database (Data): The database containing the text passages.

    Methods:
        __init__(self, database): Initializes the RAG model with the given database.
    """

    def __init__(self, database):
        """
        Initializes the RAG model with the given database.

        Args:
            database: The database containing the text passages.

        Attributes:
            tokenizer (RagTokenizer): Hugging Face's RAG tokenizer for text tokenization.
            retriever (RagRetriever): Hugging Face's RAG retriever for retrieving relevant passages from the database.
            model (RagTokenForGeneration): Hugging Face's RAG model for text generation.
            database (Data): The database containing the text passages.
        """
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
        self.model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")
        self.database = database
    
    def generate(self, question: str) -> str:
        """
        Generates a response to the given question using the RAG model.
    
        Args:
            question (str): The input question for which a response is to be generated.
    
        Returns:
            str: The generated response to the input question.
    
        Raises:
            ValueError: If the input question is empty.
    
        The method first tokenizes the input question using the RAG tokenizer, then retrieves relevant passages from the database using the RAG retriever. Finally, it generates a response using the RAG model and decodes the generated output using the RAG tokenizer.
        """
        if not question:
            raise ValueError("Input question cannot be empty.")
    
        input_dict = self.tokenizer(question, return_tensors="pt")
        retriever_results = self.retriever(input_dict["input_ids"].to("cuda"))
        out = self.model.generate(
            input_ids=input_dict["input_ids"].to("cuda"),
            retriever_results=retriever_results,
            max_length=200,
            num_return_sequences=1,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
    
    def add_vectors_to_database(self, embeddings):
        """
        Adds the given embeddings to the database.
    
        Args:
            embeddings (list or numpy.ndarray): A list or numpy array of embeddings to be added to the database.
    
        Raises:
            ValueError: If the embeddings are not in the correct format.
    
        Returns:
            None
    
        This method takes a list or numpy array of embeddings and adds them to the database. If the embeddings are not in the correct format, a ValueError will be raised.
        """
        self.database.add_vectors(embeddings)

    def get_database_index(self) -> dict:
        """
        Returns the index of the database.
    
        Returns:
            dict: The index of the database.
    
        This method retrieves and returns the index of the database. The index is a dictionary that maps unique identifiers to the corresponding passages in the database.
        """
        return self.database.get_index()