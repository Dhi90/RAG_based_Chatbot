import faiss
import numpy as np

class Database:
    """
    A class for managing a database using the Faiss library.

    Attributes:
        index (faiss.Index): The Faiss index object used to store and search the database.

    Methods:
        __init__(self): Initializes a new instance of the Database class.
        build_index(self, embeddings): Builds the Faiss index object and adds the given embeddings to it.
        search(self, query_embeddings, k=5): Searches the database using the given query embeddings and returns the indices of the k closest matches.
        add_vectors(self, embeddings): Adds the given embeddings to the database index.
        get_index(self): Returns the current database index.
    """

    def __init__(self):
        """
        Initializes a new instance of the Database class.

        Parameters:
            None

        Returns:
            None

        Raises:
            None

        This method initializes the Faiss index object, which will be used to store and search the database.
        """
        self.index = None

    def build_index(self, embeddings):
        """
        Builds the Faiss index object and adds the given embeddings to it.

        Parameters:
            embeddings (numpy.ndarray): A 2D numpy array containing the embeddings to be added to the index.

        Returns:
            None

        Raises:
            None

        This method initializes the Faiss index object, adds the given embeddings to it, and builds the index.
        """
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def search(self, query_embeddings, k=5):
        """
        Searches the database using the given query embeddings and returns the indices of the k closest matches.

        Parameters:
            query_embeddings (numpy.ndarray): A 2D numpy array containing the query embeddings.
            k (int): The number of closest matches to return.

        Returns:
            list: A list containing the indices of the k closest matches.

        Raises:
            None

        This method uses the Faiss index object to search for the k closest matches to the given query embeddings. It returns a list containing the indices of these matches.
        """
        distances, idx = self.index.search(query_embeddings, k)
        return idx.flatten().tolist()

    def add_vectors(self, embeddings):
        """
        Adds the given embeddings to the database index.

        Parameters:
            embeddings (numpy.ndarray): A 2D numpy array containing the embeddings to be added to the index.

        Returns:
            None

        Raises:
            Exception: If the index is not built.

        This method adds the given embeddings to the database index using the Faiss library.
        """
        if self.index is None:
            raise Exception("Index is not built. Please build the index first using the build_index method.")
        self.index.add(np.array(embeddings))

    def get_index(self):
        """
        Returns the current database index.

        Parameters:
            None

        Returns:
            faiss.Index: The current database index object.

        Raises:
            None

        This method returns the current database index object, which can be used to search for embeddings and add new embeddings to the index.
        """
        return self.index
