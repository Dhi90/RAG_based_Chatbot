class ChainOfThought:
    """
    A class to store and manage a chain of thoughts.

    Attributes:
        thoughts (list): A list to store the thoughts.

    Methods:
        __init__(self): Initializes the ChainOfThought object.
        add_thought(self, thought): Adds a thought to the chain.
        get_chain_of_thought(self): Returns the chain of thoughts as a string.
    """

    def __init__(self):
        """
        Initializes the ChainOfThought object.

        Parameters:
        None

        Returns:
        None
        """
        self.thoughts = []

    def add_thought(self, thought):
        """
        Adds a thought to the chain.

        Parameters:
        thought (str): The thought to be added.

        Returns:
        None
        """
        self.thoughts.append(thought)

    def get_chain_of_thought(self):
        """
        Returns the chain of thoughts as a string.

        Parameters:
        None

        Returns:
        str: The chain of thoughts as a string.
        """
        return "\n".join(self.thoughts)


class RAGProcessor:
    """
    A class to process questions using a RAG model and maintain a chain of thoughts.

    Attributes:
        rag_model (RAGModel): The RAG model used for processing questions.
        chain_of_thought (ChainOfThought): An instance of ChainOfThought to store and manage the chain of thoughts.

    Methods:
        __init__(self, rag_model): Initializes the RAGProcessor object.
        generate_response(self, question): Generates a response to the given question using the RAG model and adds the question and response to the chain of thoughts.
        get_chain_of_thought(self): Returns the chain of thoughts as a string.
    """

    def __init__(self, rag_model):
        """
        Initializes the RAGProcessor object.

        Parameters:
        rag_model (RAGModel): The RAG model used for processing questions.

        Returns:
        None
        """
        self.rag_model = rag_model
        self.chain_of_thought = ChainOfThought()

    def generate_response(self, question):
        """
        Generates a response to the given question using the RAG model and adds the question and response to the chain of thoughts.

        Parameters:
        question (str): The question to be processed.

        Returns:
        str: The generated response.
        """
        response = self.rag_model.generate(question)
        self.chain_of_thought.add_thought(f"Q: {question}\nA: {response}\n")
        return response

    def get_chain_of_thought(self):
        """
        Returns the chain of thoughts as a string.

        Parameters:
        None

        Returns:
        str: The chain of thoughts as a string.
        """
        return self.chain_of_thought.get_chain_of_thought()
