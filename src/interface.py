class UserInterface:
    """
    A class for creating a chatbot interface.

    Attributes:
        rag_processor (RagProcessor): An instance of the RagProcessor class, responsible for generating chatbot responses.

    Methods:
        start_chat(self): Starts a chat session with the user.
    """

    def __init__(self, rag_processor):
        """
        Initializes a new instance of the UserInterface class.

        Args:
            rag_processor (RagProcessor): An instance of the RagProcessor class, responsible for generating chatbot responses.

        Returns:
            None
        """
        self.rag_processor = rag_processor

    def start_chat(self):
        """
        Starts a chat session with the user.

        Args:
            None

        Returns:
            None

        Raises:
            None

        This method starts a chat session with the user by printing a welcome message and then enters a loop where it continuously prompts the user for input. If the user inputs "exit", the loop breaks. Otherwise, the chatbot generates a response using the `generate_response` method of the `rag_processor` instance and prints it.
        """
        print("Chatbot: Hi! How can I assist you today?")
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break
            response = self.rag_processor.generate_response(user_input)
            print("Chatbot:", response)