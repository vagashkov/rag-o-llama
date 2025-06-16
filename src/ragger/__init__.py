from simple_term_menu import TerminalMenu

from .ragger import Ragger

class RAGWrapper:
    """
    Simple utility class for better user experience
    """

    def load_pdf(self, ragger: Ragger):
        """
        Performs PDF document load
        """
        doc_path = input("Please enter PDF file path: ")
        ragger.ingest_pdf(doc_path)

    def answer_question(self, ragger: Ragger):
        """
        Transfers question to LLM and pronts result
        """
        user_prompt = input("Please type your question: ")
        print(ragger.ask(user_prompt))


    menu_items = [
            "[1] Load PDF",
            "[2] Chat",
            "[3] Exit"
            ]
    def __call__(self):
        """
        Main menu handler
        """

        ragger = Ragger()

        # Create main menu
        main_menu = TerminalMenu(
            self.menu_items
            )
        # Get user's choice
        user_choice = -1
        while user_choice != 2:
            user_choice = main_menu.show()
            if user_choice == 0:
                self.load_pdf(ragger)
            elif user_choice == 1:
                self.answer_question(ragger)
            