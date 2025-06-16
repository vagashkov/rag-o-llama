from ragger import RAGWrapper

doc_path = ""
embedding_model = ""
generation_model = ""

if __name__ == "__main__":
    # Create new RAG session
    oracle = RAGWrapper()
    oracle()    
    # oracle = Ragger(
    #     doc_path,
    #     embedding_model,
    #     generation_model
    # )

    # response = oracle.ask("What is this document about?")

    # print(response)
