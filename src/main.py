from ragger.ragger import Ragger

doc_path = "/home/debian/llm/data/Test.pdf"
embedding_model = "nomic-embed-text"
generation_model = "llama3.2:1b"

if __name__ == "__main__":
    oracle = Ragger(
        doc_path,
        embedding_model,
        generation_model
    )

    response = oracle.ask("What is this document about?")

    print(response)
