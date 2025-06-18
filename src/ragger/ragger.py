from os.path import exists
from typing import Dict, Any

from environs import env
from environs.exceptions import EnvError

from ollama import list as list_models, pull
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Ragger:
    """
    Given the PDF file, processes it to use as a context for queries
    """

    def check_model_exists(self, model_name: str) -> bool:
        # Checks if designated model is installed
        return model_name in [
            model["model"] for model in list_models()["models"]
        ]

    def __init__(self) -> None:
        """
        Initialize Ragger object with source document path
        and embedding and generating models
        """

        # Read env file
        env.read_env()

        # Extract embedding model name from settings
        try:
            embedding_model = env.str("EMBEDDING_MODEL")
        except EnvError:
            raise ValueError(
                "Embedding model name not found"
            )

        # Check if embedding model exist and pull it otherwise
        if not self.check_model_exists(embedding_model):
            pull(embedding_model)
        self.embedding_model = embedding_model

        # Extract generating model name from settings
        try:
            generating_model = env.str("GENERATING_MODEL")
        except EnvError:
            raise ValueError(
                "Generating model name not found"
            )

        # Check if generating model exist and pull it otherwise
        if not self.check_model_exists(generating_model):
            pull(generating_model)
        self.generating_model = ChatOllama(model=generating_model)

        try:
            context_template = env.str("CONTEXT_TEMPLATE")
        except EnvError:
            raise ValueError(
                "LLM session context template not found"
            )

        # Base context for LLM operation
        self.context_template = PromptTemplate(
            input_variables=["question"],
            template=context_template
        )

        # Extract vector db storage
        try:
            vector_db_storage = env.str("VECTOR_DB_STORAGE")
        except EnvError:
            raise ValueError(
                "Vector DB storage not found"
            )
        self.vector_db_storage = vector_db_storage

    def __str__(self) -> str:
        return (f"""
              RAGger object:
              - embedding model: {self.embedding_model}
              - generating model: {self.generating_model.model}
              - vector db storage: {self.vector_db_storage}
              - context: {self.context_template.template}
            """)

    def ingest_pdf(self, doc_path: str) -> None:
        """
        Ingests PDF file content into vector database for farther using
        """

        # Check if source file exists
        if not exists(doc_path):
            raise FileNotFoundError(
                "Source file not found: {}".format(doc_path)
            )

        # Load doc and estimate it's size
        loader = PDFPlumberLoader(file_path=doc_path)

        # Split doc into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300
        )

        doc_pages = text_splitter.split_documents(
            loader.load()
            )

        # Convert chunks into embeddings in vector database
        vector_db = Chroma.from_documents(
            documents=doc_pages,
            embedding=OllamaEmbeddings(
                model=self.embedding_model
                ),
            persist_directory=self.vector_db_storage,
            collection_name="local-rag-docs",
        )

        print("{} document pages processed".format(len(doc_pages)))

        # Prepare retriever
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(),
            self.generating_model,
            prompt=self.context_template
        )

        # ...and finally build chain
        template = """
        Answer the question based ONLY on the following context: {context}
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        self.chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.generating_model
            | StrOutputParser()
            )

    def ask(self, question: str) -> Dict[str, Any]:
        return (
            self.chain.invoke(
                input=(
                    question,
                    )
                )
            )
