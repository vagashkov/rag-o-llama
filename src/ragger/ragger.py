from logging import info
from os.path import exists
from typing import List, Dict, Any

from ollama import list as list_models, pull
from langchain.chains.base import Chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import (
    UnstructuredPDFLoader
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
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

    def __init__(
            self,
            doc_path: str,
            embedding_model: str,
            generation_model: str,
                 ) -> None:
        """
        Initialize Ragger object with source document path
        and embedding and generating models
        """

        # Check if source file exists
        if not exists(doc_path):
            raise FileNotFoundError(
                "Source file not found: {}".format(doc_path)
            )

        self.doc_path = doc_path

        # Check if embedding model exist and pull it otherwise
        if not self.check_model_exists(embedding_model):
            pull(embedding_model)
        self.embedding_model = embedding_model

        # Check if generating model exist and pull it otherwise
        if not self.check_model_exists(generation_model):
            pull(generation_model)
        self.generation_model = ChatOllama(model=generation_model)

        # Base context for LLM operation
        self.context_template = PromptTemplate(
            input_variables=["question"],
            template="""
            You are an AI language model assistant. Your task is to examine
            the data source (PDF file for example) and use its content
            to answer user's questions with maximum precise and efficiency.
            Original question: {question}
            """
        )

        print("RAGger object initilized")

    def split_pdf(self) -> List[Document]:
        """
        Loads source file and splits it into chunks
        for embedding using LLM
        """

        loader = UnstructuredPDFLoader(
            file_path=self.doc_path
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300
            )

        data_chunks = text_splitter.split_documents(
            loader.load()
        )

        info("Document splitting complete...")

        return data_chunks

    def ingest_data(self, data_chunks: List[Document]) -> Chroma:
        """
        Performs data chunks embedding and loading into vector database
        """

        vector_db = Chroma.from_documents(
            documents=data_chunks,
            embedding=OllamaEmbeddings(
                model=self.embedding_model
                ),
            collection_name="basic_rag"
        )

        print("Embedding complete...")

        return vector_db

    def prepare_retriever(self, vector_db: Chroma) -> MultiQueryRetriever:
        """
        Prepares data retriever based on ingested data and LLM context
        """

        return MultiQueryRetriever.from_llm(
            vector_db.as_retriever(),
            self.generation_model,
            prompt=self.context_template
            )

    def build_chain(self) -> Chain:
        """
        Builds LangChain for user queries processing
        """

        data_chunks = self.split_pdf()
        vector_db = self.ingest_data(data_chunks)
        retriever = self.prepare_retriever(vector_db)

        template = """
        Answer the question based ONLY on the following context: {context}
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        return (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.generation_model
            | StrOutputParser()
        )

    def ask(self, question: str) -> Dict[str, Any]:
        chain = self.build_chain()

        return (
            chain.invoke(
                input=(
                    question,
                    )
                )
            )
