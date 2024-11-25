import logging
import os
import sys

import numpy as np
from dotenv import load_dotenv
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.output_parsers import RegexParser

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.prompts import PromptTemplate
from langchain_together import Together, TogetherEmbeddings
from src.common import init_logger

init_logger("hanzo")
logger = logging.getLogger("hanzo")

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(CURRENT_PATH, "temp", "chroma_db")


class vectordb:

    def __init__(
        self,
        file=os.path.join(CURRENT_PATH, "temp", "about_me.txt"),
        db_path=CHROMA_PATH,
        # model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        model_name="togethercomputer/m2-bert-80M-32k-retrieval",
    ):
        self.file = file
        self.loader = TextLoader(self.file, encoding="windows-1252")
        self.docs = self.loader.load()
        self.db_path = db_path
        self.model_name = model_name
        self.embedding = TogetherEmbeddings(
            # model="togethercomputer/m2-bert-80M-32k-retrieval"
            model=self.model_name
        )

    def create(self, chunk_size=150, chunk_overlap=30):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(self.docs)
        logger.info("split document")

        logger.info("Start DB")
        self.db = Chroma.from_documents(
            splits, self.embedding, persist_directory=self.db_path
        )

    def load(self):
        logger.info("Load_DB")
        self.db = Chroma(
            persist_directory=self.db_path, embedding_function=self.embedding
        )
        logger.info("Done loading")
        # results = db.similarity_search_with_relevance_scores(query_text, k=3)


# print(results)


class talk:
    def __init__(self, vdb):
        self.vectordb = vdb
        # Configure prompt and retrieval chain
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})
        self.template = "Short answer only, no explaination and Avoid repeated answer. Given the context: {context} Answer the question: {question}"
        self.prompt = ChatPromptTemplate.from_template(template=self.template)

        self.llm = Together(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
            max_tokens=300,
            temperature=0.5,  # Adds randomness to outputs
            top_p=0.5,  # Nucleus sampling for diverse responses
        )

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoking(self):
        logger.info("invoking")
        ASKING = True
        while ASKING:
            input_query = input("What is your question? ")
            output = self.chain.invoke(input_query)
            logger.info(output)
            logger.info(output)
