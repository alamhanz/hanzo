import json
import logging
import os
import sys

import numpy as np
from dotenv import load_dotenv
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.prompts import PromptTemplate
from langchain_together import ChatTogether, Together, TogetherEmbeddings
from pydantic import BaseModel, Field

from .src.common import init_logger

init_logger("hanzo")
logger = logging.getLogger("hanzo")
# logger.setLevel("DEBUG")

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(CURRENT_PATH, "temp", "chroma_db")


class vectordb:

    def __init__(
        self,
        file=os.path.join(CURRENT_PATH, "..", "temp", "about_me.txt"),
        db_path=CHROMA_PATH,
        # model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        model="togethercomputer/m2-bert-80M-32k-retrieval",
    ):
        self.file = file
        self.loader = TextLoader(self.file, encoding="windows-1252")
        self.docs = self.loader.load()
        self.db_path = db_path
        self.model = model
        self.embedding = TogetherEmbeddings(
            # model="togethercomputer/m2-bert-80M-32k-retrieval"
            model=self.model
        )

    def create(self, chunk_size=100, chunk_overlap=20):
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


class Hanz(BaseModel):
    context: list = Field(description="list of context from human")
    answer: str = Field(description="answer of the question given context")


class talk:

    def __init__(
        self,
        vdb,
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        max_token=750,
        context_size=8,
    ):
        self.vectordb = vdb
        # Configure prompt and retrieval chain
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": context_size})

        ## single shot instead of conversation
        self.template = "Given the context only: {context}, Answer the following question with string one paragraph only (10 sentences maximum): {question}"
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"],
            # partial_variables={"format_instructions": format_output},
        )

        ## Max Tokens impacting the retriever

        self.llm = ChatTogether(
            model=model,
            max_tokens=max_token,
            temperature=0.8,  # Adds randomness to outputs
            top_p=0.7,  # Nucleus sampling for diverse responses
        )

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm.with_structured_output(schema=Hanz)
        )
        # | StrOutputParser()

    def invoking(self, text_input=None, verbose=False):
        logger.info("invoking")

        if text_input:
            try:
                output = self.chain.invoke(text_input)
                output_json = json.loads(output.json())
                return output_json
            except Exception as e:
                logger.info(e)
                return {
                    "context": [],
                    "answer": "I can't answer that for now. Try rephrase it.",
                }
        else:
            ASKING = True
            while ASKING:
                input_query = input("What is your question? (type 'exit' to quit) ")
                if input_query.lower() == "exit":
                    ASKING = False
                    print("Goodbye!")
                    break
                try:
                    output = self.chain.invoke(input_query)
                    if output:
                        output_json = json.loads(output.json())
                        logger.debug(output_json["context"])
                        logger.info(output_json["answer"])
                    else:
                        logger.info("No Answer")

                    if verbose:
                        logger.info("Real response: %s", output)
                except Exception as e:
                    logger.info("I may not getting any context correctly: %s", e)


# https://api.python.langchain.com/en/latest/together/chat_models/langchain_together.chat_models.ChatTogether.html
# LLM may answer with question
