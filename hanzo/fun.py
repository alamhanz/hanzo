import json
import logging
import os

from dotenv import load_dotenv

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
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


class Ragoutput(BaseModel):
    context: list = Field(description="list of the context")
    answer: str = Field(
        description="summary of the answer whioch not more than 10 sentences maximum"
    )


class talk:

    def __init__(
        self,
        vdb=None,
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        max_token=750,
        context_size=8,
    ):
        self.vectordb = vdb
        if self.vectordb is not None:
            # Configure prompt and retrieval chain
            self.retriever = self.vectordb.as_retriever(
                search_kwargs={"k": context_size}
            )
        else:
            self.retriever = None

        ## single shot instead of conversation
        self.ragtemplate = "Given the context: {context}, Based on the context only, answer the following question with string one paragraph only: {question}. Let me know if you can't answer it because lack of context"
        self.ragprompt = PromptTemplate(
            template=self.ragtemplate,
            input_variables=["context", "question"],
            # partial_variables={"format_instructions": format_output},
        )

        self.streamtemplate = "You are an expert in Data and AI. Do Not Answer question other than about Data and AI, answer it within 1 sentence. If its about data and AI answer it with maximum 15 sentences in paragraph(s). Here is the question: {question}"
        self.streamprompt = ChatPromptTemplate.from_template(
            template=self.streamtemplate,
        )

        ## Max Tokens impacting the retriever
        self.llm = ChatTogether(
            model=model,
            max_tokens=max_token,
            temperature=0.8,  # Adds randomness to outputs
            top_p=0.7,  # Nucleus sampling for diverse responses
        )

        if self.retriever is not None:
            self.ragchain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.ragprompt
                | self.llm.with_structured_output(schema=Ragoutput)
            )

        self.streamchain = (
            {"question": RunnablePassthrough()}
            | self.streamprompt
            | self.llm
            | StrOutputParser()
        )

    def invoking(self, text_input=None, verbose=False):
        logger.info("invoking")

        if text_input:
            try:
                output = self.ragchain.invoke(text_input)
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
                    output = self.ragchain.invoke(input_query)
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

    def streaming(self, input_query=None, stream=True):
        if stream:
            ASKING = True
            while ASKING:
                input_query = input("\nWhat is your question? (type 'exit' to quit) ")
                if input_query.lower() == "exit":
                    ASKING = False
                    logger.info("Goodbye!")
                    break
                else:
                    for m in self.streamchain.stream(input_query):
                        print(m, end="", flush=True)
        else:
            answer = self.streamchain.stream(input_query)
            return answer


# https://api.python.langchain.com/en/latest/together/chat_models/langchain_together.chat_models.ChatTogether.html
# LLM may answer with question
