"""Main Module"""

import json
import logging
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.prompts import PromptTemplate
from langchain_together import ChatTogether, TogetherEmbeddings

from .utils.common import init_logger
from .utils.jsonoutput import (
    ChartOptionsOutput,
    DashboardLayoutOutput,
    IndoCityOutput,
    Ragoutput,
)

init_logger("hanzo")
logger = logging.getLogger("hanzo")
# logger.setLevel("DEBUG")

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(CURRENT_PATH, "temp", "chroma_db")


class Vectordb:
    """vectordb class"""

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
        self.embedding = TogetherEmbeddings(model=self.model)
        self.db = None

    def create(self, chunk_size=100, chunk_overlap=20):
        """create vectordb

        Args:
            chunk_size (int, optional): _description_. Defaults to 100.
            chunk_overlap (int, optional): _description_. Defaults to 20.
        """
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
        """load existing db"""
        logger.info("Load_DB")
        self.db = Chroma(
            persist_directory=self.db_path, embedding_function=self.embedding
        )
        logger.info("Done loading")
        # results = db.similarity_search_with_relevance_scores(query_text, k=3)


class Talk:
    """AI Talk"""

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
        ragtemplate = """Given the list of context: {context},
            Based on the context only, answer the following question with string one paragraph only: {question}. 
            Let me know if you can't answer it because lack of context"""
        self.ragprompt = PromptTemplate(
            template=ragtemplate,
            input_variables=["context", "question"],
            # partial_variables={"format_instructions": format_output},
        )

        streamtemplate = """You are an expert in Data and AI.
            Do Not Answer question other than about Data and AI, answer it within 1 sentence. 
            If its about data and AI answer it with maximum 15 sentences in paragraph(s). 
            Here is the question: {question}"""
        self.streamprompt = ChatPromptTemplate.from_template(
            template=streamtemplate,
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
        else:
            self.ragchain = None

        self.streamchain = (
            {"question": RunnablePassthrough()}
            | self.streamprompt
            | self.llm
            | StrOutputParser()
        )

    def invoking(self, text_input=None, verbose=False):
        """_summary_

        Args:
            text_input (_type_, optional): _description_. Defaults to None.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        logger.info("invoking")

        if self.ragchain:
            if text_input:
                try:
                    output = self.ragchain.invoke(text_input)
                    output_json = json.loads(output.json())
                    return output_json
                except NotImplementedError as e:
                    logger.info(e)
                    return {
                        "context": [],
                        "answer": (
                            "I can't answer that for now. "
                            "Try rephrase or rerun it in the next 20 seconds."
                        ),
                    }
            else:
                asking_hanzo = True
                while asking_hanzo:
                    input_query = input("What is your question? (type 'exit' to quit) ")
                    if input_query.lower() == "exit":
                        asking_hanzo = False
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
                    except ReferenceError as e:
                        logger.info("I may not getting any context correctly: %s", e)
                return {}

        logger.warning("the vector db is not being setup.")
        return {}

    def streaming(self, input_query=None, stream=True):
        """_summary_

        Args:
            input_query (_type_, optional): _description_. Defaults to None.
            stream (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        answer = ""
        if stream:
            asking_hanzo = True
            while asking_hanzo:
                input_query = input("\nWhat is your question? (type 'exit' to quit) ")
                if input_query.lower() == "exit":
                    asking_hanzo = False
                    logger.info("Goodbye!")
                    break
                for m in self.streamchain.stream(input_query):
                    print(m, end="", flush=True)
            return answer

        answer = self.streamchain.stream(input_query)
        return answer


class DashboardEng:
    """Auto Dashboard"""

    def __init__(
        self,
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        max_token=750,
    ):
        dtem = """Given the sample of a table: {table},
            Suggest charts in the dashboard with this goal: {context}.
            Also, following this rules: {rules}, then {additional_rules}.
            If possible, suggest at least 10 charts or More. With 2 or 3 or 4 numberOnly charts."""

        dtem2 = """Given the chart options and its suggestion: {current_layout}, suggest a real dashboard layout. 
            The basic layout is a grid with 29 columns and 18 rows.
            Not necessary to use all charts, prioritize the most important charts.
            Please adjust the size and location following this rules: {rules}"""

        self.dash_suggest_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a data visualization expert."),
                ("human", dtem),
            ]
        )

        self.dash_adjustment_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a data visualization expert."),
                ("human", dtem2),
            ]
        )

        self.llm = ChatTogether(
            model=model,
            max_tokens=max_token,
            temperature=0.95,  # Adds randomness to outputs
            top_p=0.8,  # Nucleus sampling for diverse responses
        )

        self.chart_suggest_chain = (
            RunnablePassthrough()
            | self.dash_suggest_prompt
            | self.llm.with_structured_output(schema=ChartOptionsOutput)
        )

        self.dash_layout_chain = (
            RunnablePassthrough()
            | self.dash_adjustment_prompt
            | self.llm.with_structured_output(schema=DashboardLayoutOutput)
        )

    def suggest_layout(self, csv_input: str, context: str = None, rules: str = None):
        """_summary_

        Args:
            csv_input (str): _description_
            context (str, optional): _description_. Defaults to None.
            rules (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        input_query = {
            "table": csv_input,
            "context": context,
            "additional_rules": rules,
            "rules": """
                Keep It Simple: Presenting only the essential metrics and visualizations that align with the dashboard's goal.
                Use Clear and Consistent Visualizations: Choose the right chart type (e.g., bar charts for comparisons, line graphs for trends).
            """,
        }

        output = self.chart_suggest_chain.invoke(input_query)
        chart_options = json.loads(output.json())

        input_json = {
            "current_layout": chart_options,
            "rules": """
                Cover the whole grid, no overside, No empty space.
                Utilize Masonry-style layout, placed next to each other without intersecting or overlap.
                The width > height for all charts except table chart.
                numberOnly charts always together. lining horizontally on the top or lining vertically on the left or right side.
                numberOnly charts size ratio is 2:3 or 1:2 or 2:5 for width : height, with minimum height of 4.
                For all charts, Width and height are always larger than 3.
                Table chart is the only type that allow to have height > width.
                numberOnly charts always have the smallest size than the other charts.
            """,
        }

        output_json = self.dash_layout_chain.invoke(input_json)
        real_position = json.loads(output_json.json())

        for chart_option in chart_options["chart_options"]:
            for chart_position in real_position["chart_position"]:
                if chart_option["chart_id"] == chart_position["chart_id"]:
                    chart_option["position"] = chart_position["position"]
                    break

        return chart_options

    def generate(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {"status": "placeholder"}


class IndoCityExpert:
    """City Expert"""

    def __init__(
        self,
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        max_token=750,
    ):
        dtem = (
            "Given the list of strings seperated by comma: {cities},"
            " Please normalized those strings with the closest real cities name in Indonesia, top 3"
        )

        self.dash_suggest_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a Indonesia Geography expert"),
                ("human", dtem),
            ]
        )

        self.llm = ChatTogether(
            model=model,
            max_tokens=max_token,
            temperature=0.8,  # Adds randomness to outputs
            top_p=0.95,  # Nucleus sampling for diverse responses (highly probable words are chosen)
        )

        self.dash_suggest_chain = (
            RunnablePassthrough()
            | self.dash_suggest_prompt
            | self.llm.with_structured_output(schema=IndoCityOutput)
        )

    def normalized_cities(self, cities_list):
        """
        Given the list of strings of cities name,
        This function will suggest 3 closest real cities name in Indonesia.

        Args:
            cities_list (json): A string of cities name separated by comma.

        Returns:
            _type_: A dictionary with a key of cities and a value of list of strings of cities name.
            The list of cities name will be sorted by the closest first.
        """
        input_query = {
            "cities": cities_list,
        }

        output = self.dash_suggest_chain.invoke(input_query)
        if output is None:
            return {"cities": []}
        output_json = json.loads(output.json())
        return output_json

    def generate(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {"status": "placeholder"}
