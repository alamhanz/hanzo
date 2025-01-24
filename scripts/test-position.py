"""Main Module"""

import json
import logging
import os
from typing import List

from dotenv import load_dotenv

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.prompts import PromptTemplate
from langchain_together import ChatTogether, TogetherEmbeddings
from pydantic import BaseModel, Field

# from .src.common import init_logger

# init_logger("hanzo")
# logger = logging.getLogger("hanzo")
# logger.setLevel("DEBUG")

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(CURRENT_PATH, "temp", "chroma_db")


class DashboardDetail(BaseModel):
    """Details of each dashboard context"""

    name: str = Field(
        description="The suggested name of the plot. make the name easy to understand."
    )
    plot_type: str = Field(description="The plot suggestion type")
    plot_explaination: str = Field(
        description="The plot description in 3 sentences max which explain what should be put in each axis."
    )


class DashboardSuggestOutput(BaseModel):
    """RAG Styles"""

    top_layer: List[DashboardDetail] = Field(
        description="List of plots contexts with detailed information that located on the top"
    )
    bottom_layer: List[DashboardDetail] = Field(
        description="List of plots contexts with detailed information that located on the bottom"
    )


llm = ChatTogether(
    # model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    max_tokens=750,
    temperature=0.8,  # Adds randomness to outputs
    top_p=0.7,  # Nucleus sampling for diverse responses
)

Dtem = """Given the sample of a table: {table},
        Please suggest a dashboard layout that have a goal: {context}. The basic layout only has 2 layer, Top Layer and Bottom Layer.
        List down what plots should be on the top and the bottom, it is allowed to have zero plot in the layer. 
        There are only 6 types of plots for now: 'bar', 'line', 'numberOnly', 'textOnly', 'table', 'maps'. Also, following this rules: {rules}, then {additional_rules}.
        If possible, suggest at least 4 plots or More."""

# DashSuggestprompt = PromptTemplate(
#     template=Dtem,
#     input_variables=["context", "table", "rules", "additional_rules"],
# )

DashSuggestprompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a data visualization expert."),
        (
            "human",
            Dtem,
        ),
    ]
)


DashSuggestChain = (
    RunnablePassthrough()
    | DashSuggestprompt
    | llm.with_structured_output(schema=DashboardSuggestOutput)
)


# input_query = input("\nWhat is your question? (type 'exit' to quit) ")
# input_query = {
#     "table": """
# This dataset contains information about product sales in a retail store.
# Transaction_ID: A unique identifier for each sales transaction.
# Product_Name: The name of the product sold (e.g., Apple, Banana, etc.).
# Quantity_Sold: The number of units sold in this transaction.
# Unit_Price: The price per unit of the product.
# Transaction_Date: The date and time the transaction took place (e.g., 2025-01-23)
# Sales_Area: The geographic area or region where the sale occurred (e.g., North, South, East, West)
# Merchant_ID: A unique identifier for the merchant or store location where the transaction occurred
# Payment_Method: The method used by the customer to pay (e.g., Credit Card, Cash, Mobile Payment)
# Customer_ID: A unique identifier for the customer making the purchase. Customer may repeat transaction.
# Discount_Applied: The amount of discount applied to the product, if any (e.g., 10% off, $5 discount)
# """,
#     "context": """summary of the sales.""",
#     # "context": """when we got bad sales and worst merchant performance""",
#     # "context": """area performance""",
#     # "context": """customer repeat behaviour and top customer""",
#     "rules": """
# Keep It Simple: Focus on the most important data. Avoid clutter by presenting only the essential metrics and visualizations that align with the dashboard's goal.
# Use Clear and Consistent Visualizations: Choose the right chart type for each data point (e.g., bar charts for comparisons, line graphs for trends).
# """,
# }

input_query = {
    "table": """
This dataset contains information about product sales in a retail store.
Transaction_ID,Product_Name,Quantity_Sold,Unit_Price,Transaction_Date,Sales_Area,Merchant_ID,Payment_Method,Customer_ID,Discount_Applied
1,Apple,5,1.2,2025-01-01,North,M001,Credit Card,C001,10% off
2,Banana,3,0.8,2025-01-02,South,M002,Cash,C002,$1 discount
3,Orange,7,1.5,2025-01-03,East,M003,Mobile Payment,C001,5% off
4,Grapes,2,2.5,2025-01-04,West,M004,Credit Card,C003,No discount
5,Pineapple,4,3.0,2025-01-05,North,M001,Cash,C004,10% off
6,Mango,6,1.8,2025-01-06,South,M005,Mobile Payment,C005,$2 discount
7,Strawberry,8,2.0,2025-01-07,East,M003,Credit Card,C002,No discount
8,Peach,3,1.7,2025-01-08,West,M006,Cash,C003,10% off
9,Watermelon,10,0.5,2025-01-09,North,M007,Mobile Payment,C004,$3 discount
10,Blueberry,5,2.2,2025-01-10,South,M008,Credit Card,C001,No discount
""",
    "context": """summary of the sales.""",
    # "context": """when we got bad sales and worst merchant performance""",
    # "context": """area performance""",
    # "context": """customer repeat behaviour and top customer""",
    # "context": """cohort customer""",
    "rules": """
Layer: First Layer only contains numberOnly / textOnly
Keep It Simple: Focus on the most important data. Avoid clutter by presenting only the essential metrics and visualizations that align with the dashboard's goal.
Use Clear and Consistent Visualizations: Choose the right chart type for each data point (e.g., bar charts for comparisons, line graphs for trends).
""",
    "additional_rules": """show total customer.""",
}
output = DashSuggestChain.invoke(input_query)
output_json = json.loads(output.json())

print("Done")
# show total customer.
