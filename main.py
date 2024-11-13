# https://python.langchain.com/docs/tutorials/rag/
# https://python.langchain.com/docs/integrations/text_embedding/together/
# https://medium.com/@callumjmac/implementing-rag-in-langchain-with-chroma-a-step-by-step-guide-16fc21815339
# https://stackoverflow.com/questions/76650513/dynamically-add-more-embedding-of-new-document-in-chroma-db-langchain
# https://www.together.ai/blog/rag-tutorial-langchain
# https://python.langchain.com/docs/how_to/output_parser_json/

import logging
import os

import numpy as np
from dotenv import load_dotenv
from langchain import hub

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together import TogetherEmbeddings

from common import init_logger

init_logger(__name__)
logger = logging.getLogger(__name__)

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
CHROMA_PATH = "temp/chroma_test/"

# # Load, chunk and index the contents of the blog.
# loader = TextLoader("temp/about_me.txt", encoding="windows-1252")
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
# splits = text_splitter.split_documents(docs)

# logger.info(len(splits))


# # print(splits)
embedding = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-32k-retrieval")

# logger.info("Start DB")
# db = Chroma.from_documents(splits, embedding, persist_directory=CHROMA_PATH)

logger.info("Load_DB")
db_load = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
logger.info("Done loading")
# query_text = "Who is alamsyah?"
# results = db.similarity_search_with_relevance_scores(query_text, k=3)
# print(results)


from langchain.chains import RetrievalQA
from langchain.output_parsers import RegexParser
from langchain_core.output_parsers import StrOutputParser

# from langchain.prompts import PromptTemplate
from langchain_together import Together

# Configure prompt and retrieval chain
retriever = db_load.as_retriever()
template = "Given the context: {context} Answer the question: {question}"
prompt = ChatPromptTemplate.from_template(template=template)

llm = Together(
    model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    max_tokens=350,
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

logger.info("invoking")
# input_query = "Who is alamsyah?"
ASKING = True
while ASKING:
    input_query = input("What is your question? ")
    output = chain.invoke(input_query)
    logger.info(output)
