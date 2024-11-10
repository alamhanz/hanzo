# https://python.langchain.com/docs/tutorials/rag/
# https://python.langchain.com/docs/integrations/text_embedding/together/
# https://medium.com/@callumjmac/implementing-rag-in-langchain-with-chroma-a-step-by-step-guide-16fc21815339
# https://stackoverflow.com/questions/76650513/dynamically-add-more-embedding-of-new-document-in-chroma-db-langchain
# https://www.together.ai/blog/rag-tutorial-langchain
# https://python.langchain.com/docs/how_to/output_parser_json/

import os
from langchain import hub
from dotenv import load_dotenv
import numpy as np

from langchain_together import TogetherEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma



load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
CHROMA_PATH = "temp/chroma_test/"

# Load, chunk and index the contents of the blog.
loader = TextLoader("temp/about_me.txt", encoding='windows-1252')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

print(len(splits))


# print(splits)
embedding = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-32k-retrieval")

db = Chroma.from_documents(
    splits,
    embedding,
    persist_directory=CHROMA_PATH
  )

# Persist the database to disk
db.persist()

db_load = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

# query_text = "Who is alamsyah?"
# results = db.similarity_search_with_relevance_scores(query_text, k=3)
# print(results)


from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import RegexParser

# Configure prompt and retrieval chain
retriever = db_load.as_retriever()
template = "Given the context: {context} Answer the question: {question}"

prompt =ChatPromptTemplate.from_template(template=template)


llm = Together(
    model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    # together_api_key="..."
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

input_query = "Who is alamsyah?"
output = chain.invoke(input_query)

print(output)
