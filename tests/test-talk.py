import json
import sys

from dotenv import load_dotenv

sys.path.append("../hanzo/")
from hanzo import Talk, Vectordb

load_dotenv()
## Create VDB
# vdb = vectordb(model="BAAI/bge-large-en-v1.5")
vdb = Vectordb(
    model="BAAI/bge-large-en-v1.5",
    file="../datakoen/app/default/my_profile.txt",
    db_path="../datakoen/app/default/about_me/",
)
# vdb.create(chunk_size=200, chunk_overlap=80)
vdb.load()

## Talk
## Not all model in together enable to return json (https://docs.together.ai/docs/json-mode)
hanzo_talk = Talk(
    vdb.db,
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    max_token=800,
    context_size=5,
)
# print(hanzo_talk.invoking(text_input="where is alam lives?", verbose=True))
hanzo_talk.streaming()
