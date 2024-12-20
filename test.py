from dotenv import load_dotenv

from hanzo import talk, vectordb

load_dotenv()
# vdb = vectordb(model="BAAI/bge-large-en-v1.5")
# vdb = vectordb(
#     model="BAAI/bge-large-en-v1.5",
#     file="../datakoen/app/default/my_profile.txt",
#     db_path="../datakoen/app/default/about_me/",
# )
# vdb.create(chunk_size=200, chunk_overlap=80)
# vdb.load()

## Not all model in together enable to return json (https://docs.together.ai/docs/json-mode)
hanzo_talk = talk(
    # vdb.db,
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    max_token=800,
    context_size=5,
)

# hanzo_talk.invoking(verbose=True)
# meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
hanzo_talk.streaming()
