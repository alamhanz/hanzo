from hanzo import talk, vectordb

vdb = vectordb(model="BAAI/bge-large-en-v1.5")
# vdb.create()
vdb.load()
# print(type(vdb))

# hanzo_talk = talk(vdb.db, model="meta-llama/Meta-Llama-3-8B-Instruct-Lite")
## Not all model in together enable to return json (https://docs.together.ai/docs/json-mode)
hanzo_talk = talk(vdb.db, model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")

hanzo_talk.invoking()
# meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
