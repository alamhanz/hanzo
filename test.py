from hanzo.main import talk, vectordb

vdb = vectordb()
# vdb.create()
vdb.load()
# print(type(vdb))

hanzo_talk = talk(vdb.db)
hanzo_talk.invoking()
