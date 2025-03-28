import json
import sys

from dotenv import load_dotenv

sys.path.append("../hanzo/")
from hanzo import DashboardEng, IndoCityExpert, Talk, Vectordb

load_dotenv()
# ## Create VDB
# # vdb = vectordb(model="BAAI/bge-large-en-v1.5")
# vdb = Vectordb(
#     model="BAAI/bge-large-en-v1.5",
#     file="../datakoen/app/default/my_profile.txt",
#     db_path="../datakoen/app/default/about_me/",
# )
# # vdb.create(chunk_size=200, chunk_overlap=80)
# vdb.load()

# ## Talk
# ## Not all model in together enable to return json (https://docs.together.ai/docs/json-mode)
# hanzo_talk = Talk(
#     vdb.db,
#     model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
#     max_token=800,
#     context_size=5,
# )
# print(hanzo_talk.invoking(text_input="where is alam lives?", verbose=True))
# # hanzo_talk.streaming()


## Dashboard Engineer
dashboard_engineer = DashboardEng(model="deepseek-ai/DeepSeek-V3")
input_query = {
    "csv_input": """
This dataset contains information about product sales in a retail store.
Transaction_ID,Product_Name,Quantity_Sold,Unit_Price,Transaction_Date,Sales_Area,Merchant_ID,Payment_Method,Customer_ID,Discount_Applied
1,Apple,5,1.2,2025-01-01,North,M001,Credit Card,C001,10% off
2,Banana,3,0.8,2025-01-02,South,M002,Cash,C002,$1 discount
3,Orange,7,1.5,2025-01-03,East,M003,Mobile Payment,C001,5% off
4,Grapes,2,2.5,2025-01-04,West,M004,Credit Card,C003,No discount
""",
    "context": """summary of the sales.""",
    "rules": """show total customer.""",
}

output = dashboard_engineer.suggest_layout(**input_query)


with open("tmp.json", "w") as fp:
    json.dump(output, fp, indent=4)

print(output)


# ## City Expert
# city_expert = IndoCityExpert()
# input_query = {
#     "cities_list": ",".join(
#         [
#             "Jakrta",
#             "Sulawesi Utra",
#             "kaltim",
#             "sumut",
#             "kalteng",
#             "D.I Yogyakarta",
#             "Purwakart",
#             "Subng",
#             "sukabuma",
#             "taskmalaya",
#         ]
#     ),
# }
# output = city_expert.normalized_cities(**input_query)
# print(output)
