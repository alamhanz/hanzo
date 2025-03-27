import json
import sys

from dotenv import load_dotenv

sys.path.append("../hanzo/")
from hanzo import IndoCityExpert

load_dotenv()

## City Expert
city_expert = IndoCityExpert(model="deepseek-ai/DeepSeek-V3")
input_query = {
    "cities_list": ",".join(
        [
            "Jakrta",
            "Sulawesi Utra",
            "kaltim",
            "sumut",
            "kalteng",
            "D.I Yogyakarta",
            "Purwakart",
            "Subng",
            "sukabuma",
            "taskmalaya",
        ]
    ),
}
output = city_expert.normalized_cities(**input_query)
print(output)
