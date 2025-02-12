from typing import List, Union

from pydantic import BaseModel, Field


class Ragoutput(BaseModel):
    """RAG Styles

    Args:
        BaseModel (_type_): _description_
    """

    context: Union[List[str], str] = Field(
        description=(
            "Make the string context as list of the context. "
            "Let it be an empty list if there is no context"
        )
    )
    answer: str = Field(
        description="summary of the answer which is not more than 10 sentences maximum"
    )


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


class CityList(BaseModel):
    """List of cities"""

    city: str = Field(description="Original city name")
    suggestions: List[str] = Field(description="List of suggested cities name")


class IndoCityOutput(BaseModel):
    """RAG Styles"""

    cities: List[CityList] = Field(
        description="List of closest real cities name in Indonesia"
    )
