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


class ChartPosition(BaseModel):
    """position of the chart"""

    x: float = Field(description="Starting column (0-based index)")
    y: float = Field(description="Starting row (0-based index)")
    width: float = Field(description="x-axis size of the chart spans")
    height: float = Field(description="y-axis size of the chart spans")


class DashboardDetail(BaseModel):
    """Details of each dashboard context"""

    chart_id: str = Field(description="unique id of the chart")
    position: ChartPosition = Field(
        description="position of the chart. It could be None if the chart is not choosen to be shown. Make position None if the chart is not shown"
    )


class DashboardLayoutOutput(BaseModel):
    """RAG Styles"""

    chart_position: List[DashboardDetail] = Field(
        description="List of charts positions in grid."
    )


class ChartDetail(BaseModel):
    """Details of each dashboard context"""

    chart_id: str = Field(description="unique id of the chart")
    title: str = Field(description="title of the chart")
    description: str = Field(description="chart description")
    chart_type: str = Field(
        description="type of the chart. The types is either 'bar', 'line', 'numberOnly', 'textOnly', 'table', or 'maps'. No other types are allowed. Dont use maps if no longitude and latitude provided"
    )
    priority_level: int = Field(
        description="priority of the chart. The priority is between 1 to 10. The lower the number, the higher the priority. It is allowed to have same priority level"
    )


class ChartOptionsOutput(BaseModel):
    """RAG Styles"""

    chart_options: List[ChartDetail] = Field(
        description="List of charts contexts with detailed information"
    )


class CityList(BaseModel):
    """List of cities"""

    city: str = Field(description="Original city name.")
    suggestions: List[str] = Field(description="List of suggested real cities name.")


class IndoCityOutput(BaseModel):
    """RAG Styles"""

    cities: List[CityList] = Field(
        description="List of closest real cities name in Indonesia"
    )
