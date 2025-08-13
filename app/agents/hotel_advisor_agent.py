import datetime
import os
from enum import IntEnum
from typing import Optional, Literal, Dict, Any

import serpapi
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator, model_validator

from app.tools.handoff_tool import make_handoff_tool


class SortingOptions(IntEnum):
    """Enum for hotel sorting options supported by Google Hotels API"""

    PRICE_LOW_TO_HIGH = 1
    PRICE_HIGH_TO_LOW = 2
    RATING_HIGH_TO_LOW = 8
    POPULARITY = 16


class HotelsInput(BaseModel):
    q: str = Field(
        description="Location of the hotel (city, area, or specific hotel name)"
    )
    check_in_date: str = Field(
        description="Check-in date. The format is YYYY-MM-DD. e.g. 2024-06-22"
    )
    check_out_date: str = Field(
        description="Check-out date. The format is YYYY-MM-DD. e.g. 2024-06-28"
    )
    sort_by: SortingOptions = Field(
        description="Parameter is used for sorting the results. Default is sort by highest rating (8). "
        "Options: 1 (price low to high), 2 (price high to low), "
        "8 (rating high to low), 16 (popularity)",
        default=SortingOptions.RATING_HIGH_TO_LOW,
    )
    adults: int = Field(description="Number of adults. Default to 1.", ge=1)
    children: int = Field(description="Number of children. Default to 0.", ge=0)
    rooms: int = Field(description="Number of rooms. Default to 1.", ge=1)
    hotel_class: str = Field(
        description='Parameter defines to include only certain hotel class in the results. Format: comma-separated values (e.g. "2,3,4" for 2-4 star hotels). Leave empty for all hotel classes.',
    )
    currency: str = Field(description="Currency for pricing.")

    @field_validator("check_in_date", "check_out_date")
    @classmethod
    def validate_date_format(cls, v):
        if not v:
            raise ValueError("Date must be provided")

        try:
            date = datetime.datetime.strptime(v, "%Y-%m-%d").date()
            today = datetime.date.today()

            if date < today:
                raise ValueError(f"Date {v} is in the past")

            return v
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError(f"Date must be in YYYY-MM-DD format, got {v}")
            raise e

    @field_validator("hotel_class")
    @classmethod
    def validate_hotel_class(cls, v):
        if not v or v == "":
            return v

        classes = v.split(",")
        try:
            for hotel_class in classes:
                class_int = int(hotel_class.strip())
                if class_int < 1 or class_int > 5:
                    raise ValueError(
                        f"Hotel class must be between 1 and 5, got {hotel_class}"
                    )
        except ValueError:
            raise ValueError(f"Hotel class must be comma-separated numbers, got {v}")

        return v

    @model_validator(mode="after")
    def validate_dates_consistency(self):
        check_in = datetime.datetime.strptime(self.check_in_date, "%Y-%m-%d").date()
        check_out = datetime.datetime.strptime(self.check_out_date, "%Y-%m-%d").date()

        if check_out <= check_in:
            raise ValueError(
                f"Check-out date {self.check_out_date} must be after check-in date {self.check_in_date}"
            )

        return self


class HotelsInputSchema(BaseModel):
    params: HotelsInput


@tool(args_schema=HotelsInputSchema)
def get_hotel_recommendations(params: HotelsInput) -> Dict[str, Any]:
    """
    Find hotels using the Google Hotels engine via SerpAPI.

    Parameters:
        params (HotelsInput): Hotel search parameters including location, check-in/out dates, and room requirements.

    Returns:
        dict: Complete hotel search results including property details, amenities, and pricing information.

    Note:
        This tool requires a valid SERPAPI_API_KEY environment variable.
    """

    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY environment variable is not set")

    search_params = {
        "api_key": api_key,
        "engine": "google_hotels",
        "hl": "en",
        "gl": "us",
        "q": params.q,
        "check_in_date": params.check_in_date,
        "check_out_date": params.check_out_date,
        "currency": params.currency,
        "adults": params.adults,
        "children": params.children,
        "rooms": params.rooms,
        "sort_by": str(int(params.sort_by)),
    }

    # Add hotel_class only if it's provided
    if params.hotel_class and params.hotel_class != "":
        search_params["hotel_class"] = params.hotel_class

    try:
        search = serpapi.search(search_params)
        return search.data.get("properties", {"error": "No properties found"})
    except serpapi.exceptions.SerpApiError as e:
        print(e)
        return {"error": f"SerpAPI error: {str(e)}"}
    except Exception as e:
        print(e)
        return {"error": f"An unexpected error occurred: {str(e)}"}


model = ChatOpenAI(model="gpt-4o-2024-08-06")

# Define hotel advisor tools and ReAct agent
hotel_advisor_tools = [
    get_hotel_recommendations,
    make_handoff_tool(agent_name="supervisor"),
]
hotel_advisor = create_react_agent(
    model=model.bind_tools(hotel_advisor_tools, parallel_tool_calls=False, strict=False),
    tools=hotel_advisor_tools,
    prompt=(
        "# Hotel Expert Assistant\n\n"
        f"You are an expert hotel advisor specialized in finding and recommending the best accommodations. Today is {datetime.datetime.now().strftime('%Y-%m-%d')}.\n\n"
        "## Your Capabilities\n"
        "- Search for hotels in any location worldwide\n"
        "- Analyze hotel options based on price, rating, amenities, and location\n"
        "- Provide personalized recommendations based on guest preferences\n\n"
        "## Guidelines for Hotel Recommendations\n"
        "1. **Always provide complete information**:\n"
        "   - Hotel name and star rating\n"
        "   - Price per night with currency\n"
        "   - Location details and proximity to attractions\n"
        "   - Guest ratings and notable reviews\n"
        "   - Key amenities (pool, spa, restaurant, free breakfast, etc.)\n\n"
        "2. **Search Pattern Analysis**:\n"
        "   - Consider the number of guests and required rooms\n"
        "   - Note the length of stay and adjust recommendations accordingly\n"
        "   - For family travel, highlight family-friendly amenities\n"
        "   - For business travel, emphasize business centers and workspace availability\n\n"
        "3. **Data Processing**:\n"
        "   - Analyze all available hotel data\n"
        "   - Compare options across different metrics (price, rating, amenities, location)\n"
        "   - Highlight exceptional value or unique features\n\n"
        "## Conversation Flow\n"
        "1. **Before making tool calls**:\n"
        "   - Thoroughly understand the user's needs and preferences\n"
        "   - Ask for clarification if location, dates, or other critical information is missing\n\n"
        "2. **After receiving search results**:\n"
        "   - Provide a concise summary of the top 3-5 options\n"
        "   - Explain your recommendations with clear reasoning\n"
        "   - Format information in an easily readable way\n\n"
        "3. **For non-hotel inquiries**:\n"
        # "   - Explain that you specialize in hotel information\n"
        # "   - Ask if the user would like to be transferred to a supervisor for other assistance\n"
        # "   - Only transfer after receiving confirmation from the user\n"
        "   - Do NOT answer any non-hotel related enquiry and transfer immediately to supervisor\n\n"
        "## Common Scenarios\n"
        "- If no hotels match the criteria: Suggest alternatives with slightly different parameters\n"
        "- If hotels are expensive: Mention factors affecting price and suggest nearby alternatives\n"
        "- If search parameters are ambiguous: Ask clarifying questions before searching\n\n"
        "Always aim to be helpful, accurate, and focused on finding the best accommodation options for the user's specific needs."
    ),
    name="hotel_advisor",
)


def call_hotel_advisor(
    state: MessagesState,
) -> Command[Literal["supervisor", "human"]]:
    response = hotel_advisor.invoke(state)
    return Command(update=response, goto="human")
