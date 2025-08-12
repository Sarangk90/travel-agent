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


class FlightType(IntEnum):
    """Enum for flight types supported by Google Flights API"""

    ROUND_TRIP = 1
    ONE_WAY = 2


class FlightsInput(BaseModel):
    departure_airport: str = Field(description="Departure airport code (IATA)")
    arrival_airport: str = Field(description="Arrival airport code (IATA)")
    outbound_date: str = Field(
        description="Parameter defines the outbound date. The format is YYYY-MM-DD. e.g. 2024-06-22"
    )
    return_date: str = Field(
        description="Parameter defines the return date. The format is YYYY-MM-DD. e.g. 2024-06-28. Leave empty for one-way flights.",
    )
    adults: int = Field(
        description="Parameter defines the number of adults. Default to 1.", ge=1
    )
    children: int = Field(
        description="Parameter defines the number of children. Default to 0.", ge=0
    )
    infants_in_seat: int = Field(
        description="Parameter defines the number of infants in seat. Default to 0.",
        ge=0,
    )
    infants_on_lap: int = Field(
        description="Parameter defines the number of infants on lap. Default to 0.",
        ge=0,
    )
    type: FlightType = Field(
        description="Parameter defines the type of the flights: 1 - Round trip (default), 2 - One way",
    )
    stops: str = Field(
        description="Parameter defines the maximum number of stops. Format: comma-separated values (0,1,2,3)",
    )
    currency: str = Field(description="Currency for pricing.")

    @field_validator("departure_airport", "arrival_airport")
    @classmethod
    def validate_airport_code(cls, v):
        if not v or not isinstance(v, str) or len(v) != 3:
            raise ValueError("Airport code must be a 3-letter IATA code")
        return v.upper()

    @field_validator("outbound_date", "return_date")
    @classmethod
    def validate_date_format(cls, v):
        if not v or v == "":
            return v

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

    @model_validator(mode="after")
    def validate_flight_consistency(self):
        flight_type = self.type
        return_date = self.return_date
        outbound_date = self.outbound_date

        if flight_type == FlightType.ONE_WAY and return_date and return_date != "":
            raise ValueError("Return date should not be provided for one-way flights")

        if flight_type == FlightType.ROUND_TRIP and (not return_date or return_date == ""):
            raise ValueError("Return date is required for round-trip flights")

        # Verify return date is after outbound date for round trips
        if flight_type == FlightType.ROUND_TRIP and outbound_date and return_date and return_date != "":
            outbound = datetime.datetime.strptime(outbound_date, "%Y-%m-%d").date()
            return_d = datetime.datetime.strptime(return_date, "%Y-%m-%d").date()

            if return_d < outbound:
                raise ValueError(
                    f"Return date {return_date} must be after outbound date {outbound_date}"
                )

        return self


class FlightsInputSchema(BaseModel):
    params: FlightsInput


@tool(args_schema=FlightsInputSchema)
def find_flights(params: FlightsInput) -> Dict[str, Any]:
    """
    Find flights using the Google Flights engine via SerpAPI.
    For round trips, fetches both initial search results and complete itinerary details.

    This tool handles:
    1. Initial flight search to get options and prices
    2. For round-trips, performs follow-up queries to get complete itineraries
    3. Also makes a separate return flight search to ensure comprehensive data

    Returns a structured dictionary with all available flight information.
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY environment variable is not set")

    results = {
        "initial_search": None,  # Initial search results with outbound options
        "complete_itineraries": [],  # Complete itineraries from token-based follow-ups
        "return_flight_search": None,  # Separate search for return flights (as backup)
        "error": None,
    }

    try:
        # 1. Initial search to get flight options and tokens
        search_params = {
            "api_key": api_key,
            "engine": "google_flights",
            "hl": "en",
            "gl": "us",
            "departure_id": params.departure_airport,
            "arrival_id": params.arrival_airport,
            "outbound_date": params.outbound_date,
            "currency": params.currency,
            "adults": params.adults,
            "infants_in_seat": params.infants_in_seat,
            "stops": params.stops,
            "infants_on_lap": params.infants_on_lap,
            "children": params.children,
            "type": str(int(params.type)),
        }

        if params.type == FlightType.ROUND_TRIP and params.return_date and params.return_date != "":
            search_params["return_date"] = params.return_date

        # Execute the initial search
        initial_results = serpapi.search(search_params).data
        results["initial_search"] = initial_results

        # 2. For round trips, fetch complete itineraries using tokens
        if params.type == FlightType.ROUND_TRIP:
            # Extract tokens from best flights and other flights (limit to top 5 total for efficiency)
            tokens = []

            # First from best flights
            for flight in initial_results.get("best_flights", [])[:3]:
                if "departure_token" in flight:
                    tokens.append(flight["departure_token"])

            # Then from other flights if needed
            if len(tokens) < 5 and "other_flights" in initial_results:
                remaining_slots = 5 - len(tokens)
                for flight in initial_results.get("other_flights", [])[
                    :remaining_slots
                ]:
                    if "departure_token" in flight:
                        tokens.append(flight["departure_token"])

            # Get complete itineraries using tokens
            for i, token in enumerate(tokens):
                token_params = {
                    "api_key": api_key,
                    "engine": "google_flights",
                    "hl": "en",
                    "gl": "us",
                    "departure_token": token,
                }
                try:
                    token_result = serpapi.search(token_params).data
                    # Add reference to the original flight in initial results
                    if "best_flights" in initial_results and i < len(
                        initial_results.get("best_flights", [])
                    ):
                        token_result["original_flight_info"] = initial_results[
                            "best_flights"
                        ][i]
                    results["complete_itineraries"].append(token_result)
                except Exception as e:
                    # Continue with other tokens if one fails
                    continue

            # 3. As a backup, make a separate search for return flights
            return_params = {
                "api_key": api_key,
                "engine": "google_flights",
                "hl": "en",
                "gl": "us",
                "departure_id": params.arrival_airport,  # Swap airports for return
                "arrival_id": params.departure_airport,
                "outbound_date": params.return_date,  # Use return date as outbound
                "currency": params.currency,
                "adults": params.adults,
                "infants_in_seat": params.infants_in_seat,
                "stops": params.stops,
                "infants_on_lap": params.infants_on_lap,
                "children": params.children,
                "type": "2",  # One-way for return leg
            }

            try:
                return_results = serpapi.search(return_params).data
                results["return_flight_search"] = return_results
            except Exception as e:
                # If return search fails, continue with what we have
                pass

        return results

    except serpapi.exceptions.SerpApiError as e:
        results["error"] = f"SerpAPI error: {str(e)}"
        return results
    except Exception as e:
        results["error"] = f"An unexpected error occurred: {str(e)}"
        return results


model = ChatOpenAI(model="gpt-4o-2024-08-06")

flights_advisor_tools = [
    find_flights,
    make_handoff_tool(agent_name="supervisor"),
]

flights_advisor = create_react_agent(
    model=model.bind_tools(
        flights_advisor_tools, parallel_tool_calls=False, strict=False
    ),
    tools=flights_advisor_tools,
    prompt=(
        "# Flight Expert Assistant\n\n"
        f"You are an expert flight advisor specialized in searching and recommending optimal flight options. Today is {datetime.datetime.now().strftime('%Y-%m-%d')}.\n\n"
        "## Your Capabilities\n"
        "- Search for flights between airports using IATA codes\n"
        "- Analyze comprehensive flight data including prices, schedules, airlines, and layovers\n"
        "- Provide personalized recommendations based on user preferences\n\n"
        "- If the question is not related to flights, transfer to a supervisor\n"
        "## Guidelines for Flight Recommendations\n"
        "1. **Always provide complete information**:\n"
        "   - Flight prices with currency\n"
        "   - Airlines and flight numbers\n"
        "   - Departure and arrival times (with timezone information where available)\n"
        "   - Duration of flights and layovers\n"
        "   - Number of stops\n\n"
        "2. **Search Pattern Analysis**:\n"
        "   - For round trips: Provide both outbound and return flight options\n"
        "   - For one-way trips: Provide only the relevant flight leg\n"
        "   - If a user asks about a 'return flight' only (e.g., from destination back to origin), interpret this as a one-way flight search\n\n"
        "3. **Data Processing**:\n"
        "   - First review the initial search results for overall options\n"
        "   - For round trips:\n"
        "     * Examine the complete itineraries data for detailed outbound + return combinations\n"
        "     * If available, use the return_flight_search data to supplement information\n"
        "     * Present complete round-trip itineraries showing both outbound and return flights\n"
        "   - For one-way trips, analyze available flight options directly\n"
        "   - Compare options across different metrics (price, duration, convenience)\n"
        "   - Highlight notable features (e.g., direct flights, significant savings, premium options)\n\n"
        "4. **Handling Round-Trip Data**:\n"
        "   - Always present BOTH the outbound AND return flights for round trips\n"
        "   - First check the complete_itineraries data which contains paired outbound and return options\n"
        "   - If complete_itineraries is insufficient, combine data from initial_search and return_flight_search\n"
        "   - When presenting combinations, clearly label outbound and return flights\n\n"
        "## Conversation Flow\n"
        "1. **Before making tool calls**:\n"
        "   - Thoroughly understand the user's request\n"
        "   - Ask for clarification if airport codes, dates, or other critical information is missing\n\n"
        "2. **After receiving search results**:\n"
        "   - Provide a concise summary of the top options\n"
        "   - Explain your recommendations with clear reasoning\n"
        "   - Format information in an easily readable way\n\n"
        "3. **For non-flight inquiries**:\n"
        # "   - Explain that you specialize in flight information\n"
        # "   - Ask if the user would like to be transferred to a supervisor for other assistance\n"
        # "   - Only transfer after receiving confirmation from the user\n"
        "   - Do NOT answer any non-flight related enquiry and offer to transfer immediately\n\n"
        "## Common Scenarios\n"
        "- If no flights match the criteria: Suggest alternative dates or airports\n"
        "- If flights are expensive: Mention factors affecting price and possible alternatives\n"
        "- If search parameters are ambiguous: Ask clarifying questions before searching\n\n"
        "Always aim to be helpful, accurate, and focused on finding the best flight options for the user's specific needs."
    ),
    name="flights_advisor",
)


def call_flights_advisor(
    state: MessagesState,
) -> Command[Literal["supervisor", "human"]]:
    response = flights_advisor.invoke(state)
    return Command(update=response, goto="human")
