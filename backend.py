import os
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
import certifi
from dotenv import load_dotenv
import re
import json
import time
load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated,Any
import operator
import uuid
import asyncio
import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
# from tools.tavily_tool import tavily_search
# from tools.flight_tool import search_flights
from mcp_client import tavily_mcp_search, aviation_mcp_call, extract_destination, forecast_mcp_search, weather_mcp_search


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")


# =========================
# LLM CONFIGURATION
# =========================

FAST_MODEL = "openai/gpt-oss-20b"
ITINERARY_MODEL = "openai/gpt-oss-20b"

llm = ChatGroq(
    model=FAST_MODEL,
    api_key=GROQ_API_KEY,
    max_tokens=1000,
    max_retries=0,
)

itinerary_llm = ChatGroq(
    model=ITINERARY_MODEL,
    api_key=GROQ_API_KEY,
    max_tokens=800,
    max_retries=0,
    reasoning_effort="low",
)
# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    revision_count: int
    final_response: str

    llm_calls: int
# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}
def find_cities(text: str, cities: list[str]) -> list[str]:
    """
    Match complete city names instead of partial substrings.

    Prevents Pai from matching Spain.
    """
    text = text or ""
    matches = []

    for city in cities:
        pattern = rf"(?<!\w){re.escape(city)}(?!\w)"

        if re.search(pattern, text, re.IGNORECASE):
            matches.append(city)

    return matches


def or_na(value):
    """Return N/A for empty or missing values."""
    if value is None:
        return "N/A"

    if isinstance(value, str) and not value.strip():
        return "N/A"

    return value


def as_bool(value, default=False):
    """Safely convert boolean-like values."""

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.strip().lower()

        if value in {"true", "yes", "1", "allowed"}:
            return True

        if value in {"false", "no", "0", "blocked"}:
            return False

    if isinstance(value, (int, float)):
        return bool(value)

    return default


PROPERTY_RE = re.compile(
    r"\b(?:hotel|hostel|resort|guesthouse|guest house|"
    r"residence|inn|suites|lodge)\b",
    re.IGNORECASE
)
AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }
# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = as_bool(
            guardrail_result.get("allowed"),
            default=False
        )
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    
        
    except Exception as exc:
        print(f"Guardrail validation failed: {exc}")

        allowed = True
        guardrail_reason = (
            "Travel request validation could not be completed. "
            "Please try again."
        )

        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # Itinerary agent must always be included.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        # Do not automatically add weather_agent.
        # Weather agent should run only when selected by the supervisor.
        # Preserve the correct execution order.
        selected_agents = [
            name for name in AGENT_ORDER
            if name in selected_agents
        ]

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = ["itinerary_agent"]
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }

# =========================
# Flight Agent
# =========================

# def flight_agent(state: TravelState):
#     query = state["user_query"]
#     flight_data = search_flights(query)

#     return {
#         "flight_results": flight_data,
#         "messages": [
#             AIMessage(content="Flight results fetched.")
#         ],
#         "llm_calls": state.get("llm_calls", 0) + 1
#     }




# Flight Tool Router Prompt

FLIGHT_AGENT_PROMPT = """
You are a travel flight information specialist.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate ONLY the following information:

1. Likely departure airport, including airport name and code.
2. Likely arrival airport, including airport name and code.
3. Airlines serving this route.
4. Available flight routes and layover hubs, if known.
5. Typical flight duration.
6. Estimated round-trip airfare range, if reliable estimates are available.
7. Peak-season airfare variation, if relevant.

IMPORTANT RULES:
- Do NOT generate booking advice.
- Do NOT generate budget-saving suggestions.
- Do NOT discuss how to stay under ₹2 lakhs or any other budget.
- Do NOT generate hotel recommendations.
- Do NOT generate a trip cost breakdown.
- Do not claim that a flight is available unless availability was actually checked.
- Clearly label estimated fares as approximate.
- If information is unavailable, say so.
- Do not invent direct flights or airline routes.

Return concise flight information only.
"""



def parse_mcp_json(response):
    """Return only a dictionary from an MCP response."""

    if isinstance(response, dict):
        return response

    if isinstance(response, list):
        raw_text = "\n".join(
            item.get("text", "")
            for item in response
            if isinstance(item, dict)
            and item.get("type") == "text"
        )
    else:
        raw_text = str(response)

    try:
        data = json.loads(raw_text)

        return data if isinstance(data, dict) else None

    except (json.JSONDecodeError, TypeError):
        start = raw_text.find("{")
        end = raw_text.rfind("}")

        if start != -1 and end > start:
            try:
                data = json.loads(raw_text[start:end + 1])

                return data if isinstance(data, dict) else None

            except (json.JSONDecodeError, TypeError):
                pass

    return None

def filter_mcp_records(data, search_terms):
    """
    Extract relevant records from MCP airport/airline responses.
    Supports dictionaries, nested dictionaries and lists.
    """
    if not data:
        return []

    records = []

    def collect_records(item):
        if isinstance(item, list):
            for value in item:
                collect_records(value)

        elif isinstance(item, dict):
            nested_lists = [
                value for value in item.values()
                if isinstance(value, list)
            ]

            if nested_lists:
                for value in nested_lists:
                    collect_records(value)
            else:
                records.append(item)

    collect_records(data)

    terms = [
        str(term).strip().lower()
        for term in search_terms
        if term and str(term).strip()
    ]

    if not terms:
        return records[:20]

    matched = []

    for record in records:
        searchable = json.dumps(
            record,
            ensure_ascii=False
        ).lower()

        if any(term in searchable for term in terms):
            matched.append(record)

    return matched[:20] if matched else records[:20]


def format_weather(current_response, forecast_response, city):

    current_data = parse_mcp_json(current_response)
    forecast_data = parse_mcp_json(forecast_response)

    output = [f"### Weather in {city}"]

    if current_data:

        output.append(
            f"""
**Current Weather**

- Temperature: {or_na(current_data.get('temperature_c'))}°C
- Feels like: {or_na(current_data.get('feels_like_c'))}°C
- Humidity: {or_na(current_data.get('humidity'))}%
- Condition: {or_na(current_data.get('condition'))}
- Wind speed: {or_na(current_data.get('wind_speed'))} m/s
"""
        )

    else:
        output.append("Current weather information unavailable.")

    if forecast_data:

        forecast_items = forecast_data.get("forecast", [])

        if isinstance(forecast_items, list):

            output.append("**Available Forecast**")

            for item in forecast_items:

                if not isinstance(item, dict):
                    continue

                output.append(
                    f"- {or_na(item.get('datetime'))}: "
                    f"{or_na(item.get('temperature_c'))}°C, "
                    f"{or_na(item.get('condition'))}"
                )

    else:
        output.append("Forecast information unavailable.")

    return "\n".join(output)
# Flight Agent
def flight_agent(state: TravelState):

    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    constraints = state.get("trip_constraints", {})

    origin = str(constraints.get("origin") or "").strip()
    destination = str(
        constraints.get("destination") or ""
    ).strip()

    try:

        airports_raw = asyncio.run(
            aviation_mcp_call("list_airports")
        )

        airlines_raw = asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        airports = parse_mcp_json(airports_raw)
        airlines = parse_mcp_json(airlines_raw)

        search_terms = [
            origin,
            destination
        ]

        relevant_airports = filter_mcp_records(
            airports,
            search_terms
        )

        relevant_airlines = filter_mcp_records(
            airlines,
            search_terms
        )

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=json.dumps(
                relevant_airports,
                ensure_ascii=False
            )[:5000],
            airline_data=json.dumps(
                relevant_airlines,
                ensure_ascii=False
            )[:3000]
        )

        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = str(response.content)

    except Exception as error:

        flight_data = (
            f"Flight information unavailable: {str(error)}"
        )

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight recommendations generated.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
import re



def extract_hotel_records(search_results, city):
    hotels = []
    seen = set()

    blocked_domains = [
    "youtube.com",
    "youtu.be",
    "skyscanner.com",
    "cbinsights.com",
    "finance.yahoo.com"
]

    ignored_headings = {
        "photos",
        "reviews",
        "description",
        "overview",
        "facilities",
        "location",
        "contact",
        "website",
        "directions",
        "see all",
        "budget & hotels",
        "introduction",
        "conclusion"
    }

    property_keywords = [
        "hotel",
        "hostel",
        "resort",
        "guesthouse",
        "guest house",
        "residence",
        "inn",
        "suites",
        "lodge"
    ]

    for source in search_results:

        title = source.get("title", "")
        url = source.get("url", "")
        content = source.get("content", "")

        url_lower = url.lower()

        # Reject irrelevant sources.
        if any(domain in url_lower for domain in blocked_domains):
            continue

        candidates = []

        # Extract actual property names from page titles.
        title_name = re.split(
            r",\s*(?=\d|Chiang Mai|Phuket|Bangkok)",
            title,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0].strip()

        if any(
            keyword in title_name.lower()
            for keyword in property_keywords
        ):
            candidates.append(title_name)

        # Extract Markdown headings from page content.
        headings = re.findall(
            r"^#{2,4}\s+(?:\d+[\.\)]\s*)?(.+?)\s*$",
            content,
            flags=re.MULTILINE
        )

        candidates.extend(headings)

        for name in candidates:

            name = re.sub(r"\s+", " ", name).strip()
            name = re.sub(r"\*\*|__", "", name)
            name = name.strip(" -*#")

            # Remove descriptive suffixes.
            name = re.sub(
                r",\s*(?:social|eco[\s\-‑]*friendly|budget|"
                r"affordable|luxury|family[\s\-‑]*friendly).*$",
                "",
                name,
                flags=re.IGNORECASE
            )
            name = re.sub(
                r"\s+[—–-]\s+(?:best|budget|cheap|luxury).*$",
                "",
                name,
                flags=re.IGNORECASE
            )

            # Remove appended addresses.
            name = re.sub(
                r"\s+\d{1,5}/.*$",
                "",
                name
            ).strip()

            name_lower = name.lower()

            # Ignore generic headings.
            if name_lower in ignored_headings:
                continue

            if name_lower.startswith((
                "channel:",
                "reviews image",
                "image ",
                "best places to stay",
                "where to stay",
                "ver todos",
                "day ",
                "chapter ",
                "section "
            )):
                continue

            if any(
                phrase in name_lower
                for phrase in [
                    "alternatives and competitors",
                    "hotels in ",
                    "hostels in ",
                    "best budget hotels",
                    "cheap hotels",
                    "luxury hotels",
                    "budget hostels",
                    "best places to stay"
                ]
            ):
                continue

            if (
                len(name) < 4
                or len(name) > 75
                or "$" in name
                or "%" in name
                or name.endswith(":")
            ):
                continue

            # Only accept headings that look like property names.
            # Page titles with a property keyword are also accepted.
            if not any(
                keyword in name_lower
                for keyword in property_keywords
            ):
                continue

            key = name_lower

            if key in seen:
                continue

            seen.add(key)

            hotels.append({
                "hotel_name": name,
                "location": city,
                "source_title": title,
                "source_url": url
            })

    return hotels


# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):

    print("\nINSIDE HOTEL AGENT\n")

    query = state["user_query"]

    try:

        # --------------------------------
        # 1. Identify destinations
        # --------------------------------

        cities = [
            "Bangkok",
            "Chiang Mai",
            "Pai",
            "Phuket",
            "Krabi"
        ]

        selected_cities = [
            city for city in cities
            if city.lower() in query.lower()
        ]

        if selected_cities:
            destinations = selected_cities

        elif "thailand" in query.lower():
            destinations = [
                "Bangkok",
                "Chiang Mai",
                "Phuket"
            ]

        else:
            destination = extract_destination(query)

            if not destination:
                raise ValueError(
                    "Could not identify the travel destination."
                )

            destinations = [destination]

        all_hotel_results = []

        # --------------------------------
        # 2. Search each destination
        # --------------------------------

        for destination in destinations:

            print(f"\nSearching hotels in {destination}...")

            search_queries = [
                (
                    f"Best budget hotels and hostels in {destination}. "
                    "List individual hotel property names, not categories. "
                    "Include hotel names and source links."
                ),
                (
                    f"Individual hotel names in {destination} "
                    "budget accommodation Booking.com Agoda "
                    "hostels guesthouses"
                ),
                (
                    f"Top rated affordable places to stay in {destination}. "
                    "Hotel property names and hostel names."
                )
            ]

            search_results = []

            # --------------------------------
            # 3. Tavily search with retries
            # --------------------------------

            for attempt, search_query in enumerate(
                search_queries,
                start=1
            ):

                print(
                    f"Search attempt {attempt} for {destination}"
                )

                try:

                    raw_results = asyncio.run(
                        tavily_mcp_search(search_query)
                    )

                    data = parse_mcp_json(raw_results)

                    if isinstance(data, dict):

                        results = data.get("results", [])

                        if results:
                            search_results.extend(results)

                            print(
                                f"Found {len(results)} results."
                            )

                        else:
                            print("No results found.")

                    else:
                        print("Could not parse Tavily response.")

                except Exception as search_error:

                    print(
                        f"Search attempt {attempt} failed: "
                        f"{search_error}"
                    )

                # Stop retrying once results are found.
                if search_results:
                    break

            # --------------------------------
            # 4. Check search results
            # --------------------------------

            if not search_results:

                all_hotel_results.append(
                    f"\n### {destination}\n"
                    "No hotel search results found after retries."
                )

                continue

            print(
                f"\nTotal search results for {destination}: "
                f"{len(search_results)}"
            )

            # --------------------------------
            # 5. Extract hotel properties
            # --------------------------------

            hotels = extract_hotel_records(
                search_results,
                destination
            )

            print(
                f"Extracted {len(hotels)} hotel names "
                f"for {destination}"
            )

            if not hotels:

                all_hotel_results.append(
                    f"\n### {destination}\n"
                    "Search results were found, but no individual "
                    "hotel property names could be extracted."
                )

                continue

            # --------------------------------
            # 6. Format hotel results
            # --------------------------------

            formatted = []

            for index, hotel in enumerate(
                hotels[:6],
                start=1
            ):

                formatted.append(
                    f"""
**{index}. {hotel['hotel_name']}**

- Location: {hotel['location']}
- Source: {hotel['source_title']}
- Source page: {hotel['source_url']}
"""
                )

            city_results = (
                f"\n### Hotel Suggestions in {destination}\n"
                + "\n".join(formatted)
            )

            all_hotel_results.append(city_results)

        # --------------------------------
        # 7. Final hotel response
        # --------------------------------

        hotel_results = (
            "# Budget Hotel Search Results\n"
            + "\n".join(all_hotel_results)
            + """

IMPORTANT:
- These are hotel suggestions extracted from web search results.
- They are not confirmed reservations.
- Hotel names are extracted from source-page content.
- Search results do not guarantee that a property is currently available.
- Prices and room availability must be checked separately.
- Source pages may contain multiple properties and may not be
  direct hotel booking links.
"""
        )

    except Exception as error:

        hotel_results = (
            f"Hotel information unavailable: {str(error)}"
        )

    print("\nHOTEL AGENT RESULTS:\n")
    print(hotel_results)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content=hotel_results)
        ],
        "llm_calls": state.get("llm_calls", 0)
    }


# =========================
# Weather Agent
# =========================


def weather_agent(state: TravelState):

    print("\nINSIDE WEATHER AGENT\n")

    query = state["user_query"].lower()

    cities = []

    for city in ["Bangkok", "Chiang Mai", "Pai", "Phuket", "Krabi"]:
        if city.lower() in query:
            cities.append(city)

    if not cities and "thailand" in query:
        cities = ["Bangkok", "Chiang Mai", "Pai"]

    if not cities:
        city = extract_destination(state["user_query"])
        cities = [city]

    weather_sections = []

    for city in cities:

        print(f"Fetching weather for {city}...")

        try:
            current = asyncio.run(weather_mcp_search(city))
            forecast = asyncio.run(forecast_mcp_search(city))

            formatted_weather = format_weather(
                current,
                forecast,
                city
            )

            weather_sections.append(formatted_weather)

        except Exception as e:
            weather_sections.append(
                f"Weather unavailable for {city}: {str(e)}"
            )

    weather_results = "\n\n".join(weather_sections)

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(content=weather_results)
        ],
        "llm_calls": state.get("llm_calls", 0)
    }
# =========================
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent
# =========================
# i have to add all these functions because i have free tier groq model which which has a limit of tokens so i have to split the itinerary agent into multiple functions and call them one by one to get the final result

 
REQUIRED_SECTIONS = ["Morning", "Afternoon", "Evening", "Food", "Transport"]
 
# Tolerant heading pattern: "### Day 3", "### **Day 3: Title**", "## Day 3 -"
DAY_HEADING_RE = re.compile(
    r"^#{1,4}[ \t]*\**[ \t]*Day[ \t]*(\d+)\b",
    re.IGNORECASE | re.MULTILINE,
)
 
 
def extract_total_days(
    query: str,
    trip_constraints: dict | None = None
) -> int:

    q = (query or "").lower()

    constraints = trip_constraints or {}

    # 1. Prefer the supervisor's duration.
    duration = str(
        constraints.get("duration") or ""
    ).strip().lower()

    def parse_duration(value):

        if not value:
            return None

        # 7D/6N
        match = re.search(
            r"\b(\d+)\s*d\s*/\s*\d+\s*n\b",
            value
        )

        if match:
            return int(match.group(1))

        # 10 nights
        match = re.search(
            r"\b(\d+)\s*nights?\b",
            value
        )

        if match:
            return int(match.group(1)) + 1

        # 7 days / 7-day trip
        match = re.search(
            r"\b(\d+)\s*-?\s*days?\b",
            value
        )

        if match:
            return int(match.group(1))

        # 2 weeks
        match = re.search(
            r"\b(\d+)\s*-?\s*weeks?\b",
            value
        )

        if match:
            return int(match.group(1)) * 7

        # One week
        word_numbers = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4
        }

        for word, number in word_numbers.items():

            if re.search(
                rf"\b{word}\s+weeks?\b",
                value
            ):
                return number * 7

        if re.search(r"\bweekend\b", value):
            return 2

        if re.search(r"\ba\s+week\b", value):
            return 7

        return None

    # Prefer supervisor duration.
    total = parse_duration(duration)

    if total:
        return min(total, 30)

    # 2. Explicit query-level duration.
    total = parse_duration(q)

    if total:
        return min(total, 30)

    # 3. Sum separate destination durations.
    day_matches = re.findall(
        r"\b(\d+)\s+days?\s+in\s+"
        r"[a-z][a-z\s-]*",
        q
    )

    if day_matches:

        total = sum(
            int(number)
            for number in day_matches
        )

        return min(total, 30)

    # 4. Default duration.
    return 7
 
 
def normalize_day_text(text: str, day: int) -> str:
    """
    If the model wrote the heading without '#' (e.g. '**Day 4: Hakone**'),
    add '### ' so the heading regex can find it.
    """
    text = text.strip()
    first_line, _, rest = text.partition("\n")
 
    if re.match(rf"^\s*\**\s*Day\s*{day}\b", first_line, re.IGNORECASE):
        first_line = "### " + first_line.lstrip("# ").lstrip()
 
    return first_line + ("\n" + rest if rest else "")
 
 

def check_day(text: str, day: int) -> list:
    problems = []

    headings = [
        int(number)
        for number in DAY_HEADING_RE.findall(text)
    ]

    if not headings:
        problems.append("No Day heading found")

    elif headings != [day]:
        problems.append(
            f"Expected only Day {day}, found {headings}"
        )

    for section in REQUIRED_SECTIONS:

        # Accept both **Morning:** and **Morning**:
        pattern = (
            rf"^[ \t>#*_\-•\d.)]*"
            rf"(?:\*\*)?{section}"
            rf"(?::\*\*|\*\*)?"
            rf"[ \t]*:?[ \t]*"
        )

        if not re.search(
            pattern,
            text,
            re.IGNORECASE | re.MULTILINE
        ):
            problems.append(
                f"Missing section: {section}"
            )

    # Validate Transport content separately.
    transport_match = re.search(
        r"^[ \t>#*_\-•\d.)]*"
        r"(?:\*\*)?Transport(?::\*\*|\*\*)?"
        r"[ \t]*:?[ \t]*(.*)$",
        text,
        re.IGNORECASE | re.MULTILINE
    )

    if not transport_match:
        problems.append("Missing Transport section")

    else:
        inline_content = transport_match.group(1).strip()

        remaining_text = text[
            transport_match.end():
        ].strip()

        if len(inline_content + remaining_text) < 20:
            problems.append(
                "Transport section is empty or too short"
            )

    return problems

def day_is_complete(text: str, day: int) -> bool:
    return not check_day(text, day)
 
def summarize_previous_days(previous_days: list) -> str:
    """
    Keep only previously completed activities.
    Avoid sending entire previous-day itineraries to Groq.
    """
    activities = []

    for day_text in previous_days:
        for line in day_text.splitlines():
            line = line.strip()

            if line.startswith(("- ", "* ")):
                activities.append(line[:120])

    if not activities:
        return "No previous activities. This is the first day."

    return "\n".join(activities[-12:])

def generate_day(
    day: int,
    total_days: int,
    common_context: str,
    previous_days: list,
    max_attempts: int = 3,
):
    previous = summarize_previous_days(previous_days)

    base_prompt = f"""
{common_context}

ACTIVITIES ALREADY PLANNED:
{previous}

TASK:
Generate ONLY Day {day} of {total_days}.

Use exactly this format:

### Day {day}: [Title]

**Morning:**
- 2 short activities.

**Afternoon:**
- 2 short activities.

**Evening:**
- 2 short activities.

**Food:**
- 2 local food recommendations.

**Transport:**
- 2 concise sentences explaining transport and approximate travel times.

RULES:
- Generate exactly one day.
- Do not generate other days.
- Do not repeat activities already listed.
- Keep the response concise, preferably under 250 words.
- Use realistic information.
- Do not invent confirmed bookings.
- Do not add a budget table.
- Complete all five sections.
"""

    calls_used = 0
    last_text = ""
    feedback = ""

    for attempt in range(1, max_attempts + 1):

        try:
            calls_used += 1

            response = itinerary_llm.invoke([
                SystemMessage(
                    content=(
                        "You are a professional travel planner. "
                        "Generate concise and complete daily itineraries."
                    )
                ),
                HumanMessage(
                    content=base_prompt + feedback
                ),
            ])

            text = normalize_day_text(
                str(response.content),
                day
            )

            finish = response.response_metadata.get(
                "finish_reason"
            )

            problems = check_day(text, day)

            print(
                f"Day {day}, Attempt {attempt}: "
                f"finish={finish}, "
                f"problems={problems or 'none'}"
            )

            last_text = text

            if finish != "length" and not problems:
                return text, calls_used

            if finish == "length":
                feedback = """
Your previous response was truncated.
Use shorter bullet points.
Complete all five sections.
Do not generate additional days.
"""

            else:
                feedback = (
                    "\n\nFix these problems in your previous response: "
                    + "; ".join(problems)
                    + ". Return the complete day only."
                )

        except Exception as error:

            error_text = str(error)
            error_lower = error_text.lower()

            print(
                f"Day {day}, Attempt {attempt} failed: {error}"
            )

            if attempt >= max_attempts:
                break

            # Handle Groq rate limits separately.
            if (
                "429" in error_text
                or "rate_limit" in error_lower
                or "tokens per minute" in error_lower
            ):

                retry_match = re.search(
                    r"try again in\s+([\d.]+)s",
                    error_text,
                    re.IGNORECASE
                )

                if retry_match:
                    wait_seconds = float(retry_match.group(1)) + 2
                else:
                    wait_seconds = 20

                print(
                    f"Groq rate limit reached. "
                    f"Waiting {wait_seconds:.1f} seconds..."
                )

                time.sleep(wait_seconds)

            else:
                time.sleep(2 * attempt)

    raise ValueError(
        f"Day {day} generation failed after {calls_used} attempts. "
        f"Last output:\n{last_text[-500:]}"
    )
 
def validate_itinerary_days(itinerary, total_days):
    """Every day must appear exactly once, in order."""
    found = [int(d) for d in DAY_HEADING_RE.findall(itinerary)]
    expected = list(range(1, total_days + 1))
 
    return {
        "valid": found == expected,  # catches duplicates and wrong order
        "found_days": found,
        "missing_days": sorted(set(expected) - set(found)),
        "extra_days": sorted(set(found) - set(expected)),
    }
 
 
def itinerary_agent(state: TravelState):

    query = state["user_query"]

    # --------------------------------
    # 1. Extract trip duration
    # --------------------------------

    total_days = extract_total_days(
        query,
        state.get("trip_constraints")
    )

    print(f"\nTrip length: {total_days} days")

    # --------------------------------
    # 2. Prepare compact common context
    # --------------------------------
    selected_agents = state.get("selected_agents", [])

    weather_was_selected = "weather_agent" in selected_agents

    weather_info = (
        state.get("weather_results", "")
        if weather_was_selected
        else ""
    )

    weather_section = (
        f"""
    ## 4. Weather Information

    {weather_info}
    """
        if weather_was_selected
        else ""
    )
    

    common_context = f"""
You are an expert international travel planner.

USER REQUEST:
{query[:800]}

FLIGHT INFORMATION:
{state.get('flight_results', '')[:350]}

HOTEL INFORMATION:
{state.get('hotel_results', '')[:350]}

WEATHER INFORMATION:
{weather_info[:250] if weather_was_selected else "Not requested. Do not generate weather information."}

- If no weather data is available, do not invent temperatures,
  rainfall, forecasts, or weather conditions.
- Do not generate a weather forecast from general knowledge.

BUDGET INFORMATION:
{state.get('budget_results', '')[:350]}

PLANNING RULES:
- Follow the requested destination and duration.
- Make the itinerary practical and budget-friendly.
- Include realistic transport times.
- Consider weather conditions only when actual weather data is available.
- Do not invent confirmed bookings.
- Keep each day concise.
"""

    # --------------------------------
    # 3. Generate one day at a time
    # --------------------------------

    itinerary_parts = []
    total_llm_calls = 0

    for day in range(1, total_days + 1):

        print(f"\nGenerating Day {day}/{total_days}...")

        day_text, calls_used = generate_day(
            day=day,
            total_days=total_days,
            common_context=common_context,
            previous_days=itinerary_parts,
        )

        itinerary_parts.append(day_text)
        total_llm_calls += calls_used

        # Allow the TPM window to recover between successful days.
        if day < total_days:
            print("Waiting 15 seconds before generating the next day...")
            time.sleep(15)

    # --------------------------------
    # 4. Assemble itinerary
    # --------------------------------

    daily_itinerary = "\n\n".join(itinerary_parts)

    # --------------------------------
    # 5. Validate all days
    # --------------------------------

    validation = validate_itinerary_days(
        daily_itinerary,
        total_days
    )

    print("\nITINERARY DAY VALIDATION")
    print(validation)

    if not validation["valid"]:
        raise ValueError(
            f"Incomplete itinerary! "
            f"Found days: {validation['found_days']}. "
            f"Missing days: {validation['missing_days']}. "
            f"Extra days: {validation['extra_days']}."
        )

    print("SUCCESS: All itinerary days are present and complete.")

    # --------------------------------
    # 6. Get budget report
    # --------------------------------

    budget_report = state.get(
        "budget_results",
        "Budget analysis is not available."
    )

    # --------------------------------
    # 7. Create complete itinerary
    # --------------------------------

    complete_itinerary = f"""
# Complete {total_days}-Day Travel Itinerary

## 1. Trip Summary

{query}

## 2. Flight Information

{state.get('flight_results', 'Not available')}

## 3. Hotel Suggestions

{state.get('hotel_results', 'Not available')}



{weather_section}

## 5. Day-by-Day Itinerary

{daily_itinerary}

## 6. Budget Analysis

{budget_report}

## 7. Final Travel Tips

- Check visa and entry requirements.
- Confirm flight and hotel availability.
- Check the latest weather forecast.
- Carry essential travel documents.
- Keep extra money for unexpected expenses.
"""

    # --------------------------------
    # 8. Human approval request
    # --------------------------------

    approval_request = (
        "Please review the generated draft itinerary. "
        "Approve it to create the final polished plan, "
        "or provide feedback for revision."
    )

    return {
        "itinerary": complete_itinerary,
        "approval_request": approval_request,
        "messages": [
            AIMessage(content=complete_itinerary)
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + total_llm_calls
        ),
    }
 
# =========================
# Human-in-the-Loop approval
# =========================

def human_approval_agent(state: TravelState):

    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get(
                "supervisor_reasoning", ""
            ),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    if not isinstance(review, dict):
        review = {}

    approved = as_bool(
        review.get("approved"),
        default=False
    )

    human_feedback = str(
        review.get("feedback") or ""
    ).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [
            AIMessage(
                content=(
                    "Itinerary approved."
                    if approved
                    else "Itinerary revision requested."
                )
            )
        ],
    }


# =========================
# Final Response Agent - original format kept, HITL feedback added
# =========================

def update_itinerary_section(
    draft: str,
    section_number: int,
    section_title: str,
    new_content: str,
    insert_before: int | None = None,
) -> str:
    """
    Replace an existing numbered section or insert it
    before another section without modifying other content.
    """

    heading = (
        f"## {section_number}. {section_title}"
    )

    pattern = re.compile(
        rf"^##\s*{section_number}\.\s*"
        rf"{re.escape(section_title)}\s*$"
        rf".*?"
        rf"(?=^##\s*\d+\.|\Z)",
        re.MULTILINE | re.DOTALL
    )

    replacement = (
        f"{heading}\n\n"
        f"{new_content.strip()}\n\n"
    )

    if pattern.search(draft):

        return pattern.sub(
            replacement,
            draft,
            count=1
        )

    # Insert a missing section before the specified section.
    if insert_before is not None:

        anchor = re.compile(
            rf"(?=^##\s*{insert_before}\.)",
            re.MULTILINE
        )

        if anchor.search(draft):

            return anchor.sub(
                replacement,
                draft,
                count=1
            )

    return draft.rstrip() + "\n\n" + replacement


def revision_agent(state: TravelState):

    feedback = state.get("human_feedback", "").strip()

    if not feedback:
        feedback = "Please improve the itinerary."

    draft = state.get("itinerary", "")

    total_days = extract_total_days(
        state["user_query"],
        state.get("trip_constraints")
    )

    # --------------------------------
    # 1. Validate original itinerary
    # --------------------------------

    validation = validate_itinerary_days(
        draft,
        total_days
    )

    if not validation["valid"]:
        raise ValueError(
            f"Original itinerary is incomplete: {validation}"
        )

    print("\nINSIDE REVISION AGENT")
    print(f"User feedback: {feedback}")

    feedback_lower = feedback.lower()

    # --------------------------------
    # 2. Identify requested temporary agents
    # --------------------------------

    requested_agents = []

    keywords = {
        "flight_agent": [
            "flight", "flights", "airline",
            "airlines", "airport", "airfare"
        ],
        "hotel_agent": [
            "hotel", "hotels", "hostel",
            "accommodation", "stay", "resort"
        ],
        "weather_agent": [
            "weather", "forecast", "temperature",
            "climate", "rainfall"
        ],
        "budget_agent": [
            "budget", "cost", "expense",
            "expenses", "price", "affordable",
            "affordability", "how much"
        ]
    }

    for agent, agent_keywords in keywords.items():

        if any(
            word in feedback_lower
            for word in agent_keywords
        ):
            requested_agents.append(agent)

    # --------------------------------
    # 3. Handle itinerary-only requests
    # --------------------------------

    if not requested_agents:

        if "itinerary" in feedback_lower:
            message = (
                "The itinerary is already permanently included "
                "in your travel plan. All original itinerary "
                "days have been preserved."
            )
        else:
            raise ValueError(
                "I could not identify which travel information "
                "you want to update. Please specify flights, "
                "hotels, weather, budget, or itinerary."
            )

        return {
            "itinerary": draft,
            "revision_count": state.get("revision_count", 0) + 1,
            "approval_request": message + " Please review again.",
            "approved": False,
            "human_feedback": "",
            "messages": [AIMessage(content=message)]
        }

    # --------------------------------
    # 4. Create temporary agent state
    # --------------------------------

    temporary_state = dict(state)

    temporary_state["user_query"] = (
        state["user_query"]
        + "\n\nAdditional revision request: "
        + feedback
    )

    # These selections are temporary.
    # Do not overwrite the original selected_agents.
    temporary_state["selected_agents"] = (
        requested_agents + ["itinerary_agent"]
    )

    current_draft = draft

    current_llm_calls = state.get("llm_calls", 0)

    # --------------------------------
    # 5. Run only requested agents
    # --------------------------------

    agent_functions = {
        "flight_agent": flight_agent,
        "hotel_agent": hotel_agent,
        "weather_agent": weather_agent,
        "budget_agent": budget_agent
    }

    section_details = {
        "flight_agent": (2, "Flight Information", 3),
        "hotel_agent": (3, "Hotel Suggestions", 4),
        "weather_agent": (4, "Weather Information", 5),
        "budget_agent": (6, "Budget Analysis", 7)
    }

    updated_results = {}

    for agent_name in requested_agents:

        print(f"\nRunning temporary agent: {agent_name}")

        agent_function = agent_functions[agent_name]

        agent_output = agent_function(temporary_state)

        section_number, section_title, next_section = (
            section_details[agent_name]
        )

        result_key = {
            "flight_agent": "flight_results",
            "hotel_agent": "hotel_results",
            "weather_agent": "weather_results",
            "budget_agent": "budget_results"
        }[agent_name]

        new_content = str(
            agent_output.get(result_key, "")
        ).strip()

        if not new_content:
            raise ValueError(
                f"{agent_name} did not return any results."
            )

        # --------------------------------
        # 6. Update only the requested section
        # --------------------------------

        current_draft = update_itinerary_section(
            draft=current_draft,
            section_number=section_number,
            section_title=section_title,
            new_content=new_content,
            insert_before=next_section
        )

        # Save the updated specialist result.
        updated_results[result_key] = new_content

        # Preserve LLM call accounting.
        current_llm_calls = max(
            current_llm_calls,
            agent_output.get(
                "llm_calls",
                current_llm_calls
            )
        )

    # --------------------------------
    # 7. Validate original itinerary days
    # --------------------------------

    final_validation = validate_itinerary_days(
        current_draft,
        total_days
    )

    print("\nREVISION VALIDATION")
    print(final_validation)

    if not final_validation["valid"]:
        raise ValueError(
            "Revision changed the original itinerary days. "
            f"Validation: {final_validation}"
        )

    print(
        "\nSUCCESS: Requested information updated. "
        "Original itinerary days preserved."
    )

    # --------------------------------
    # 8. Return updated itinerary
    # --------------------------------

    revision_count = (
        state.get("revision_count", 0) + 1
    )

    updated_agent_names = ", ".join(
        requested_agents
    )

    return {
        "itinerary": current_draft,

        **updated_results,

        "revision_count": revision_count,

        "approval_request": (
            f"Revision {revision_count} is ready. "
            f"Updated information: {updated_agent_names}. "
            "The original itinerary has been preserved. "
            "Please review and approve the revised plan."
        ),

        "approved": False,
        "human_feedback": "",

        "messages": [
            AIMessage(
                content=(
                    f"Successfully updated: {updated_agent_names}. "
                    "Original itinerary preserved."
                )
            )
        ],

        "llm_calls": current_llm_calls
    }


# Final Agent: Only publishes an approved itinerary

def final_agent(state: TravelState):

    if not state.get("approved", False):
        raise ValueError(
            "Cannot publish an itinerary without human approval."
        )

    final_response = state.get("itinerary", "")

    return {
        "final_response": final_response,
        "messages": [
            AIMessage(content="Human-approved final itinerary generated.")
        ],
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


def route_after_approval(state: TravelState) -> str:

    if state.get("approved", False):
        return "final_agent"

    return "revision_agent"

# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("revision_agent", revision_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")

graph.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {
        "final_agent": "final_agent",
        "revision_agent": "revision_agent",
    }
)

graph.add_edge("revision_agent", "human_approval")

graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = get_database_url()
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
        "revision_count": result.get("revision_count", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "revision_count": 0,
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)