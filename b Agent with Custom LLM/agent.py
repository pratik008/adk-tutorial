import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import os

# Database of cities and their weather information
WEATHER_DATABASE = {
    "new york": {
        "condition": "sunny",
        "temperature_celsius": 25,
        "temperature_fahrenheit": 77,
    },
    "london": {
        "condition": "rainy", 
        "temperature_celsius": 18,
        "temperature_fahrenheit": 64,
    },
    "tokyo": {
        "condition": "cloudy",
        "temperature_celsius": 22,
        "temperature_fahrenheit": 72,
    },
    "sydney": {
        "condition": "partly cloudy",
        "temperature_celsius": 27,
        "temperature_fahrenheit": 81,
    },
}

# Database of cities and their timezone identifiers
TIMEZONE_DATABASE = {
    "new york": "America/New_York",
    "london": "Europe/London",
    "tokyo": "Asia/Tokyo",
    "sydney": "Australia/Sydney",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "mumbai": "Asia/Kolkata",
    "los angeles": "America/Los_Angeles",
}

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    city_key = city.lower()
    if city_key in WEATHER_DATABASE:
        weather = WEATHER_DATABASE[city_key]
        return {
            "status": "success",
            "report": (
                f"The weather in {city} is {weather['condition']} with a temperature of "
                f"{weather['temperature_celsius']} degrees Celsius "
                f"({weather['temperature_fahrenheit']} degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """
    city_key = city.lower()
    if city_key in TIMEZONE_DATABASE:
        tz_identifier = TIMEZONE_DATABASE[city_key]
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


# Determine which model to use based on environment variable
use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

if use_azure:
    # Configure Azure OpenAI model with LiteLLM
    model = LiteLlm(
        model=f"azure/{os.getenv('DEPLOYMENT_NAME')}", # gpt-4o
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("ENDPOINT_URL"),
        api_version=os.getenv("API_VERSION")
    )
    print("Using Azure OpenAI model")
else:
    # Use Google's Gemini model
    model = "gemini-2.0-flash-exp"  # or another Gemini model version
    print("Using Google Gemini model")



root_agent = Agent(
    name="weather_time_agent",
    model=model,
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city."
    ),
    tools=[get_weather, get_current_time],
)