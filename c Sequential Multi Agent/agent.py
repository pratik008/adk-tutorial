import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.agents import SequentialAgent, ParallelAgent

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

# Database for city name corrections (shorthands, misspellings, etc.)
CITY_CORRECTIONS = {
    # Common shorthands
    "nyc": "new york",
    "ny": "new york",
    "la": "los angeles",
    "sf": "san francisco",
    "tokyo": "tokyo",
    
    # Common misspellings
    "sidney": "sydney",
    "sydny": "sydney",
    "londan": "london",
    "londun": "london",
    "tokio": "tokyo",
    "new yrok": "new york",
    "new yok": "new york",
    "paaris": "paris",
    "pari": "paris",
    "berln": "berlin",
    "barlin": "berlin",
}

def validate_city_name(city: str) -> dict:
    """Validates and corrects city names, handling shorthands and misspellings.
    
    Args:
        city (str): The user-provided city name that may contain errors.
        
    Returns:
        dict: Status and the corrected city name or error message.
    """
    if not city or not isinstance(city, str):
        return {
            "status": "error",
            "error_message": "Please provide a valid city name."
        }
    
    city_lower = city.lower().strip()
    
    # Check if the city is already valid
    if city_lower in WEATHER_DATABASE or city_lower in TIMEZONE_DATABASE:
        return {
            "status": "success",
            "corrected_city": city
        }
    
    # Check if we have a correction for this city
    if city_lower in CITY_CORRECTIONS:
        corrected = CITY_CORRECTIONS[city_lower]
        return {
            "status": "success",
            "corrected_city": corrected,
            "original_city": city
        }
    
    # No correction found
    return {
        "status": "error",
        "error_message": f"I couldn't recognize '{city}'. Please provide a valid city name."
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


model = "gemini-2.0-flash-exp"  # or another Gemini model version

# Create the validation agent
validation_agent = Agent(
    name="city_validation_agent",
    model=model,
    description="Agent that validates and corrects city names",
    instruction=(
        "You are an agent that validates city names, correcting spelling errors and "
        "expanding shorthand names to their full form."
    ),
    tools=[validate_city_name],
)

# Create the weather and time agent
weather_time_agent = Agent(
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

# Create the sequential agent that first validates the city name and then gets weather/time
root_agent = SequentialAgent(
    name="enhanced_weather_time_agent",
    description=(
        "Enhanced agent that validates city names before answering questions about "
        "weather and time for the corrected city."
    ),
    sub_agents=[validation_agent, weather_time_agent],
)