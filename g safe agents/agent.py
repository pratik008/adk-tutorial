import datetime
import re
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional, List

from google.adk import Agent
from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.tools import BaseTool, ToolContext

# Weather database
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

# Timezone database
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

# City corrections database
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

# Define blocked terms for safety checks
BLOCKED_TERMS = [
    "bomb", "terrorist", "hack", "steal", "murder", "kill", 
    "illegal", "fraud", "abuse", "harmful", "violent",
    "weapon", "explicit", "offensive", "gambling", "suicide"
]

# State management utilities
def before_agent(callback_context: InvocationContext):
    """Initialize the state with default user preferences if not already set."""
    # Set default temperature unit if not already in state
    if "temperature_unit" not in callback_context.state:
        callback_context.state["temperature_unit"] = "celsius"
    
    # Initialize history of cities if not already in state
    if "city_history" not in callback_context.state:
        callback_context.state["city_history"] = []
    
    # Initialize safety metrics if not already in state
    if "safety_metrics" not in callback_context.state:
        callback_context.state["safety_metrics"] = {
            "blocked_attempts": 0,
            "last_blocked_time": None,
            "blocked_terms_detected": []
        }
    
    # Log current state (for debugging)
    print(f"Current state: {callback_context.state}")

def rate_limit_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Simple rate limiting implementation."""
    # Ensure no empty text parts
    for content in llm_request.contents:
        for part in content.parts:
            if part.text == "":
                part.text = " "

def safety_check(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Safety check callback that detects and blocks harmful content.
    
    This function:
    1. Scans user inputs for blocked terms
    2. Maintains metrics on blocked attempts
    3. Either allows the request to proceed or replaces it with a safety message
    
    Args:
        callback_context: The callback context containing state
        llm_request: The LLM request to check
    """
    # Get current time for tracking
    current_time = datetime.datetime.now().isoformat()
    
    # Get input text from request
    input_texts = []
    for content in llm_request.contents:
        for part in content.parts:
            if hasattr(part, 'text') and part.text:
                input_texts.append(part.text.lower())
    
    # Join all input texts
    combined_input = " ".join(input_texts)
    
    # Check for blocked terms
    detected_terms = []
    for term in BLOCKED_TERMS:
        if re.search(r'\b' + re.escape(term) + r'\b', combined_input):
            detected_terms.append(term)
    
    # If blocked terms detected, modify the request
    if detected_terms:
        # Update safety metrics in state
        if "safety_metrics" not in callback_context.state:
            callback_context.state["safety_metrics"] = {
                "blocked_attempts": 0,
                "last_blocked_time": None,
                "blocked_terms_detected": []
            }
        
        metrics = callback_context.state["safety_metrics"]
        metrics["blocked_attempts"] += 1
        metrics["last_blocked_time"] = current_time
        metrics["blocked_terms_detected"].extend(detected_terms)
        
        # Clear the original contents
        llm_request.contents.clear()
        
        # Replace with safety message prompt
        from google.genai.types import Part, Content
        safety_prompt = (
            "The user's query contained potentially harmful content that I cannot respond to. "
            "Please provide a polite response explaining that you cannot assist with "
            "harmful, illegal, unethical, or dangerous requests. Offer to help with "
            "weather and time-related queries instead."
        )
        
        # Create new content with safety prompt
        new_content = Content(
            parts=[Part(text=safety_prompt)],
            role="user"
        )
        llm_request.contents.append(new_content)
        
        print(f"Safety check blocked request. Detected terms: {detected_terms}")
    else:
        # Pass the rate limit check as well
        rate_limit_callback(callback_context, llm_request)

def update_city_history(city: str, tool_context: ToolContext) -> None:
    """Helper function to update the city history in state.
    
    Args:
        city (str): The city name to add to history.
        tool_context (ToolContext): The tool context containing state.
    """
    # Ensure city_history exists
    if "city_history" not in tool_context.state:
        tool_context.state["city_history"] = []
    
    # Normalize city name
    city_lower = city.lower().strip()
    
    # Add to history if not already the most recent
    if not tool_context.state["city_history"] or tool_context.state["city_history"][-1] != city_lower:
        tool_context.state["city_history"].append(city_lower)
        # Keep only the most recent 5 cities
        if len(tool_context.state["city_history"]) > 5:
            tool_context.state["city_history"] = tool_context.state["city_history"][-5:]

# Core Tool Implementations
def validate_city_name(city: str, tool_context: ToolContext) -> dict:
    """Validates and corrects city names, handling shorthands and misspellings.
    
    Args:
        city (str): The user-provided city name that may contain errors.
        tool_context (ToolContext): Context containing the state.
        
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
        # Add valid city to history
        update_city_history(city_lower, tool_context)
                
        return {
            "status": "success",
            "corrected_city": city_lower
        }
    
    # Check if we have a correction for this city
    if city_lower in CITY_CORRECTIONS:
        corrected = CITY_CORRECTIONS[city_lower]
        
        # Add corrected city to history
        update_city_history(corrected, tool_context)
        
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

def get_weather(city: str, tool_context: ToolContext) -> dict:
    """Retrieves the current weather report for a specified city using user's preferred unit.

    Args:
        city (str): The name of the city for which to retrieve the weather report.
        tool_context (ToolContext): Context containing the state with user preferences.

    Returns:
        dict: status and result or error msg.
    """
    city_key = city.lower()
    
    # Get the user's preferred temperature unit from state
    temperature_unit = tool_context.state.get("temperature_unit", "celsius")
    
    # Always update the city history in state
    update_city_history(city_key, tool_context)
    
    if city_key in WEATHER_DATABASE:
        weather = WEATHER_DATABASE[city_key]
        
        # Format response based on user preference
        if temperature_unit.lower() == "fahrenheit":
            temperature_str = f"{weather['temperature_fahrenheit']} degrees Fahrenheit"
        else:
            temperature_str = f"{weather['temperature_celsius']} degrees Celsius"
            
        return {
            "status": "success",
            "report": f"The weather in {city} is {weather['condition']} with a temperature of {temperature_str}.",
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }

def get_current_time(city: str, tool_context: ToolContext) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.
        tool_context (ToolContext): Context containing the state.

    Returns:
        dict: status and result or error msg.
    """
    city_key = city.lower()
    
    # Always update the city history
    update_city_history(city_key, tool_context)
    
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

def update_temperature_preference(unit: str, tool_context: ToolContext) -> dict:
    """Updates the user's temperature unit preference.
    
    Args:
        unit (str): The temperature unit preference (celsius or fahrenheit).
        tool_context (ToolContext): Context containing the state.
        
    Returns:
        dict: Status and confirmation message.
    """
    unit = unit.lower().strip()
    
    # Validate the unit
    if unit not in ["celsius", "fahrenheit"]:
        return {
            "status": "error",
            "error_message": "Invalid temperature unit. Please choose 'celsius' or 'fahrenheit'."
        }
    
    # Update the preference in state
    tool_context.state["temperature_unit"] = unit
    
    return {
        "status": "success",
        "message": f"Your temperature unit preference has been updated to {unit}."
    }

def get_recent_cities(tool_context: ToolContext) -> dict:
    """Retrieves the user's recently searched cities.
    
    Args:
        tool_context (ToolContext): Context containing the state.
        
    Returns:
        dict: Status and list of recently searched cities.
    """
    city_history = tool_context.state.get("city_history", [])
    
    if not city_history:
        return {
            "status": "success",
            "message": "You haven't searched for any cities yet."
        }
    
    # Format the city list
    city_list = ", ".join(city_history)
    
    return {
        "status": "success",
        "message": f"Your recently searched cities: {city_list}",
        "cities": city_history
    }

def get_safety_metrics(tool_context: ToolContext) -> dict:
    """Retrieves safety metrics from the agent's state.
    
    Args:
        tool_context (ToolContext): Context containing the state.
        
    Returns:
        dict: Safety metrics information.
    """
    # Get safety metrics from state
    metrics = tool_context.state.get("safety_metrics", {
        "blocked_attempts": 0,
        "last_blocked_time": None,
        "blocked_terms_detected": []
    })
    
    # Format the response
    if metrics["blocked_attempts"] == 0:
        message = "No safety violations have been detected in this session."
    else:
        message = (
            f"Safety system has blocked {metrics['blocked_attempts']} "
            f"request(s). Last blocked: {metrics['last_blocked_time']}. "
            f"Terms detected: {', '.join(metrics['blocked_terms_detected'])}."
        )
    
    return {
        "status": "success",
        "message": message,
        "metrics": metrics
    }

def combine_weather_time_info(city: Optional[str] = None, tool_context: Optional[ToolContext] = None) -> dict:
    """Combines weather and time information into a single response.
    
    Args:
        city (Optional[str], optional): The city name. Defaults to None.
        tool_context (Optional[ToolContext], optional): Context containing the state.
        
    Returns:
        dict: Combined weather and time information or error message.
    """
    if not city:
        return {
            "status": "error",
            "error_message": "No city information was provided."
        }
    
    # Format the output in a user-friendly way
    unit_preference = "default"
    if tool_context and "temperature_unit" in tool_context.state:
        unit_preference = tool_context.state["temperature_unit"]
    
    return {
        "status": "success",
        "message": f"Here's the information for {city} (temperature displayed in {unit_preference}):",
        "city": city
    }

# Define the model to use
model = "gemini-2.0-flash-exp"  # or another Gemini model version

# Create the validation agent with state awareness and safety checks
validation_agent = Agent(
    name="city_validation_agent",
    model=model,
    description="Agent that validates and corrects city names",
    instruction=(
        "You are an agent that validates city names, correcting spelling errors and "
        "expanding shorthand names to their full form. You also update the search history."
    ),
    tools=[validate_city_name],
    before_agent_callback=before_agent,
    before_model_callback=safety_check,
)

# Create separate weather agent with state awareness and safety checks
weather_agent = Agent(
    name="weather_agent",
    model=model,
    description="Agent to answer questions about the weather in a city",
    instruction=(
        "You are a helpful agent who can provide weather information for a city. "
        "You adapt your responses to show temperatures in the user's preferred unit. "
        "You also track the cities that users search for in their history."
    ),
    tools=[get_weather],
    before_agent_callback=before_agent,
    before_model_callback=safety_check,
)

# Create separate time agent with state awareness and safety checks
time_agent = Agent(
    name="time_agent",
    model=model,
    description="Agent to answer questions about the current time in a city",
    instruction=(
        "You are a helpful agent who can provide current time information for a city. "
        "You also track the cities that users search for in their history."
    ),
    tools=[get_current_time],
    before_agent_callback=before_agent,
    before_model_callback=safety_check,
)

# Create a combination agent with state awareness and safety checks
combination_agent = Agent(
    name="combination_agent",
    model=model,
    description="Agent that combines weather and time information",
    instruction=(
        "You are a helpful agent who combines weather and time information for a city "
        "into a comprehensive response, respecting the user's temperature unit preference."
    ),
    tools=[combine_weather_time_info],
    before_agent_callback=before_agent,
    before_model_callback=safety_check,
)

# Create a preferences agent to handle user preference updates with safety checks
preferences_agent = Agent(
    name="preferences_agent",
    model=model,
    description="Agent that manages user preferences",
    instruction=(
        "You are a helpful agent who manages user preferences, such as temperature units. "
        "You can update preferences and provide information about current settings."
    ),
    tools=[update_temperature_preference, get_recent_cities, get_safety_metrics],
    before_agent_callback=before_agent,
    before_model_callback=safety_check,
)

# Create the parallel agent that gets weather and time information simultaneously
# This agent will share state because each sub-agent inherits state from the parent
parallel_weather_time_agent = ParallelAgent(
    name="parallel_weather_time_agent",
    description="Gets weather and time information in parallel while maintaining state",
    sub_agents=[weather_agent, time_agent],
)

# Create the sequential agent pipeline:
# 1. First validates the city name
# 2. Then runs weather and time agents in parallel
# 3. Finally combines the results
# All while maintaining shared state and with safety checks
safe_root_agent = SequentialAgent(
    name="safe_stateful_parallel_weather_time_agent",
    description=(
        "Enhanced stateful agent with safety checks that validates city names before answering questions about "
        "weather and time in parallel for the corrected city, then combines the results. "
        "The agent maintains state about user preferences and search history."
    ),
    sub_agents=[validation_agent, parallel_weather_time_agent, combination_agent],
    before_agent_callback=before_agent,
)

# Create a standalone agent that can handle all operations including preferences
# with safety checks throughout
root_agent = Agent(
    name="safe_weather_preferences_agent",
    model=model,
    description="A standalone agent that can handle all weather operations including preferences, with safety checks",
    instruction=(
        "You are a helpful weather assistant that remembers user preferences like temperature units. "
        "You can provide weather and time information for cities, and you adapt your responses based on user preferences. "
        "You can also update user preferences when requested and show search history."
        "\n\n"
        "You have built-in safety features that prevent responding to harmful, dangerous, "
        "illegal, or unethical requests. If a user asks for something inappropriate, "
        "you will politely explain that you cannot assist with such requests."
        "\n\n"
        "Current features you support: "
        "- Weather information for supported cities "
        "- Time information for supported cities "
        "- Setting temperature unit preference (celsius/fahrenheit) "
        "- Remembering recently searched cities "
        "- Safety metrics tracking"
    ),
    tools=[
        validate_city_name,
        get_weather,
        get_current_time,
        update_temperature_preference,
        get_recent_cities,
        get_safety_metrics,
        combine_weather_time_info
    ],
    before_agent_callback=before_agent,
    before_model_callback=safety_check,
    # Set output_key to save agent responses in state
    output_key="last_response"
)