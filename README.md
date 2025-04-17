# Google ADK Tutorial Project

This repository contains a series of progressively complex examples demonstrating how to build AI agents using the Google Agent Development Kit (ADK). Each folder contains a different implementation showcasing various agent patterns and capabilities.

## Overview

The Google Agent Development Kit (ADK) is a framework for building, testing, and deploying AI agents powered by Google's Gemini models. This tutorial walks through different agent architectures and features to help you understand how to build robust AI agents for various use cases.

## Project Structure

The examples are arranged in order of increasing complexity:

- **a Agent With Tool**: Introduction to basic agent functionality with tool use
- **b Agent with Custom LLM**: Configuring agents to use custom language models
- **c Sequential Multi Agent**: Building a pipeline of agents that execute sequentially
- **d Parallel Multi Agent**: Creating agents that perform tasks concurrently
- **e Stateful Agent**: Building agents that maintain state between interactions
- **f Parallel Stateful Agent**: Combining parallel execution with state management
- **g Safe Agents**: Implementing safety features and guardrails in agents

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google API key with access to Gemini models

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd adk-tutorial
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv adk-env
   source adk-env/bin/activate  # On Windows: adk-env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Google API key:
   ```
   export GOOGLE_API_KEY=your_api_key_here
   ```

## Example Descriptions

### a Agent With Tool
A basic example showing how to create an agent that can use external tools to complete tasks. This demonstrates the fundamental pattern of agent-tool interaction.

### b Agent with Custom LLM
Shows how to configure an agent to use a specific language model, with custom parameters and settings.

### c Sequential Multi Agent
Demonstrates how to build a pipeline of agents that pass information between them in sequence, where each agent handles a specific subtask.

### d Parallel Multi Agent
Explores how to create agents that execute multiple subtasks concurrently, improving efficiency for independent tasks.

### e Stateful Agent
Introduces state management in agents, allowing them to remember information between turns in a conversation.

### f Parallel Stateful Agent
Combines parallel execution with state management, demonstrating how to share state between concurrent agents.

### g Safe Agents
Implements safety measures, input validation, and content filtering to build responsible AI agents that avoid harmful or inappropriate responses.

## Usage Examples

To run any of the examples, navigate to the project root and use:

```bash
python "folder name/agent.py"
```

For example:

```bash
python "a Agent With Tool/agent.py"
```

## Dependencies

The project requires the following main packages:
- `google-adk`: Google's Agent Development Kit
- `google-generativeai`: Google's Generative AI Python SDK
- `python-dotenv`: For environment variable management

See `requirements.txt` for specific version requirements.

## Resources

- [Google ADK Documentation](https://ai.google.dev/docs/agents_overview)
- [Google Gemini API Documentation](https://ai.google.dev/docs/gemini_api_overview)

## License

This tutorial is for educational purposes.


## Author

Created by Pratik Mehta  
GitHub: [pratik008](https://github.com/pratik008)


---

*Note: This tutorial is designed to provide a step-by-step introduction to Google's Agent Development Kit. The examples progress from simple to complex patterns to help you understand how to build sophisticated AI agents.* 