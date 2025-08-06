# Perplexia AI - Demo Search Agent

## Running the Application

This application is a demo search agent called perplexia. All the code for this application is in the `perplexia_ai/` directory. This application uses langchain and langgraph for the AI processing and gradio for the UI.

### How to Run

Run the application using the command:
```bash
uv run python run.py
```

The run script takes command line parameter `--mode` which can take options:
- `part1` - Query Understanding 
- `part2` - Basic Tools
- `part3` - Memory

Currently, we are only concerned with `part1`, so run the app as:
```bash
uv run python run.py --mode part1
```

## Architecture Overview

### Core Components

- **Application Entry Point**: `run.py` - Main script that parses command line arguments and launches the Gradio interface
- **Demo Factory**: `perplexia_ai/app.py` - Creates and configures the Gradio chat interface based on the selected mode
- **Chat Interface**: `perplexia_ai/core/chat_interface.py` - Abstract base class defining the core chat functionality
- **Mode Factory**: `perplexia_ai/week1/factory.py` - Factory pattern for creating different week 1 implementations

### Directory Structure

```
perplexia_ai/
├── __init__.py
├── app.py                    # Gradio demo creation and configuration
├── core/
│   ├── __init__.py
│   └── chat_interface.py     # Abstract ChatInterface base class
├── tools/
│   ├── __init__.py
│   └── calculator.py         # Basic arithmetic calculator tool
└── week1/                    # Week 1 assignment implementations
    ├── __init__.py
    ├── factory.py            # Mode selection factory
    ├── part1.py              # Query Understanding implementation
    ├── part2.py              # Basic Tools implementation  
    └── part3.py              # Memory implementation
```

### Week 1 Implementation Modes

1. **Part 1 - Query Understanding** (`QueryUnderstandingChat`)
   - Focus: Classify different types of questions and format responses accordingly
   - Features: Query classification, professional response formatting
   - Status: Template implementation (students need to implement)

2. **Part 2 - Basic Tools** (`BasicToolsChat`) 
   - Focus: Add calculator functionality to the chat interface
   - Features: Query understanding + arithmetic calculations
   - Status: Template implementation (students need to implement)

3. **Part 3 - Memory** (`MemoryChat`)
   - Focus: Add conversation memory and context management
   - Features: Query understanding + tools + conversation history
   - Status: Template implementation (students need to implement)

### Technology Stack

- **UI Framework**: Gradio - Provides the chat interface and web UI
- **AI Framework**: LangChain/LangGraph - For AI processing and workflow management
- **Environment Management**: UV - Python package and environment manager
- **Configuration**: Python dotenv - Environment variable management

### Key Features

- **Modular Architecture**: Each week part is implemented as a separate class following the ChatInterface protocol
- **Factory Pattern**: Clean mode selection and instantiation through factory methods  
- **Extensible Design**: Abstract base class allows for different implementations across assignments
- **Tool Integration**: Calculator tool demonstrates how external tools can be integrated
- **Chat History Support**: Interface supports maintaining conversation context (Part 3)

### Development Notes

- All implementations currently return placeholder responses ("hello")
- Students are expected to implement the actual functionality in the TODO sections
- The architecture is designed to support progression from basic query handling to complex tool usage and memory management
- Error handling and validation are built into the calculator tool as an example