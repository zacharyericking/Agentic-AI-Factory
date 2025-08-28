# Agentic AI Factory

A Python package that simplifies the generation of specialized AI agents using LangGraph with persona-based decomposition and task assignment. The system uses Google Gemini to intelligently break down complex personas into specialized sub-agents and orchestrate their execution through directed acyclic graphs (DAGs).

## Features

🤖 **Persona-Based Agent Generation**: Register personas that define agent behavior and capabilities
🧠 **Intelligent Decomposition**: Uses Gemini to break personas into specialized sub-personas
📋 **Automated Task Assignment**: Gemini assigns specific tasks to each sub-agent based on their capabilities
🔧 **Integrated Tool System**: Built-in RAG (Retrieval-Augmented Generation) and Knowledge Graph tools
📊 **DAG Orchestration**: Uses LangGraph to create and execute directed acyclic graphs
🎯 **Comprehensive Results**: Synthesizes outputs from all agents into final answers

## Installation

```bash
pip install agentic-ai-factory
```

Or install from source:

```bash
git clone https://github.com/your-username/agentic-ai-factory.git
cd agentic-ai-factory
pip install -e .
```

## Quick Start

```python
import asyncio
import os
from agentic_ai_factory import AgenticFactory
from agentic_ai_factory.core.models import Persona, ToolType

async def main():
    # Initialize with Gemini API key
    factory = AgenticFactory(gemini_api_key=os.getenv("GEMINI_API_KEY"))
    
    # Create and register a persona
    research_persona = Persona(
        name="Research Analyst",
        description="Expert in gathering and analyzing information",
        expertise_areas=["research", "analysis", "synthesis"],
        personality_traits=["analytical", "thorough", "objective"],
        preferred_tools=[ToolType.RAG, ToolType.WEB_SEARCH]
    )
    
    factory.register_persona(research_persona)
    
    # Configure tools (optional)
    tool_configs = factory.configure_rag_tool()
    
    # Solve a complex question
    result = await factory.solve_with_agents(
        question="What are the latest trends in AI and their business impact?",
        persona_name="Research Analyst",
        tool_configs=tool_configs
    )
    
    print(f"Final Answer: {result.final_answer}")
    print(f"Agents Used: {result.agents_used}")
    print(f"Execution Time: {result.execution_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Concepts

### Personas
Personas define the characteristics, expertise, and behavior of your agents:

```python
persona = Persona(
    name="Data Scientist",
    description="Expert in statistical analysis and machine learning",
    expertise_areas=["statistics", "ML", "data visualization"],
    personality_traits=["analytical", "methodical", "curious"],
    communication_style="technical but accessible",
    preferred_tools=[ToolType.RAG, ToolType.COMPUTATION],
    constraints=["validate statistical significance"]
)
```

### Automatic Decomposition
The system uses Gemini to automatically:
1. **Decompose personas** into specialized sub-personas
2. **Break down tasks** into manageable subtasks
3. **Assign tasks** to the most suitable sub-agents
4. **Create dependencies** between tasks for optimal execution order

### Tool Integration
Built-in tools include:

- **RAG (Retrieval-Augmented Generation)**: Document storage and retrieval using ChromaDB
- **Knowledge Graph**: Graph-based knowledge storage using Neo4j
- **Web Search**: External information retrieval
- **Database**: Structured data access
- **API**: External service integration
- **Computation**: Mathematical and statistical operations

### DAG Orchestration
Uses LangGraph to create directed acyclic graphs that:
- Execute agents in the correct dependency order
- Pass context between agents
- Synthesize results into comprehensive answers
- Handle failures gracefully

## Advanced Usage

### Multiple Personas

```python
# Register multiple specialized personas
factory.register_persona(data_scientist_persona)
factory.register_persona(business_strategist_persona)
factory.register_persona(technical_architect_persona)

# Each persona will create different specialized agents
# for the same complex question
```

### Tool Configuration

```python
# Configure RAG tool
rag_config = factory.configure_rag_tool(
    collection_name="my_documents",
    embedding_model="all-MiniLM-L6-v2",
    persist_directory="./my_db"
)

# Configure Knowledge Graph
kg_config = factory.configure_knowledge_graph_tool(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="your_password"
)

# Combine configurations
tool_configs = {**rag_config, **kg_config}
```

### Document Management

```python
# Add documents to RAG system
documents = [
    "AI is transforming healthcare...",
    "Machine learning requires quality data...",
    # ... more documents
]

await factory.add_documents_to_rag(documents)

# Query RAG directly
results = await factory.query_rag("What is machine learning?")
```

## Examples

The package includes comprehensive examples:

- `basic_usage.py`: Simple persona registration and question solving
- `advanced_usage.py`: Multiple personas, complex workflows, and tool integration

Run examples:

```bash
cd agentic_ai_factory/examples
python basic_usage.py
python advanced_usage.py
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Persona       │    │   Gemini AI      │    │   Specialized   │
│   Registration  │───▶│   Decomposition  │───▶│   Sub-Agents    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Task           │    │   Tool          │
                       │   Assignment     │    │   Integration   │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────────────────────────────┐
                       │         LangGraph DAG                   │
                       │         Orchestration                   │
                       └─────────────────────────────────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │  Synthesized    │
                               │  Final Answer   │
                               └─────────────────┘
```

## Requirements

- Python 3.8+
- Google Gemini API key
- Optional: Neo4j for Knowledge Graph functionality
- Optional: ChromaDB for RAG functionality (installed automatically)

## Configuration

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

For Knowledge Graph functionality, ensure Neo4j is running:

```bash
# Using Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Additional tool integrations (SQL databases, external APIs)
- [ ] Web interface for workflow visualization
- [ ] Enhanced persona templates and examples
- [ ] Performance optimizations for large-scale deployments
- [ ] Integration with additional LLM providers
