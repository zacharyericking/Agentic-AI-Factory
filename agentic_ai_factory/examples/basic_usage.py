"""Basic usage example of the Agentic AI Factory."""

import asyncio
import os
from agentic_ai_factory import AgenticFactory
from agentic_ai_factory.core.models import Persona, ToolType


async def main():
    """Demonstrate basic usage of the Agentic AI Factory."""
    
    # Initialize the factory with Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    factory = AgenticFactory(gemini_api_key=gemini_api_key)
    
    # Create and register a research analyst persona
    research_persona = Persona(
        name="Research Analyst",
        description="An expert research analyst skilled in gathering, analyzing, and synthesizing information from multiple sources",
        expertise_areas=["market research", "data analysis", "report writing", "competitive analysis"],
        personality_traits=["analytical", "detail-oriented", "objective", "thorough"],
        communication_style="professional and data-driven",
        preferred_tools=[ToolType.RAG, ToolType.WEB_SEARCH, ToolType.DATABASE],
        constraints=["must cite sources", "provide evidence-based conclusions"]
    )
    
    factory.register_persona(research_persona)
    
    # Configure RAG tool (optional)
    rag_config = factory.configure_rag_tool(
        collection_name="research_docs",
        persist_directory="./research_db"
    )
    
    # Configure Knowledge Graph tool (optional)
    kg_config = factory.configure_knowledge_graph_tool(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    
    # Combine tool configurations
    tool_configs = {**rag_config, **kg_config}
    
    # Add some sample documents to RAG (optional)
    sample_docs = [
        "Artificial Intelligence is transforming various industries including healthcare, finance, and transportation.",
        "Machine Learning algorithms can be supervised, unsupervised, or reinforcement-based.",
        "Natural Language Processing enables computers to understand and generate human language.",
        "Deep Learning uses neural networks with multiple layers to learn complex patterns.",
        "Computer Vision allows machines to interpret and understand visual information."
    ]
    
    await factory.add_documents_to_rag(sample_docs)
    
    # Solve a complex question using the persona
    question = "What are the current trends in artificial intelligence and how are they impacting different industries?"
    
    print(f"Question: {question}")
    print("\\nProcessing with Agentic AI Factory...")
    print("-" * 50)
    
    # Execute the workflow
    result = await factory.solve_with_agents(
        question=question,
        persona_name="Research Analyst",
        tool_configs=tool_configs
    )
    
    # Display results
    print(f"\\nWorkflow ID: {result.workflow_id}")
    print(f"Success: {result.success}")
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print(f"Agents Used: {len(result.agents_used)}")
    print(f"Execution Order: {' -> '.join(result.execution_order)}")
    
    print("\\nFinal Answer:")
    print("=" * 50)
    print(result.final_answer)
    
    if result.intermediate_results:
        print("\\nIntermediate Results:")
        print("-" * 30)
        for agent_id, agent_result in result.intermediate_results.items():
            print(f"\\n{agent_id}:")
            if isinstance(agent_result, dict) and 'result' in agent_result:
                print(agent_result['result'])
            else:
                print(agent_result)
    
    # Clean up
    factory.cleanup_workflow(result.workflow_id)
    print("\\nWorkflow cleaned up successfully!")


if __name__ == "__main__":
    asyncio.run(main())
