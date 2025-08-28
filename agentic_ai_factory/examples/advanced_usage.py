"""Advanced usage example with multiple personas and complex workflows."""

import asyncio
import os
from agentic_ai_factory import AgenticFactory
from agentic_ai_factory.core.models import Persona, ToolType


async def main():
    """Demonstrate advanced usage with multiple personas."""
    
    # Initialize the factory
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return
    
    factory = AgenticFactory(gemini_api_key=gemini_api_key)
    
    # Create multiple specialized personas
    
    # 1. Data Scientist Persona
    data_scientist = Persona(
        name="Data Scientist",
        description="Expert in statistical analysis, machine learning, and data visualization",
        expertise_areas=["statistics", "machine learning", "data visualization", "predictive modeling"],
        personality_traits=["analytical", "methodical", "curious", "detail-oriented"],
        communication_style="technical but accessible",
        preferred_tools=[ToolType.RAG, ToolType.COMPUTATION, ToolType.DATABASE],
        constraints=["validate statistical significance", "explain methodology clearly"]
    )
    
    # 2. Business Strategist Persona  
    business_strategist = Persona(
        name="Business Strategist",
        description="Strategic business consultant focused on market analysis and business development",
        expertise_areas=["strategic planning", "market analysis", "business development", "competitive intelligence"],
        personality_traits=["strategic", "pragmatic", "results-oriented", "visionary"],
        communication_style="executive-level strategic",
        preferred_tools=[ToolType.RAG, ToolType.WEB_SEARCH, ToolType.KNOWLEDGE_GRAPH],
        constraints=["focus on actionable insights", "consider market dynamics"]
    )
    
    # 3. Technical Architect Persona
    tech_architect = Persona(
        name="Technical Architect",
        description="Senior technical architect specializing in system design and technology strategy",
        expertise_areas=["system architecture", "technology strategy", "scalability", "security"],
        personality_traits=["systematic", "thorough", "innovative", "pragmatic"],
        communication_style="technical and precise",
        preferred_tools=[ToolType.KNOWLEDGE_GRAPH, ToolType.DATABASE, ToolType.API],
        constraints=["consider scalability and security", "provide implementation guidance"]
    )
    
    # Register all personas
    factory.register_persona(data_scientist)
    factory.register_persona(business_strategist)
    factory.register_persona(tech_architect)
    
    # Configure tools
    rag_config = factory.configure_rag_tool(
        collection_name="enterprise_knowledge",
        persist_directory="./enterprise_db",
        top_k=10
    )
    
    kg_config = factory.configure_knowledge_graph_tool()
    
    tool_configs = {**rag_config, **kg_config}
    
    # Add comprehensive knowledge base
    enterprise_docs = [
        "Cloud computing adoption has increased by 300% in the last 5 years across enterprises.",
        "Microservices architecture enables better scalability and maintainability for large applications.",
        "Data privacy regulations like GDPR and CCPA are reshaping how companies handle customer data.",
        "Machine learning models require continuous monitoring and retraining to maintain accuracy.",
        "API-first design principles are becoming standard for modern software architecture.",
        "DevOps practices can reduce deployment time by 90% and improve system reliability.",
        "Customer experience is now the top competitive differentiator for 73% of companies.",
        "Artificial intelligence is expected to contribute $13 trillion to global GDP by 2030.",
        "Cybersecurity threats have increased by 600% during the pandemic.",
        "Sustainable technology practices are becoming mandatory for ESG compliance."
    ]
    
    print("Adding enterprise knowledge to RAG system...")
    await factory.add_documents_to_rag(enterprise_docs, collection_name="enterprise_knowledge")
    
    # Complex multi-faceted question
    complex_question = """
    Our company is considering a digital transformation initiative that involves:
    1. Migrating our legacy systems to cloud-native architecture
    2. Implementing AI/ML capabilities for customer personalization
    3. Ensuring compliance with data privacy regulations
    4. Developing a 3-year technology roadmap
    
    What would be a comprehensive strategy that addresses technical, business, and data science perspectives?
    """
    
    print("Complex Question:")
    print("=" * 60)
    print(complex_question)
    print("\\nProcessing with multiple specialized agents...")
    print("-" * 60)
    
    # Test with different personas
    personas_to_test = ["Data Scientist", "Business Strategist", "Technical Architect"]
    
    results = {}
    
    for persona_name in personas_to_test:
        print(f"\\nSolving with {persona_name} persona...")
        
        result = await factory.solve_with_agents(
            question=complex_question,
            persona_name=persona_name,
            tool_configs=tool_configs
        )
        
        results[persona_name] = result
        
        print(f"✓ {persona_name} workflow completed in {result.execution_time:.2f}s")
        print(f"  Agents used: {len(result.agents_used)}")
        print(f"  Success: {result.success}")
    
    # Display comprehensive results
    print("\\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("="*80)
    
    for persona_name, result in results.items():
        print(f"\\n{persona_name.upper()} PERSPECTIVE:")
        print("-" * 50)
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"Agents in Workflow: {' -> '.join(result.execution_order)}")
        print("\\nAnalysis:")
        print(result.final_answer)
        
        if result.intermediate_results:
            print("\\nDetailed Agent Results:")
            for i, (agent_id, agent_result) in enumerate(result.intermediate_results.items(), 1):
                print(f"\\n  Agent {i} ({agent_id}):")
                if isinstance(agent_result, dict) and 'result' in agent_result:
                    print(f"    {agent_result['result'][:200]}...")
                else:
                    print(f"    {str(agent_result)[:200]}...")
    
    # Cleanup all workflows
    print("\\n" + "-"*60)
    print("CLEANUP")
    print("-"*60)
    
    for persona_name, result in results.items():
        success = factory.cleanup_workflow(result.workflow_id)
        print(f"✓ Cleaned up {persona_name} workflow: {success}")
    
    # Show available personas and workflows
    print(f"\\nRegistered Personas: {factory.list_personas()}")
    print(f"Active Workflows: {factory.list_workflows()}")
    print(f"Available Tool Types: {factory.get_available_tool_types()}")


if __name__ == "__main__":
    asyncio.run(main())
