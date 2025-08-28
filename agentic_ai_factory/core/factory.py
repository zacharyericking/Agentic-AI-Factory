"""Main factory class that orchestrates the entire agentic AI system."""

import logging
import uuid
from typing import Dict, List, Optional, Any

from .models import Persona, SubPersona, Task, Agent, WorkflowResult, ToolType
from ..personas.persona_manager import PersonaManager
from ..agents.agent_builder import AgentBuilder
from ..dag.orchestrator import DAGOrchestrator
from ..tools.tool_registry import tool_registry


class AgenticFactory:
    """Main factory class for creating and managing agentic AI workflows."""
    
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-1.5-pro"):
        """Initialize the Agentic Factory.
        
        Args:
            gemini_api_key: API key for Google Gemini
            model_name: Gemini model to use
        """
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        
        # Initialize components
        self.persona_manager = PersonaManager(gemini_api_key, model_name)
        self.agent_builder = AgentBuilder(gemini_api_key, model_name)
        self.dag_orchestrator = DAGOrchestrator(self.agent_builder)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.logger.info("Agentic Factory initialized successfully")
    
    def register_persona(self, persona: Persona) -> None:
        """Register a persona that will be common among all instances of agents.
        
        Args:
            persona: The persona to register
        """
        self.persona_manager.register_persona(persona)
        self.logger.info(f"Registered persona: {persona.name}")
    
    async def solve_with_agents(
        self,
        question: str,
        persona_name: str,
        tool_configs: Dict[str, Dict[str, Any]] = None
    ) -> WorkflowResult:
        """Solve a question using persona-based agent decomposition.
        
        This is the main method that:
        1. Decomposes the persona into sub-personas using Gemini
        2. Breaks down the task using Gemini  
        3. Creates specialized agents with appropriate tools
        4. Orchestrates execution using a DAG
        
        Args:
            question: The question/task to solve
            persona_name: Name of the registered persona to use
            tool_configs: Configuration for tools (optional)
            
        Returns:
            WorkflowResult containing the final answer and execution details
        """
        workflow_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Starting workflow {workflow_id} for question: {question[:100]}...")
            
            # Step 1: Decompose persona using Gemini
            self.logger.info(f"Decomposing persona '{persona_name}' using Gemini")
            sub_personas = await self.persona_manager.decompose_persona(persona_name, question)
            
            # Step 2: Assign tasks to sub-personas using Gemini
            self.logger.info("Assigning tasks to sub-personas using Gemini")
            tasks = await self.persona_manager.assign_tasks_to_subpersonas(persona_name, question)
            
            # Step 3: Create agents from sub-personas and tasks
            self.logger.info("Creating specialized agents")
            agents = []
            for i, (sub_persona, task) in enumerate(zip(sub_personas, tasks)):
                agent_id = f"{workflow_id}_agent_{i}"
                agent = self.agent_builder.create_agent(
                    agent_id=agent_id,
                    persona=sub_persona,
                    task=task,
                    tool_configs=tool_configs or {}
                )
                agents.append(agent)
            
            # Step 4: Create and execute DAG workflow
            self.logger.info("Creating DAG workflow")
            workflow = self.dag_orchestrator.create_workflow(workflow_id, agents, question)
            
            self.logger.info("Executing DAG workflow")
            result = await self.dag_orchestrator.execute_workflow(workflow_id)
            
            # Update result with original question
            result.original_question = question
            
            self.logger.info(f"Workflow {workflow_id} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            
            # Return failed result
            return WorkflowResult(
                workflow_id=workflow_id,
                original_question=question,
                agents_used=[],
                execution_order=[],
                final_answer=f"Workflow failed: {str(e)}",
                intermediate_results={},
                execution_time=0.0,
                success=False
            )
    
    def configure_tool(
        self,
        tool_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Configure a tool for use in agents.
        
        Args:
            tool_type: Type of tool (rag, knowledge_graph, etc.)
            config: Tool configuration parameters
            
        Returns:
            Tool configuration dictionary ready for use
        """
        return {tool_type: config}
    
    def configure_rag_tool(
        self,
        collection_name: str = "default_collection",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Dict[str, Any]]:
        """Configure RAG tool with common parameters.
        
        Args:
            collection_name: Name of the document collection
            embedding_model: Sentence transformer model name
            persist_directory: Directory to persist ChromaDB
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            RAG tool configuration
        """
        return self.configure_tool("rag", {
            "collection_name": collection_name,
            "embedding_model": embedding_model,
            "persist_directory": persist_directory,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold
        })
    
    def configure_knowledge_graph_tool(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ) -> Dict[str, Dict[str, Any]]:
        """Configure Knowledge Graph tool with Neo4j parameters.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            
        Returns:
            Knowledge Graph tool configuration
        """
        return self.configure_tool("knowledge_graph", {
            "uri": uri,
            "username": username,
            "password": password,
            "database": database
        })
    
    def list_personas(self) -> List[str]:
        """List all registered personas."""
        return self.persona_manager.list_personas()
    
    def get_persona(self, name: str) -> Optional[Persona]:
        """Get a registered persona by name."""
        return self.persona_manager.get_persona(name)
    
    def get_workflow_result(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get the result of a completed workflow."""
        return self.dag_orchestrator.get_workflow_result(workflow_id)
    
    def list_workflows(self) -> List[str]:
        """List all workflow IDs."""
        return self.dag_orchestrator.list_workflows()
    
    def cleanup_workflow(self, workflow_id: str) -> bool:
        """Clean up a workflow and its associated resources."""
        success = self.dag_orchestrator.remove_workflow(workflow_id)
        
        # Clean up agents associated with this workflow
        agents_to_remove = []
        for agent_id in self.agent_builder.list_agents():
            if agent_id.startswith(workflow_id):
                agents_to_remove.append(agent_id)
        
        for agent_id in agents_to_remove:
            self.agent_builder.remove_agent(agent_id)
        
        return success
    
    def get_available_tool_types(self) -> List[str]:
        """Get list of available tool types."""
        return tool_registry.list_tool_types()
    
    def get_tool_schema(self, tool_type: str) -> Optional[Dict]:
        """Get schema for a tool type."""
        return tool_registry.get_tool_schema(tool_type)
    
    async def add_documents_to_rag(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]] = None,
        collection_name: str = "default_collection"
    ) -> bool:
        """Add documents to RAG system for later retrieval.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            collection_name: Collection to add documents to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a temporary RAG tool to add documents
            from ..tools.base_tool import ToolConfig
            from ..tools.rag_tool import RAGTool
            
            config = ToolConfig(
                name="temp_rag",
                description="Temporary RAG tool for document addition",
                parameters={"collection_name": collection_name}
            )
            
            rag_tool = RAGTool(config)
            await rag_tool.initialize()
            
            result = await rag_tool.execute(
                operation="add_documents",
                documents=documents,
                metadatas=metadatas
            )
            
            return result.success
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to RAG: {str(e)}")
            return False
    
    async def query_rag(
        self,
        query: str,
        collection_name: str = "default_collection",
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Query the RAG system directly.
        
        Args:
            query: Query string
            collection_name: Collection to query
            n_results: Number of results to return
            
        Returns:
            List of matching documents with similarity scores
        """
        try:
            from ..tools.base_tool import ToolConfig
            from ..tools.rag_tool import RAGTool
            
            config = ToolConfig(
                name="temp_rag",
                description="Temporary RAG tool for querying",
                parameters={"collection_name": collection_name}
            )
            
            rag_tool = RAGTool(config)
            await rag_tool.initialize()
            
            result = await rag_tool.execute(
                operation="query",
                query=query,
                n_results=n_results
            )
            
            if result.success:
                return result.data.get("results", [])
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to query RAG: {str(e)}")
            return []
