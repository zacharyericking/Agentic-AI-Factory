"""Agent builder for creating specialized agents from personas and tasks."""

import logging
from typing import Dict, List, Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from ..core.models import Agent, SubPersona, Task, ToolType
from ..tools.tool_registry import tool_registry, ToolConfig
from ..tools.base_tool import BaseTool


class AgentBuilder:
    """Builds specialized agents from personas and tasks."""
    
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-1.5-pro"):
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self.agents: Dict[str, Agent] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0.7
        )
    
    def create_agent(
        self,
        agent_id: str,
        persona: SubPersona,
        task: Task,
        tool_configs: Dict[str, Dict[str, Any]] = None
    ) -> Agent:
        """Create an agent from a sub-persona and task."""
        
        # Create tools based on required tool types
        agent_tools = []
        if tool_configs:
            for tool_type in task.required_tools:
                tool_type_str = tool_type.value if hasattr(tool_type, 'value') else str(tool_type)
                
                if tool_type_str in tool_configs:
                    tool_id = f"{agent_id}_{tool_type_str}"
                    config = ToolConfig(
                        name=f"{persona.name}_{tool_type_str}",
                        description=f"{tool_type_str} tool for {persona.name}",
                        parameters=tool_configs[tool_type_str]
                    )
                    
                    try:
                        tool = tool_registry.create_tool(tool_type_str, tool_id, config)
                        agent_tools.append(tool_id)
                        self.logger.info(f"Created {tool_type_str} tool for agent {agent_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create {tool_type_str} tool for agent {agent_id}: {str(e)}")
        
        # Create agent
        agent = Agent(
            id=agent_id,
            name=f"{persona.name}_Agent",
            persona=persona,
            assigned_task=task,
            tools=agent_tools
        )
        
        self.agents[agent_id] = agent
        self.logger.info(f"Created agent: {agent_id}")
        
        return agent
    
    async def execute_agent_task(self, agent_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute an agent's assigned task."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent '{agent_id}' not found")
        
        agent = self.agents[agent_id]
        context = context or {}
        
        try:
            # Update agent status
            agent.status = "running"
            
            # Initialize tools
            await self._initialize_agent_tools(agent)
            
            # Create system prompt based on persona and task
            system_prompt = self._create_system_prompt(agent)
            
            # Create task execution prompt
            task_prompt = self._create_task_prompt(agent, context)
            
            # Execute task using LLM with tool access
            result = await self._execute_with_tools(agent, system_prompt, task_prompt, context)
            
            # Update agent status
            agent.status = "completed"
            
            self.logger.info(f"Agent {agent_id} completed task successfully")
            
            return {
                "agent_id": agent_id,
                "task_id": agent.assigned_task.id,
                "success": True,
                "result": result,
                "context": context
            }
            
        except Exception as e:
            agent.status = "failed"
            self.logger.error(f"Agent {agent_id} failed to execute task: {str(e)}")
            
            return {
                "agent_id": agent_id,
                "task_id": agent.assigned_task.id,
                "success": False,
                "error": str(e),
                "context": context
            }
    
    async def _initialize_agent_tools(self, agent: Agent) -> None:
        """Initialize all tools for an agent."""
        for tool_id in agent.tools:
            tool = tool_registry.get_tool(tool_id)
            if tool:
                await tool.initialize()
    
    def _create_system_prompt(self, agent: Agent) -> str:
        """Create system prompt based on agent's persona."""
        persona = agent.persona
        task = agent.assigned_task
        
        return f"""
You are {persona.name}, an AI agent with the following characteristics:

PERSONA:
- Description: {persona.description}
- Specific Role: {persona.specific_role}
- Capabilities: {', '.join(persona.capabilities)}
- Parent Persona: {persona.parent_persona}

ASSIGNED TASK:
- Task ID: {task.id}
- Description: {task.description}
- Type: {task.task_type}
- Priority: {task.priority}
- Expected Output: {task.expected_output}

AVAILABLE TOOLS:
{self._format_available_tools(agent)}

INSTRUCTIONS:
1. You must complete your assigned task according to your persona characteristics
2. Use the available tools when necessary to gather information or perform operations
3. Provide clear, actionable results that match the expected output format
4. Consider task dependencies and context from other agents
5. Maintain consistency with your persona's expertise and communication style

Remember: You are working as part of a larger system. Your output will be used by other agents or to answer the original question.
"""
    
    def _create_task_prompt(self, agent: Agent, context: Dict[str, Any]) -> str:
        """Create task-specific prompt with context."""
        task = agent.assigned_task
        
        context_info = ""
        if context:
            context_info = f"\nCONTEXT FROM PREVIOUS AGENTS:\n{self._format_context(context)}\n"
        
        return f"""
{context_info}
TASK TO COMPLETE:
{task.description}

EXPECTED OUTPUT FORMAT:
{task.expected_output}

Please complete this task now. Use your available tools as needed and provide a comprehensive response.
"""
    
    def _format_available_tools(self, agent: Agent) -> str:
        """Format available tools for the system prompt."""
        if not agent.tools:
            return "No specific tools available."
        
        tool_descriptions = []
        for tool_id in agent.tools:
            tool = tool_registry.get_tool(tool_id)
            if tool:
                schema = tool.get_schema()
                tool_descriptions.append(f"- {tool.name}: {tool.description}")
        
        return "\n".join(tool_descriptions) if tool_descriptions else "No tools initialized."
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for the prompt."""
        formatted_context = []
        for key, value in context.items():
            if isinstance(value, dict) and 'result' in value:
                formatted_context.append(f"{key}: {value['result']}")
            else:
                formatted_context.append(f"{key}: {value}")
        
        return "\n".join(formatted_context)
    
    async def _execute_with_tools(
        self,
        agent: Agent,
        system_prompt: str,
        task_prompt: str,
        context: Dict[str, Any]
    ) -> str:
        """Execute agent task with tool integration."""
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt)
        ]
        
        # For now, execute without tool integration (can be enhanced with LangGraph)
        # This is a simplified version - in production, you'd want full tool integration
        response = await self.llm.ainvoke(messages)
        
        # Here you could add tool calling logic based on the response
        # For example, if the response indicates a need to use RAG or Knowledge Graph
        result = await self._process_tool_requests(agent, response.content, context)
        
        return result
    
    async def _process_tool_requests(
        self,
        agent: Agent,
        response: str,
        context: Dict[str, Any]
    ) -> str:
        """Process any tool requests in the agent's response."""
        
        # Simple tool integration - in a full implementation, you'd parse the response
        # for tool calls and execute them appropriately
        
        # For demonstration, let's check if the agent needs to use RAG
        if "search" in response.lower() or "retrieve" in response.lower():
            await self._use_rag_if_available(agent, response, context)
        
        # Check if knowledge graph operations are needed
        if "knowledge" in response.lower() or "relationship" in response.lower():
            await self._use_knowledge_graph_if_available(agent, response, context)
        
        return response
    
    async def _use_rag_if_available(self, agent: Agent, response: str, context: Dict[str, Any]) -> None:
        """Use RAG tool if available for the agent."""
        for tool_id in agent.tools:
            tool = tool_registry.get_tool(tool_id)
            if tool and isinstance(tool, type(tool_registry.tool_classes.get("rag", type(None)))):
                try:
                    # Extract search query from response (simplified)
                    query = self._extract_search_query(response)
                    if query:
                        result = await tool.execute(operation="query", query=query)
                        if result.success:
                            context[f"{agent.id}_rag_results"] = result.data
                except Exception as e:
                    self.logger.warning(f"Failed to use RAG tool for agent {agent.id}: {str(e)}")
    
    async def _use_knowledge_graph_if_available(self, agent: Agent, response: str, context: Dict[str, Any]) -> None:
        """Use Knowledge Graph tool if available for the agent."""
        for tool_id in agent.tools:
            tool = tool_registry.get_tool(tool_id)
            if tool and isinstance(tool, type(tool_registry.tool_classes.get("knowledge_graph", type(None)))):
                try:
                    # Simple knowledge graph query (this could be more sophisticated)
                    query = "MATCH (n) RETURN n LIMIT 10"
                    result = await tool.execute(operation="query", cypher_query=query)
                    if result.success:
                        context[f"{agent.id}_kg_results"] = result.data
                except Exception as e:
                    self.logger.warning(f"Failed to use Knowledge Graph tool for agent {agent.id}: {str(e)}")
    
    def _extract_search_query(self, text: str) -> Optional[str]:
        """Extract search query from text (simplified implementation)."""
        # This is a very basic implementation - in practice, you'd want more sophisticated parsing
        import re
        
        # Look for patterns like "search for X" or "find information about Y"
        patterns = [
            r"search for (.+?)(?:\.|$)",
            r"find information about (.+?)(?:\.|$)",
            r"retrieve (.+?)(?:\.|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """List all agent IDs."""
        return list(self.agents.keys())
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent and clean up its tools."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Clean up tools
            for tool_id in agent.tools:
                tool_registry.remove_tool(tool_id)
            
            del self.agents[agent_id]
            self.logger.info(f"Removed agent: {agent_id}")
            return True
        
        return False
