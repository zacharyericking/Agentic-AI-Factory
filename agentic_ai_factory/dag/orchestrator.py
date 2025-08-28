"""DAG orchestrator using LangGraph for coordinating agent execution."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
import time
import networkx as nx
from langgraph.graph import StateGraph, START, END

from ..core.models import Agent, Task, DAGNode, WorkflowResult
from ..agents.agent_builder import AgentBuilder


class DAGOrchestrator:
    """Orchestrates agent execution using a directed acyclic graph with LangGraph."""
    
    def __init__(self, agent_builder: AgentBuilder):
        self.agent_builder = agent_builder
        self.logger = logging.getLogger(__name__)
        self.workflows: Dict[str, Any] = {}
        self.workflow_results: Dict[str, WorkflowResult] = {}
    
    def create_workflow(
        self,
        workflow_id: str,
        agents: List[Agent],
        original_question: str
    ) -> Any:
        """Create a workflow DAG from agents and their task dependencies."""
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(agents)
        
        # Validate DAG (no cycles)
        if not nx.is_directed_acyclic_graph(dependency_graph):
            raise ValueError("Task dependencies contain cycles - not a valid DAG")
        
        # Create LangGraph workflow
        workflow = self._create_langgraph_workflow(workflow_id, agents, dependency_graph, original_question)
        
        self.workflows[workflow_id] = workflow
        self.logger.info(f"Created workflow: {workflow_id} with {len(agents)} agents")
        
        return workflow
    
    async def execute_workflow(self, workflow_id: str) -> WorkflowResult:
        """Execute a workflow and return the result."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        
        workflow = self.workflows[workflow_id]
        start_time = time.time()
        
        try:
            # Initialize workflow state
            initial_state = {
                "workflow_id": workflow_id,
                "context": {},
                "completed_tasks": set(),
                "failed_tasks": set(),
                "results": {},
                "original_question": "",
                "final_answer": ""
            }
            
            # Execute workflow
            final_state = await workflow.ainvoke(initial_state)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = WorkflowResult(
                workflow_id=workflow_id,
                original_question=final_state.get("original_question", ""),
                agents_used=list(final_state.get("results", {}).keys()),
                execution_order=final_state.get("execution_order", []),
                final_answer=final_state.get("final_answer", ""),
                intermediate_results=final_state.get("results", {}),
                execution_time=execution_time,
                success=len(final_state.get("failed_tasks", set())) == 0
            )
            
            self.workflow_results[workflow_id] = result
            self.logger.info(f"Workflow {workflow_id} completed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Workflow {workflow_id} failed after {execution_time:.2f}s: {str(e)}")
            
            result = WorkflowResult(
                workflow_id=workflow_id,
                original_question="",
                agents_used=[],
                execution_order=[],
                final_answer=f"Workflow failed: {str(e)}",
                intermediate_results={},
                execution_time=execution_time,
                success=False
            )
            
            self.workflow_results[workflow_id] = result
            return result
    
    def _build_dependency_graph(self, agents: List[Agent]) -> nx.DiGraph:
        """Build a NetworkX dependency graph from agents and their tasks."""
        graph = nx.DiGraph()
        
        # Add nodes for each agent/task
        for agent in agents:
            graph.add_node(agent.id, agent=agent, task=agent.assigned_task)
        
        # Add edges based on task dependencies
        for agent in agents:
            task = agent.assigned_task
            for dependency_task_id in task.dependencies:
                # Find the agent with the dependency task
                dependency_agent = self._find_agent_by_task_id(agents, dependency_task_id)
                if dependency_agent:
                    graph.add_edge(dependency_agent.id, agent.id)
        
        return graph
    
    def _find_agent_by_task_id(self, agents: List[Agent], task_id: str) -> Optional[Agent]:
        """Find an agent by its assigned task ID."""
        for agent in agents:
            if agent.assigned_task.id == task_id:
                return agent
        return None
    
    def _create_langgraph_workflow(
        self,
        workflow_id: str,
        agents: List[Agent],
        dependency_graph: nx.DiGraph,
        original_question: str
    ) -> Any:
        """Create a LangGraph workflow from the dependency graph."""
        
        # Create the graph
        workflow = StateGraph(dict)
        
        # Add start node
        workflow.add_node("start", self._start_node)
        
        # Add agent execution nodes
        for agent in agents:
            workflow.add_node(agent.id, self._create_agent_executor(agent))
        
        # Add synthesis node
        workflow.add_node("synthesize", self._synthesize_results)
        
        # Add end node
        workflow.add_node("end", self._end_node)
        
        # Add edges based on dependencies
        workflow.add_edge(START, "start")
        
        # Connect start to agents with no dependencies
        root_agents = [agent.id for agent in agents if not list(dependency_graph.predecessors(agent.id))]
        for agent_id in root_agents:
            workflow.add_edge("start", agent_id)
        
        # Add dependency edges
        for edge in dependency_graph.edges():
            workflow.add_edge(edge[0], edge[1])
        
        # Connect leaf agents to synthesis
        leaf_agents = [agent.id for agent in agents if not list(dependency_graph.successors(agent.id))]
        for agent_id in leaf_agents:
            workflow.add_edge(agent_id, "synthesize")
        
        # Connect synthesis to end
        workflow.add_edge("synthesize", "end")
        workflow.add_edge("end", END)
        
        # Compile the workflow
        compiled_workflow = workflow.compile()
        
        return compiled_workflow
    
    def _start_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Start node for the workflow."""
        state["execution_order"] = []
        state["start_time"] = time.time()
        self.logger.info(f"Starting workflow: {state.get('workflow_id', 'unknown')}")
        return state
    
    def _create_agent_executor(self, agent: Agent):
        """Create an agent executor function for LangGraph."""
        
        async def execute_agent(state: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a specific agent."""
            agent_id = agent.id
            
            self.logger.info(f"Executing agent: {agent_id}")
            
            try:
                # Execute the agent's task
                result = await self.agent_builder.execute_agent_task(agent_id, state["context"])
                
                # Update state
                state["results"][agent_id] = result
                state["execution_order"].append(agent_id)
                
                if result["success"]:
                    state["completed_tasks"].add(agent.assigned_task.id)
                    # Add result to context for subsequent agents
                    state["context"][f"{agent_id}_result"] = result["result"]
                else:
                    state["failed_tasks"].add(agent.assigned_task.id)
                
                self.logger.info(f"Agent {agent_id} completed: {'success' if result['success'] else 'failed'}")
                
            except Exception as e:
                self.logger.error(f"Agent {agent_id} execution failed: {str(e)}")
                state["failed_tasks"].add(agent.assigned_task.id)
                state["results"][agent_id] = {
                    "agent_id": agent_id,
                    "success": False,
                    "error": str(e)
                }
            
            return state
        
        return execute_agent
    
    async def _synthesize_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from all agents to create final answer."""
        
        self.logger.info("Synthesizing results from all agents")
        
        try:
            # Collect all successful results
            successful_results = []
            for agent_id, result in state["results"].items():
                if result.get("success", False):
                    successful_results.append(f"Agent {agent_id}: {result.get('result', 'No result')}")
            
            if successful_results:
                # Create a synthesis prompt
                synthesis_prompt = f"""
                Original Question: {state.get('original_question', 'Unknown')}
                
                Results from specialized agents:
                {chr(10).join(successful_results)}
                
                Please synthesize these results into a comprehensive final answer that addresses the original question.
                """
                
                # Use the first available agent's LLM for synthesis
                # In a real implementation, you might want a dedicated synthesis agent
                if self.agent_builder.agents:
                    from langchain.schema import HumanMessage
                    response = await self.agent_builder.llm.ainvoke([HumanMessage(content=synthesis_prompt)])
                    state["final_answer"] = response.content
                else:
                    state["final_answer"] = "\\n".join(successful_results)
            else:
                state["final_answer"] = "No successful results to synthesize."
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {str(e)}")
            state["final_answer"] = f"Synthesis failed: {str(e)}"
        
        return state
    
    def _end_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """End node for the workflow."""
        end_time = time.time()
        start_time = state.get("start_time", end_time)
        execution_time = end_time - start_time
        
        self.logger.info(f"Workflow {state.get('workflow_id', 'unknown')} completed in {execution_time:.2f}s")
        
        state["end_time"] = end_time
        state["total_execution_time"] = execution_time
        
        return state
    
    def get_workflow_result(self, workflow_id: str) -> Optional[WorkflowResult]:
        """Get the result of a completed workflow."""
        return self.workflow_results.get(workflow_id)
    
    def list_workflows(self) -> List[str]:
        """List all workflow IDs."""
        return list(self.workflows.keys())
    
    def remove_workflow(self, workflow_id: str) -> bool:
        """Remove a workflow and its results."""
        removed = False
        
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            removed = True
        
        if workflow_id in self.workflow_results:
            del self.workflow_results[workflow_id]
            removed = True
        
        if removed:
            self.logger.info(f"Removed workflow: {workflow_id}")
        
        return removed
    
    def visualize_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow structure for visualization."""
        if workflow_id not in self.workflows:
            return None
        
        # This would return data suitable for visualization
        # Implementation depends on your visualization needs
        return {
            "workflow_id": workflow_id,
            "nodes": [],  # Node information
            "edges": [],  # Edge information
            "status": "ready"
        }
