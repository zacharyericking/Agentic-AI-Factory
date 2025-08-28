"""Core data models for the Agentic AI Factory."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class TaskType(str, Enum):
    """Types of tasks that can be assigned to agents."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    EXECUTION = "execution"
    COMMUNICATION = "communication"


class ToolType(str, Enum):
    """Types of tools available to agents."""
    RAG = "rag"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    WEB_SEARCH = "web_search"
    DATABASE = "database"
    API = "api"
    COMPUTATION = "computation"


class Persona(BaseModel):
    """Represents a persona that defines agent behavior and capabilities."""
    
    name: str = Field(..., description="Name of the persona")
    description: str = Field(..., description="Detailed description of the persona")
    expertise_areas: List[str] = Field(default_factory=list, description="Areas of expertise")
    personality_traits: List[str] = Field(default_factory=list, description="Key personality traits")
    communication_style: str = Field(default="professional", description="Communication style")
    preferred_tools: List[ToolType] = Field(default_factory=list, description="Preferred tool types")
    constraints: List[str] = Field(default_factory=list, description="Operational constraints")
    
    class Config:
        use_enum_values = True


class SubPersona(BaseModel):
    """Represents a decomposed sub-persona for specific tasks."""
    
    name: str = Field(..., description="Name of the sub-persona")
    parent_persona: str = Field(..., description="Name of the parent persona")
    description: str = Field(..., description="Description of the sub-persona")
    specific_role: str = Field(..., description="Specific role within the task")
    capabilities: List[str] = Field(default_factory=list, description="Specific capabilities")
    assigned_task: Optional[str] = Field(None, description="Assigned task description")
    task_type: Optional[TaskType] = Field(None, description="Type of assigned task")
    required_tools: List[ToolType] = Field(default_factory=list, description="Required tools for the task")
    
    class Config:
        use_enum_values = True


class Task(BaseModel):
    """Represents a task that can be assigned to an agent."""
    
    id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="Task description")
    task_type: TaskType = Field(..., description="Type of task")
    priority: int = Field(default=1, description="Task priority (1-10)")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    required_tools: List[ToolType] = Field(default_factory=list, description="Required tools")
    expected_output: str = Field(..., description="Description of expected output")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    class Config:
        use_enum_values = True


class Agent(BaseModel):
    """Represents an agent instance."""
    
    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    persona: SubPersona = Field(..., description="Agent's persona")
    assigned_task: Task = Field(..., description="Assigned task")
    tools: List[str] = Field(default_factory=list, description="Available tool instances")
    status: str = Field(default="idle", description="Current agent status")
    
    class Config:
        use_enum_values = True


class DAGNode(BaseModel):
    """Represents a node in the directed acyclic graph."""
    
    id: str = Field(..., description="Node identifier")
    agent_id: str = Field(..., description="Associated agent ID")
    task_id: str = Field(..., description="Associated task ID")
    dependencies: List[str] = Field(default_factory=list, description="Dependent node IDs")
    status: str = Field(default="pending", description="Node execution status")
    result: Optional[Dict[str, Any]] = Field(None, description="Node execution result")


class WorkflowResult(BaseModel):
    """Represents the result of a workflow execution."""
    
    workflow_id: str = Field(..., description="Workflow identifier")
    original_question: str = Field(..., description="Original question/task")
    agents_used: List[str] = Field(default_factory=list, description="List of agent IDs used")
    execution_order: List[str] = Field(default_factory=list, description="Execution order of tasks")
    final_answer: str = Field(..., description="Final synthesized answer")
    intermediate_results: Dict[str, Any] = Field(default_factory=dict, description="Intermediate results")
    execution_time: float = Field(..., description="Total execution time in seconds")
    success: bool = Field(..., description="Whether the workflow completed successfully")
