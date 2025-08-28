"""
Agentic AI Factory - A Python package that simplifies the generation of Agents using LangGraph
with persona-based decomposition and task assignment.
"""

from .core.factory import AgenticFactory
from .personas.persona_manager import PersonaManager
from .agents.agent_builder import AgentBuilder
from .dag.orchestrator import DAGOrchestrator

__version__ = "0.1.0"
__all__ = ["AgenticFactory", "PersonaManager", "AgentBuilder", "DAGOrchestrator"]
