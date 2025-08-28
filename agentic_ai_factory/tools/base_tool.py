"""Base tool interface and common functionality."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    """Configuration for a tool."""
    
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    required_env_vars: List[str] = Field(default_factory=list, description="Required environment variables")


class ToolResult(BaseModel):
    """Result from a tool execution."""
    
    success: bool = Field(..., description="Whether the tool execution was successful")
    data: Any = Field(None, description="Tool execution result data")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.name = config.name
        self.description = config.description
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the tool (setup connections, load models, etc.)."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's parameter schema."""
        pass
    
    async def _ensure_initialized(self) -> None:
        """Ensure the tool is initialized before use."""
        if not self._initialized:
            await self.initialize()
            self._initialized = True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
