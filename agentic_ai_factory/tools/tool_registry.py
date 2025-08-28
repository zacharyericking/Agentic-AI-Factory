"""Tool registry for managing available tools."""

import logging
from typing import Dict, List, Optional, Type
from .base_tool import BaseTool, ToolConfig
from .rag_tool import RAGTool
from .knowledge_graph_tool import KnowledgeGraphTool


class ToolRegistry:
    """Registry for managing and instantiating tools."""
    
    def __init__(self):
        self.tool_classes: Dict[str, Type[BaseTool]] = {}
        self.tool_instances: Dict[str, BaseTool] = {}
        self.logger = logging.getLogger(__name__)
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self) -> None:
        """Register built-in tool classes."""
        self.register_tool_class("rag", RAGTool)
        self.register_tool_class("knowledge_graph", KnowledgeGraphTool)
    
    def register_tool_class(self, tool_type: str, tool_class: Type[BaseTool]) -> None:
        """Register a tool class."""
        self.tool_classes[tool_type] = tool_class
        self.logger.info(f"Registered tool class: {tool_type}")
    
    def create_tool(self, tool_type: str, tool_id: str, config: ToolConfig) -> BaseTool:
        """Create a tool instance."""
        if tool_type not in self.tool_classes:
            raise ValueError(f"Unknown tool type: {tool_type}")
        
        tool_class = self.tool_classes[tool_type]
        tool_instance = tool_class(config)
        self.tool_instances[tool_id] = tool_instance
        
        self.logger.info(f"Created tool instance: {tool_id} (type: {tool_type})")
        return tool_instance
    
    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """Get a tool instance by ID."""
        return self.tool_instances.get(tool_id)
    
    def list_tool_types(self) -> List[str]:
        """List available tool types."""
        return list(self.tool_classes.keys())
    
    def list_tool_instances(self) -> List[str]:
        """List created tool instances."""
        return list(self.tool_instances.keys())
    
    def remove_tool(self, tool_id: str) -> bool:
        """Remove a tool instance."""
        if tool_id in self.tool_instances:
            tool = self.tool_instances[tool_id]
            if hasattr(tool, 'close'):
                tool.close()
            del self.tool_instances[tool_id]
            self.logger.info(f"Removed tool instance: {tool_id}")
            return True
        return False
    
    def get_tool_schema(self, tool_type: str) -> Optional[Dict]:
        """Get schema for a tool type."""
        if tool_type not in self.tool_classes:
            return None
        
        # Create a temporary instance to get schema
        temp_config = ToolConfig(name=f"temp_{tool_type}", description="Temporary instance")
        temp_instance = self.tool_classes[tool_type](temp_config)
        return temp_instance.get_schema()


# Global tool registry instance
tool_registry = ToolRegistry()
