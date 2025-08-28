"""Knowledge Graph tool implementation using Neo4j."""

import logging
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver
import json

from .base_tool import BaseTool, ToolConfig, ToolResult


class KnowledgeGraphTool(BaseTool):
    """Knowledge Graph tool for graph-based knowledge storage and retrieval."""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.driver: Optional[Driver] = None
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.uri = config.parameters.get("uri", "bolt://localhost:7687")
        self.username = config.parameters.get("username", "neo4j")
        self.password = config.parameters.get("password", "password")
        self.database = config.parameters.get("database", "neo4j")
    
    async def initialize(self) -> None:
        """Initialize the Knowledge Graph tool with Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            self.logger.info("Successfully connected to Neo4j database")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Knowledge Graph tool: {str(e)}")
            raise
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute Knowledge Graph operations."""
        await self._ensure_initialized()
        
        operation = kwargs.get("operation", "query")
        
        try:
            if operation == "create_node":
                return await self._create_node(kwargs)
            elif operation == "create_relationship":
                return await self._create_relationship(kwargs)
            elif operation == "query":
                return await self._query_graph(kwargs)
            elif operation == "find_path":
                return await self._find_path(kwargs)
            elif operation == "get_neighbors":
                return await self._get_neighbors(kwargs)
            elif operation == "add_knowledge":
                return await self._add_knowledge(kwargs)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )
        
        except Exception as e:
            self.logger.error(f"Knowledge Graph tool execution failed: {str(e)}")
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    async def _create_node(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Create a node in the knowledge graph."""
        label = kwargs.get("label", "Entity")
        properties = kwargs.get("properties", {})
        
        if not properties:
            return ToolResult(
                success=False,
                error="Node properties are required"
            )
        
        with self.driver.session(database=self.database) as session:
            # Create node with properties
            query = f"CREATE (n:{label} $properties) RETURN n"
            result = session.run(query, properties=properties)
            node = result.single()["n"]
            
            return ToolResult(
                success=True,
                data={
                    "node_id": node.id,
                    "labels": list(node.labels),
                    "properties": dict(node)
                }
            )
    
    async def _create_relationship(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Create a relationship between two nodes."""
        from_node_id = kwargs.get("from_node_id")
        to_node_id = kwargs.get("to_node_id")
        relationship_type = kwargs.get("relationship_type", "RELATED_TO")
        properties = kwargs.get("properties", {})
        
        if not from_node_id or not to_node_id:
            return ToolResult(
                success=False,
                error="Both from_node_id and to_node_id are required"
            )
        
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (a), (b)
            WHERE id(a) = $from_id AND id(b) = $to_id
            CREATE (a)-[r:{relationship_type} $properties]->(b)
            RETURN r
            """
            result = session.run(
                query,
                from_id=from_node_id,
                to_id=to_node_id,
                properties=properties
            )
            relationship = result.single()
            
            if relationship:
                rel = relationship["r"]
                return ToolResult(
                    success=True,
                    data={
                        "relationship_id": rel.id,
                        "type": rel.type,
                        "properties": dict(rel)
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    error="Failed to create relationship (nodes might not exist)"
                )
    
    async def _query_graph(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Execute a Cypher query on the knowledge graph."""
        cypher_query = kwargs.get("cypher_query", "")
        parameters = kwargs.get("parameters", {})
        
        if not cypher_query:
            return ToolResult(
                success=False,
                error="Cypher query is required"
            )
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query, parameters)
            
            records = []
            for record in result:
                record_dict = {}
                for key in record.keys():
                    value = record[key]
                    if hasattr(value, 'id'):  # Node or Relationship
                        record_dict[key] = {
                            "id": value.id,
                            "labels": list(value.labels) if hasattr(value, 'labels') else None,
                            "type": value.type if hasattr(value, 'type') else None,
                            "properties": dict(value)
                        }
                    else:
                        record_dict[key] = value
                records.append(record_dict)
            
            return ToolResult(
                success=True,
                data={
                    "results": records,
                    "count": len(records)
                }
            )
    
    async def _find_path(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Find paths between two nodes."""
        start_node_id = kwargs.get("start_node_id")
        end_node_id = kwargs.get("end_node_id")
        max_depth = kwargs.get("max_depth", 5)
        
        if not start_node_id or not end_node_id:
            return ToolResult(
                success=False,
                error="Both start_node_id and end_node_id are required"
            )
        
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (start), (end)
            WHERE id(start) = $start_id AND id(end) = $end_id
            MATCH path = shortestPath((start)-[*1..{max_depth}]-(end))
            RETURN path
            """
            result = session.run(query, start_id=start_node_id, end_id=end_node_id)
            
            paths = []
            for record in result:
                path = record["path"]
                path_data = {
                    "nodes": [{"id": node.id, "labels": list(node.labels), "properties": dict(node)} 
                             for node in path.nodes],
                    "relationships": [{"id": rel.id, "type": rel.type, "properties": dict(rel)} 
                                   for rel in path.relationships],
                    "length": len(path.relationships)
                }
                paths.append(path_data)
            
            return ToolResult(
                success=True,
                data={
                    "paths": paths,
                    "count": len(paths)
                }
            )
    
    async def _get_neighbors(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Get neighboring nodes of a given node."""
        node_id = kwargs.get("node_id")
        depth = kwargs.get("depth", 1)
        relationship_types = kwargs.get("relationship_types", [])
        
        if not node_id:
            return ToolResult(
                success=False,
                error="node_id is required"
            )
        
        with self.driver.session(database=self.database) as session:
            # Build relationship type filter
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"
            
            query = f"""
            MATCH (start)-[r{rel_filter}*1..{depth}]-(neighbor)
            WHERE id(start) = $node_id
            RETURN DISTINCT neighbor, r
            """
            result = session.run(query, node_id=node_id)
            
            neighbors = []
            for record in result:
                neighbor = record["neighbor"]
                relationships = record["r"]
                
                neighbor_data = {
                    "node": {
                        "id": neighbor.id,
                        "labels": list(neighbor.labels),
                        "properties": dict(neighbor)
                    },
                    "relationships": []
                }
                
                if isinstance(relationships, list):
                    for rel in relationships:
                        neighbor_data["relationships"].append({
                            "id": rel.id,
                            "type": rel.type,
                            "properties": dict(rel)
                        })
                
                neighbors.append(neighbor_data)
            
            return ToolResult(
                success=True,
                data={
                    "neighbors": neighbors,
                    "count": len(neighbors)
                }
            )
    
    async def _add_knowledge(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Add structured knowledge to the graph."""
        entities = kwargs.get("entities", [])
        relationships = kwargs.get("relationships", [])
        
        created_nodes = []
        created_relationships = []
        
        with self.driver.session(database=self.database) as session:
            # Create entities
            for entity in entities:
                label = entity.get("label", "Entity")
                properties = entity.get("properties", {})
                
                query = f"CREATE (n:{label} $properties) RETURN n"
                result = session.run(query, properties=properties)
                node = result.single()["n"]
                created_nodes.append({
                    "id": node.id,
                    "labels": list(node.labels),
                    "properties": dict(node)
                })
            
            # Create relationships
            for rel in relationships:
                from_props = rel.get("from_properties", {})
                to_props = rel.get("to_properties", {})
                rel_type = rel.get("type", "RELATED_TO")
                rel_props = rel.get("properties", {})
                
                # Find or create nodes and relationship
                query = f"""
                MERGE (a:{rel.get('from_label', 'Entity')} $from_props)
                MERGE (b:{rel.get('to_label', 'Entity')} $to_props)
                CREATE (a)-[r:{rel_type} $rel_props]->(b)
                RETURN r
                """
                result = session.run(
                    query,
                    from_props=from_props,
                    to_props=to_props,
                    rel_props=rel_props
                )
                relationship = result.single()["r"]
                created_relationships.append({
                    "id": relationship.id,
                    "type": relationship.type,
                    "properties": dict(relationship)
                })
        
        return ToolResult(
            success=True,
            data={
                "created_nodes": created_nodes,
                "created_relationships": created_relationships,
                "total_nodes": len(created_nodes),
                "total_relationships": len(created_relationships)
            }
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's parameter schema."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create_node", "create_relationship", "query", "find_path", "get_neighbors", "add_knowledge"],
                    "description": "Operation to perform"
                },
                "label": {
                    "type": "string",
                    "description": "Node label"
                },
                "properties": {
                    "type": "object",
                    "description": "Node or relationship properties"
                },
                "cypher_query": {
                    "type": "string",
                    "description": "Cypher query to execute"
                },
                "parameters": {
                    "type": "object",
                    "description": "Query parameters"
                },
                "from_node_id": {
                    "type": "integer",
                    "description": "Source node ID for relationship"
                },
                "to_node_id": {
                    "type": "integer",
                    "description": "Target node ID for relationship"
                },
                "relationship_type": {
                    "type": "string",
                    "description": "Type of relationship"
                },
                "node_id": {
                    "type": "integer",
                    "description": "Node ID for operations"
                },
                "depth": {
                    "type": "integer",
                    "description": "Search depth",
                    "default": 1
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Entities to add to the graph"
                },
                "relationships": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Relationships to add to the graph"
                }
            },
            "required": ["operation"]
        }
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
