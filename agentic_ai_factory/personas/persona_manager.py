"""Persona management system with Gemini integration for persona decomposition."""

import logging
from typing import Dict, List, Optional
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.models import Persona, SubPersona, Task, TaskType, ToolType


class PersonaManager:
    """Manages persona registration and decomposition using Gemini."""
    
    def __init__(self, gemini_api_key: str, model_name: str = "gemini-1.5-pro"):
        """Initialize the PersonaManager with Gemini API key."""
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self.personas: Dict[str, Persona] = {}
        self.sub_personas: Dict[str, List[SubPersona]] = {}
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0.7
        )
        
        self.logger = logging.getLogger(__name__)
    
    def register_persona(self, persona: Persona) -> None:
        """Register a new persona."""
        self.personas[persona.name] = persona
        self.logger.info(f"Registered persona: {persona.name}")
    
    def get_persona(self, name: str) -> Optional[Persona]:
        """Get a registered persona by name."""
        return self.personas.get(name)
    
    def list_personas(self) -> List[str]:
        """List all registered persona names."""
        return list(self.personas.keys())
    
    async def decompose_persona(self, persona_name: str, main_task: str) -> List[SubPersona]:
        """Decompose a persona into component sub-personas using Gemini."""
        if persona_name not in self.personas:
            raise ValueError(f"Persona '{persona_name}' not found")
        
        persona = self.personas[persona_name]
        
        # Create prompt for Gemini to decompose the persona
        prompt = self._create_decomposition_prompt(persona, main_task)
        
        try:
            # Call Gemini to decompose the persona
            response = await self.llm.ainvoke(prompt)
            sub_personas = self._parse_decomposition_response(response.content, persona_name)
            
            # Store the decomposed sub-personas
            self.sub_personas[persona_name] = sub_personas
            
            self.logger.info(f"Decomposed persona '{persona_name}' into {len(sub_personas)} sub-personas")
            return sub_personas
            
        except Exception as e:
            self.logger.error(f"Error decomposing persona '{persona_name}': {str(e)}")
            raise
    
    async def assign_tasks_to_subpersonas(self, persona_name: str, main_task: str) -> List[Task]:
        """Use Gemini to break down the main task and assign to sub-personas."""
        if persona_name not in self.sub_personas:
            raise ValueError(f"No sub-personas found for '{persona_name}'. Decompose first.")
        
        sub_personas = self.sub_personas[persona_name]
        
        # Create prompt for task breakdown and assignment
        prompt = self._create_task_assignment_prompt(sub_personas, main_task)
        
        try:
            response = await self.llm.ainvoke(prompt)
            tasks = self._parse_task_assignment_response(response.content, sub_personas)
            
            # Update sub-personas with assigned tasks
            for i, task in enumerate(tasks):
                if i < len(sub_personas):
                    sub_personas[i].assigned_task = task.description
                    sub_personas[i].task_type = task.task_type
                    sub_personas[i].required_tools = task.required_tools
            
            self.logger.info(f"Assigned {len(tasks)} tasks to sub-personas of '{persona_name}'")
            return tasks
            
        except Exception as e:
            self.logger.error(f"Error assigning tasks to sub-personas of '{persona_name}': {str(e)}")
            raise
    
    def get_sub_personas(self, persona_name: str) -> List[SubPersona]:
        """Get sub-personas for a given persona."""
        return self.sub_personas.get(persona_name, [])
    
    def _create_decomposition_prompt(self, persona: Persona, main_task: str) -> str:
        """Create a prompt for Gemini to decompose a persona."""
        return f"""
You are an expert in persona analysis and task decomposition. Given the following persona and main task, 
decompose the persona into 3-5 specialized sub-personas that would work together to accomplish the task.

Main Persona:
- Name: {persona.name}
- Description: {persona.description}
- Expertise Areas: {', '.join(persona.expertise_areas)}
- Personality Traits: {', '.join(persona.personality_traits)}
- Communication Style: {persona.communication_style}
- Preferred Tools: {', '.join([tool.value for tool in persona.preferred_tools])}
- Constraints: {', '.join(persona.constraints)}

Main Task: {main_task}

Please decompose this persona into specialized sub-personas. For each sub-persona, provide:
1. Name (should reflect their specific role)
2. Description (specific role and responsibilities)
3. Specific Role (within the context of the main task)
4. Capabilities (what they can do)

Format your response as a JSON array of objects with the following structure:
[
  {{
    "name": "Sub-persona name",
    "description": "Detailed description of the sub-persona",
    "specific_role": "Their specific role in accomplishing the main task",
    "capabilities": ["capability1", "capability2", "capability3"]
  }}
]

Focus on creating complementary sub-personas that together can handle all aspects of the main task.
"""
    
    def _create_task_assignment_prompt(self, sub_personas: List[SubPersona], main_task: str) -> str:
        """Create a prompt for Gemini to assign tasks to sub-personas."""
        sub_persona_info = []
        for sp in sub_personas:
            sub_persona_info.append(f"- {sp.name}: {sp.description} (Role: {sp.specific_role})")
        
        return f"""
You are an expert task planner. Given the following sub-personas and main task, break down the main task 
into specific subtasks and assign them to the appropriate sub-personas.

Sub-personas:
{chr(10).join(sub_persona_info)}

Main Task: {main_task}

Please create specific tasks for each sub-persona. For each task, provide:
1. ID (unique identifier)
2. Description (what needs to be done)
3. Task Type (one of: research, analysis, synthesis, validation, execution, communication)
4. Priority (1-10, where 10 is highest)
5. Dependencies (list of task IDs this task depends on, if any)
6. Required Tools (list from: rag, knowledge_graph, web_search, database, api, computation)
7. Expected Output (what the task should produce)

Available Task Types: research, analysis, synthesis, validation, execution, communication
Available Tool Types: rag, knowledge_graph, web_search, database, api, computation

Format your response as a JSON array of task objects:
[
  {{
    "id": "task_1",
    "description": "Specific task description",
    "task_type": "research",
    "priority": 5,
    "dependencies": [],
    "required_tools": ["rag", "web_search"],
    "expected_output": "What this task should produce"
  }}
]

Ensure tasks are logically sequenced and dependencies are properly set.
"""
    
    def _parse_decomposition_response(self, response: str, parent_persona: str) -> List[SubPersona]:
        """Parse Gemini's response for persona decomposition."""
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            raise ValueError("Could not find JSON in Gemini response")
        
        try:
            data = json.loads(json_match.group())
            sub_personas = []
            
            for item in data:
                sub_persona = SubPersona(
                    name=item['name'],
                    parent_persona=parent_persona,
                    description=item['description'],
                    specific_role=item['specific_role'],
                    capabilities=item['capabilities']
                )
                sub_personas.append(sub_persona)
            
            return sub_personas
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Error parsing decomposition response: {str(e)}")
            raise ValueError(f"Invalid JSON format in Gemini response: {str(e)}")
    
    def _parse_task_assignment_response(self, response: str, sub_personas: List[SubPersona]) -> List[Task]:
        """Parse Gemini's response for task assignment."""
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            raise ValueError("Could not find JSON in Gemini response")
        
        try:
            data = json.loads(json_match.group())
            tasks = []
            
            for item in data:
                task = Task(
                    id=item['id'],
                    description=item['description'],
                    task_type=TaskType(item['task_type']),
                    priority=item.get('priority', 1),
                    dependencies=item.get('dependencies', []),
                    required_tools=[ToolType(tool) for tool in item.get('required_tools', [])],
                    expected_output=item['expected_output']
                )
                tasks.append(task)
            
            return tasks
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Error parsing task assignment response: {str(e)}")
            raise ValueError(f"Invalid JSON format in Gemini response: {str(e)}")
