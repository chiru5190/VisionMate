"""
Assistant controller module for VisionMate.

Coordinates natural language command parsing and response generation
with live perception pipeline detections and short-term query memory.
"""

from typing import Dict, Any, List, Optional
from assistant.commands import ObjectCommandParser
from assistant.responses import ObjectResponseGenerator
from assistant.query_memory import ObjectQueryMemory


class AssistantController:
    """
    High-level controller coordinating query parsing, query memory resolution, and response formatting.
    """

    def __init__(self, query_memory: Optional[ObjectQueryMemory] = None):
        """
        Initialize command parser, response generator, and query memory.
        """
        self.command_parser = ObjectCommandParser()
        self.response_generator = ObjectResponseGenerator()
        self.query_memory = query_memory if query_memory is not None else ObjectQueryMemory(max_history_frames=30)
        self.last_command: Optional[str] = None
        self.last_response: Optional[str] = None

    def process_command(
        self,
        command: Optional[str],
        detections: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Process a user query string against query memory and live detections, returning a text response.

        Args:
            command (str): Raw user query/command string.
            detections (list, optional): List of live enriched detection dictionaries.

        Returns:
            str: Generated natural language feedback text.
        """
        # Parse command string into structured intent dict
        parsed_command = self.command_parser.parse(command)

        # Update query memory with live detections if provided
        if detections is not None:
            self.query_memory.update(detections)

        # Search query memory if an object target was requested
        target_object = parsed_command.get("object")
        query_results = None
        if target_object:
            query_results = self.query_memory.find_object(target_object)

        # Generate response using parsed intent, live detections, and query memory results
        response = self.response_generator.generate(
            command=parsed_command,
            detections=detections,
            query_results=query_results
        )

        # Track last processed command and response
        self.last_command = command
        self.last_response = response

        return response
