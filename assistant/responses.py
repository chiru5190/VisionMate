"""
Natural-language response generator module for VisionMate.

Generates concise, human-friendly text feedback for target object queries
based on parsed command intents, live perception pipeline detections, and query memory history.
"""

from typing import Dict, Any, List, Optional


class ObjectResponseGenerator:
    """
    Deterministic response generator converting perception metadata and user intent
    into natural spoken/written feedback.
    """

    # Human-friendly distance category phrases
    DISTANCE_PHRASES: Dict[str, str] = {
        "far": "far away",
        "medium": "at medium distance",
        "close": "nearby",
        "very_close": "very close"
    }

    def _get_plural(self, word: str) -> str:
        """
        Return the plural form of an object label for multi-object responses.
        """
        w = word.strip().lower()
        if w == "person":
            return "people"
        elif w.endswith("y") and not w.endswith(("ay", "ey", "oy", "uy")):
            return w[:-1] + "ies"
        elif w.endswith(("s", "sh", "ch", "x", "z")):
            return w + "es"
        else:
            return w + "s"

    def _format_position(self, pos_dict: Optional[Dict[str, Any]]) -> str:
        """
        Convert 2D spatial position metadata into a natural language phrase.
        """
        if not isinstance(pos_dict, dict):
            return "in view"

        horiz = str(pos_dict.get("horizontal", "center")).lower()
        vert = str(pos_dict.get("vertical", "middle")).lower()

        vert_prefix = ""
        if vert == "top":
            vert_prefix = "upper "
        elif vert == "bottom":
            vert_prefix = "lower "

        if horiz == "left":
            return f"on the {vert_prefix}left".replace("  ", " ")
        elif horiz == "right":
            return f"on the {vert_prefix}right".replace("  ", " ")
        else:
            if vert_prefix:
                return f"in the {vert_prefix}center".replace("  ", " ")
            return "in the center"

    def _format_distance(self, dist_dict: Optional[Dict[str, Any]]) -> str:
        """
        Convert relative distance metadata into a natural language phrase.
        """
        if not isinstance(dist_dict, dict):
            return ""
        category = str(dist_dict.get("category", "")).lower()
        return self.DISTANCE_PHRASES.get(category, "")

    def _format_single_match(self, target_object: str, detection: Dict[str, Any]) -> str:
        """
        Format a single object detection into a natural language sentence.
        """
        pos_phrase = self._format_position(detection.get("position"))
        dist_phrase = self._format_distance(detection.get("distance"))

        if dist_phrase:
            return f"I found your {target_object.lower()} {pos_phrase}, {dist_phrase}."
        else:
            return f"I found your {target_object.lower()} {pos_phrase}."

    def _format_multiple_matches(self, target_object: str, matches: List[Dict[str, Any]]) -> str:
        """
        Format a multi-object response when several instances of the requested object are detected.
        """
        count = len(matches)
        plural_name = self._get_plural(target_object)

        if count == 2:
            pos1 = self._format_position(matches[0].get("position"))
            pos2 = self._format_position(matches[1].get("position"))
            return f"I found 2 {plural_name}. One is {pos1} and one is {pos2}."
        else:
            return f"I found {count} {plural_name}."

    def generate(
        self,
        command: Optional[Dict[str, Any]],
        detections: Optional[List[Dict[str, Any]]],
        query_results: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate a natural language text response for a given command, detections, and query memory results.

        Args:
            command (dict): Parsed command dict from ObjectCommandParser (e.g. {"intent": "find_object", "object": "bottle"}).
            detections (list, optional): List of live detection dictionaries from perception pipeline.
            query_results (list, optional): Structured match results from ObjectQueryMemory.find_object().

        Returns:
            str: Human-friendly text feedback.
        """
        if not isinstance(command, dict):
            return "I'm not sure what you're looking for."

        intent = command.get("intent")
        target_object = command.get("object")

        if intent != "find_object" or not target_object or not isinstance(target_object, str):
            return "I'm not sure what you're looking for."

        clean_target = target_object.strip().lower()

        # Handle ObjectQueryMemory results if available
        if query_results is not None:
            if not isinstance(query_results, list) or len(query_results) == 0:
                return f"I can't see your {clean_target} right now."

            if len(query_results) == 1:
                match = query_results[0]
                pos_phrase = self._format_position(match.get("position"))
                dist_phrase = self._format_distance(match.get("distance"))
                is_current = match.get("is_in_current_frame", False)

                verb = "I found" if is_current else "I recently saw"
                if dist_phrase:
                    return f"{verb} your {clean_target} {pos_phrase}, {dist_phrase}."
                else:
                    return f"{verb} your {clean_target} {pos_phrase}."
            else:
                count = len(query_results)
                plural_name = self._get_plural(clean_target)
                all_current = all(m.get("is_in_current_frame", False) for m in query_results)
                verb = "I found" if all_current else "I recently saw"

                if count == 2:
                    pos1 = self._format_position(query_results[0].get("position"))
                    pos2 = self._format_position(query_results[1].get("position"))
                    return f"{verb} 2 {plural_name}. One is {pos1} and one is {pos2}."
                else:
                    return f"{verb} {count} {plural_name}."

        # Fallback to direct detections list filtering if query_results is None
        if not isinstance(detections, list) or len(detections) == 0:
            return f"I can't see your {clean_target} right now."

        matches = []
        for d in detections:
            if isinstance(d, dict):
                class_name = str(d.get("class_name", "")).strip().lower()
                if class_name == clean_target:
                    matches.append(d)

        if not matches:
            return f"I can't see your {clean_target} right now."

        matches.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)

        if len(matches) == 1:
            return self._format_single_match(clean_target, matches[0])
        else:
            return self._format_multiple_matches(clean_target, matches)
