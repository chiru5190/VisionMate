"""
Natural-language command parser module for VisionMate.

Provides deterministic, rule-based parsing of user voice/text queries into
structured intent and target object dictionaries without external ML/LLM dependencies.
"""

import re
import string
from typing import Dict, Any, Optional


class ObjectCommandParser:
    """
    Rule-based command parser for object search and location requests.
    """

    # Mapping common colloquial terms and aliases to standard YOLO/COCO class names
    OBJECT_ALIASES: Dict[str, str] = {
        # Mobile devices
        "phone": "cell phone",
        "phones": "cell phone",
        "mobile": "cell phone",
        "mobiles": "cell phone",
        "mobile phone": "cell phone",
        "mobile phones": "cell phone",
        "cellphone": "cell phone",
        "cellphones": "cell phone",
        "cell phone": "cell phone",
        "cell phones": "cell phone",
        "smartphone": "cell phone",
        "smartphones": "cell phone",

        # Drinkware & Containers
        "bottle": "bottle",
        "bottles": "bottle",
        "water bottle": "bottle",
        "water bottles": "bottle",
        "cup": "cup",
        "cups": "cup",
        "mug": "cup",
        "mugs": "cup",
        "coffee mug": "cup",
        "coffee mugs": "cup",
        "glass": "cup",
        "glasses": "cup",
        "wine glass": "wine glass",
        "wine glasses": "wine glass",

        # Computers & Tech
        "laptop": "laptop",
        "laptops": "laptop",
        "notebook": "laptop",
        "notebooks": "laptop",
        "computer": "laptop",
        "computers": "laptop",
        "keyboard": "keyboard",
        "keyboards": "keyboard",
        "mouse": "mouse",
        "mice": "mouse",
        "remote": "remote",
        "remotes": "remote",
        "remote control": "remote",
        "tv": "tv",
        "tvs": "tv",
        "television": "tv",
        "televisions": "tv",

        # Personal Effects & Bags
        "book": "book",
        "books": "book",
        "backpack": "backpack",
        "backpacks": "backpack",
        "bag": "backpack",
        "bags": "backpack",
        "handbag": "handbag",
        "handbags": "handbag",
        "suitcase": "suitcase",
        "suitcases": "suitcase",
        "keys": "keys",
        "key": "keys",
        "clock": "clock",
        "clocks": "clock",
        "watch": "clock",
        "watches": "clock",
        "scissors": "scissors",
        "vase": "vase",
        "vases": "vase",

        # People
        "person": "person",
        "people": "person",
        "man": "person",
        "men": "person",
        "woman": "person",
        "women": "person",

        # Furniture
        "chair": "chair",
        "chairs": "chair",
        "seat": "chair",
        "seats": "chair",
        "couch": "couch",
        "couches": "couch",
        "sofa": "couch",
        "sofas": "couch",
        "dining table": "dining table",
        "table": "dining table",
        "tables": "dining table",
        "bed": "bed",
        "beds": "bed",
    }

    # Intent trigger phrases signaling an object search or location query
    FIND_TRIGGERS = [
        "find",
        "where is",
        "where's",
        "where can i find",
        "locate",
        "look for",
        "search for",
        "can you find",
        "do you see",
        "is there",
        "can you see",
        "show me",
        "where are",
    ]

    # Stop words and filler terms to ignore during token extraction
    STOP_WORDS = {
        "my", "the", "a", "an", "any", "some", "please", "can", "you",
        "find", "where", "is", "are", "look", "for", "search", "locate",
        "do", "see", "there", "i", "get", "me", "show", "could", "would"
    }

    def _normalize_text(self, command: str) -> str:
        """
        Clean input: convert to lowercase, strip punctuation, and collapse whitespace.
        """
        if not isinstance(command, str):
            return ""

        text = command.lower()
        # Strip punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Collapse whitespace
        return " ".join(text.split())

    def _is_find_command(self, text: str) -> bool:
        """
        Determine if text contains a find/locate intent trigger.
        """
        for trigger in self.FIND_TRIGGERS:
            if trigger in text:
                return True
        return False

    def _extract_object(self, text: str) -> Optional[str]:
        """
        Extract and normalize the target object from command text using alias matching.
        """
        # 1. Match multi-word/single-word aliases ordered by character length descending
        sorted_aliases = sorted(self.OBJECT_ALIASES.keys(), key=len, reverse=True)

        for alias in sorted_aliases:
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text):
                return self.OBJECT_ALIASES[alias]

        # 2. Token-based fallback check after removing stop words
        tokens = text.split()
        candidate_tokens = [t for t in tokens if t not in self.STOP_WORDS]

        if candidate_tokens:
            phrase = " ".join(candidate_tokens)
            if phrase in self.OBJECT_ALIASES:
                return self.OBJECT_ALIASES[phrase]

            for token in candidate_tokens:
                if token in self.OBJECT_ALIASES:
                    return self.OBJECT_ALIASES[token]

        return None

    def parse(self, command: str) -> Dict[str, Any]:
        """
        Parse a command string into a structured intent dictionary.

        Args:
            command (str): Input query string.

        Returns:
            dict: {
                "intent": "find_object" | "unknown",
                "object": str | None
            }
        """
        default_response = {"intent": "unknown", "object": None}

        if not command or not isinstance(command, str):
            return default_response

        normalized_text = self._normalize_text(command)
        if not normalized_text:
            return default_response

        if not self._is_find_command(normalized_text):
            return default_response

        target_object = self._extract_object(normalized_text)
        if not target_object:
            return default_response

        return {
            "intent": "find_object",
            "object": target_object
        }
