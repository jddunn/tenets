"""Stopword management for different contexts.

This module manages multiple stopword sets for different purposes:
- Minimal set for code search (preserve accuracy)
- Aggressive set for prompt parsing (extract intent)
- Intent action words for filtering from keyword matching
- Custom sets for specific domains

The intent action words feature filters common action words (like "fix", "debug", "implement")
from keyword matching after intent detection, preventing generic intent words from affecting
file ranking while preserving domain-specific terms for accurate matching.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from tenets.utils.logger import get_logger


@dataclass
class StopwordSet:
    """A set of stopwords with metadata.

    Attributes:
        name: Name of this stopword set
        words: Set of stopword strings
        description: What this set is used for
        source_file: Path to source file
    """

    name: str
    words: Set[str]
    description: str
    source_file: Optional[Path] = None

    def __contains__(self, word: str) -> bool:
        """Check if word is in stopword set."""
        return word.lower() in self.words

    def filter(self, words: List[str]) -> List[str]:
        """Filter stopwords from word list.

        Args:
            words: List of words to filter

        Returns:
            Filtered list without stopwords
        """
        return [w for w in words if w.lower() not in self.words]


class StopwordManager:
    """Manages multiple stopword sets for different contexts."""

    # Default data directory relative to package
    DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "stopwords"

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize stopword manager.

        Args:
            data_dir: Directory containing stopword files
        """
        self.logger = get_logger(__name__)
        self.data_dir = data_dir or self.DEFAULT_DATA_DIR
        self._sets: dict[str, StopwordSet] = {}

        # Load default sets
        self._load_default_sets()

    def _load_default_sets(self):
        """Load default stopword sets from data files.
        
        Loads multiple stopword sets for different contexts:
        - code: Minimal set for code search accuracy
        - prompt: Aggressive set for prompt parsing
        - intent_actions: Action words from intent detection to filter from keyword matching
        """
        # Code stopwords (minimal)
        code_file = self.data_dir / "code_minimal.txt"
        if code_file.exists():
            self._sets["code"] = self._load_set_from_file(
                code_file, name="code", description="Minimal stopwords for code search"
            )
        else:
            # Fallback to hardcoded minimal set
            self._sets["code"] = StopwordSet(
                name="code",
                words={
                    "the",
                    "a",
                    "an",
                    "is",
                    "are",
                    "was",
                    "were",
                    "be",
                    "been",
                    "to",
                    "of",
                    "and",
                    "or",
                    "in",
                    "on",
                    "at",
                    "by",
                    "for",
                    "with",
                },
                description="Minimal stopwords for code search (fallback)",
            )

        # Prompt stopwords (aggressive)
        prompt_file = self.data_dir / "prompt_aggressive.txt"
        if prompt_file.exists():
            self._sets["prompt"] = self._load_set_from_file(
                prompt_file, name="prompt", description="Aggressive stopwords for prompt parsing"
            )
        else:
            # Fallback to moderate set
            self._sets["prompt"] = StopwordSet(
                name="prompt",
                words=self._sets["code"].words
                | {
                    "i",
                    "me",
                    "my",
                    "you",
                    "your",
                    "he",
                    "she",
                    "it",
                    "we",
                    "they",
                    "this",
                    "that",
                    "these",
                    "those",
                    "have",
                    "has",
                    "had",
                    "do",
                    "does",
                    "did",
                    "will",
                    "would",
                    "could",
                    "should",
                    "may",
                    "might",
                    "must",
                    "can",
                    "need",
                    "want",
                    "please",
                    "help",
                    "make",
                    "create",
                    "implement",
                    "add",
                    "get",
                    "set",
                    "show",
                    "find",
                    "use",
                    "using",
                },
                description="Aggressive stopwords for prompt parsing (fallback)",
            )

        # Load intent action words that should be filtered from keyword matching
        self._load_intent_action_words()

    def _load_intent_action_words(self):
        """Load intent action words from intent patterns file.
        
        These are common action words from intent detection that should be
        filtered from keyword matching to prevent them from affecting file ranking.
        This allows intent detection to work while preventing generic words like
        "fix", "debug", "implement" from matching unrelated files.
        """
        # Try to load from intent patterns file
        intent_patterns_file = self.data_dir.parent / "pattterns" / "intent_patterns.json"
        intent_action_words = set()
        
        try:
            if intent_patterns_file.exists():
                with open(intent_patterns_file, encoding="utf-8") as f:
                    patterns = json.load(f)
                    
                # Extract keywords from each intent type
                for intent_type, intent_data in patterns.items():
                    if "keywords" in intent_data:
                        # Add common action words that shouldn't affect file ranking
                        for keyword in intent_data["keywords"]:
                            # Filter only generic action words, not domain-specific terms
                            if keyword.lower() in {
                                # Debug intent actions
                                "debug", "fix", "solve", "resolve", "troubleshoot", 
                                "investigate", "diagnose", "bug", "error", "issue", 
                                "problem", "broken", "crash", "exception",
                                
                                # Implement intent actions  
                                "implement", "add", "create", "build", "develop",
                                "make", "write", "code", "feature", "new",
                                "setup", "initialize", "design", "integrate", 
                                "enable", "deploy",
                                
                                # Refactor intent actions
                                "refactor", "restructure", "improve", "optimize",
                                "clean", "cleanup", "modernize", "simplify",
                                "decouple", "extract", "rewrite", "redesign",
                                "migrate", "update", "upgrade",
                                
                                # Test intent actions
                                "test", "testing", "coverage", "spec", "suite",
                                "unit", "integration", "mock", "stub",
                                
                                # Understand intent actions
                                "understand", "explain", "how", "what", "why",
                                "where", "when", "show", "describe", "works",
                                "analyze", "explore", "review", "documentation",
                                
                                # Document intent actions
                                "document", "docs", "readme", "comment", "annotate",
                                "describe", "clarify", "detail", "specify"
                            }:
                                intent_action_words.add(keyword.lower())
                                
                self.logger.debug(f"Loaded {len(intent_action_words)} intent action words")
                
        except Exception as e:
            self.logger.warning(f"Failed to load intent patterns: {e}")
            
        # Always create the set with at least a fallback list
        if not intent_action_words:
            # Fallback to most common action words
            intent_action_words = {
                "fix", "debug", "implement", "add", "create", "build",
                "refactor", "test", "understand", "explain", "document",
                "issue", "bug", "error", "problem", "feature"
            }
            
        self._sets["intent_actions"] = StopwordSet(
            name="intent_actions",
            words=intent_action_words,
            description="Intent action words to filter from keyword matching",
            source_file=intent_patterns_file if intent_patterns_file.exists() else None
        )

    def _load_set_from_file(self, file_path: Path, name: str, description: str) -> StopwordSet:
        """Load stopword set from file.

        Args:
            file_path: Path to stopword file
            name: Name for this set
            description: Description of set purpose

        Returns:
            Loaded StopwordSet
        """
        words = set()

        try:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith("#"):
                        words.add(line.lower())

            self.logger.debug(f"Loaded {len(words)} stopwords from {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to load stopwords from {file_path}: {e}")

        return StopwordSet(name=name, words=words, description=description, source_file=file_path)

    def get_set(self, name: str) -> Optional[StopwordSet]:
        """Get a stopword set by name.

        Args:
            name: Name of stopword set ('code', 'prompt', 'intent_actions', etc.)

        Returns:
            StopwordSet or None if not found
        """
        return self._sets.get(name)
    
    def get_combined_set(self, *names: str, filter_intent_actions: bool = False) -> StopwordSet:
        """Get a combined stopword set from multiple named sets.
        
        Args:
            *names: Names of sets to combine
            filter_intent_actions: Whether to include intent action words
            
        Returns:
            Combined StopwordSet
        """
        combined_words = set()
        combined_names = []
        
        for name in names:
            if name in self._sets:
                combined_words.update(self._sets[name].words)
                combined_names.append(name)
                
        # Optionally add intent action words for filtering
        if filter_intent_actions and "intent_actions" in self._sets:
            combined_words.update(self._sets["intent_actions"].words)
            combined_names.append("intent_actions")
            
        return StopwordSet(
            name="+".join(combined_names),
            words=combined_words,
            description=f"Combined stopword set: {', '.join(combined_names)}"
        )

    def add_custom_set(self, name: str, words: Set[str], description: str = "") -> StopwordSet:
        """Add a custom stopword set.

        Args:
            name: Name for the set
            words: Set of stopword strings
            description: What this set is for

        Returns:
            Created StopwordSet
        """
        stopword_set = StopwordSet(
            name=name, words={w.lower() for w in words}, description=description
        )
        self._sets[name] = stopword_set
        return stopword_set

    def combine_sets(self, sets: List[str], name: str = "combined") -> StopwordSet:
        """Combine multiple stopword sets.

        Args:
            sets: Names of sets to combine
            name: Name for combined set

        Returns:
            Combined StopwordSet
        """
        combined_words = set()

        for set_name in sets:
            if set_name in self._sets:
                combined_words |= self._sets[set_name].words

        return StopwordSet(
            name=name, words=combined_words, description=f"Combined from: {', '.join(sets)}"
        )
