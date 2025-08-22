"""Stopword management for different contexts.

This module manages multiple stopword sets for different purposes:
- Minimal set for code search (preserve accuracy)
- Aggressive set for prompt parsing (extract intent)
- Custom sets for specific domains
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

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
        """Load default stopword sets from data files."""
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
            name: Name of stopword set ('code', 'prompt', etc.)

        Returns:
            StopwordSet or None if not found
        """
        return self._sets.get(name)

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
