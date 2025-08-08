"""Token optimization for context generation.

The optimizer ensures we make the best use of available tokens by
intelligently selecting what to include and what to summarize.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from tenets.config import TenetsConfig
from tenets.models.analysis import FileAnalysis
from tenets.utils.tokens import count_tokens
from tenets.models.llm import get_model_limits
from tenets.utils.logger import get_logger


@dataclass
class TokenBudget:
    """Manages token allocation for context building.
    
    Attributes:
        total_limit: Total token budget available.
        model: Optional target model name.
        prompt_tokens: Tokens consumed by the prompt/instructions.
        response_reserve: Reserved tokens for model output.
        structure_tokens: Reserved tokens for headers/formatting.
        git_tokens: Reserved tokens for git metadata.
        tenet_tokens: Reserved tokens for tenet injection.
    """
    
    total_limit: int
    model: Optional[str] = None
    
    # Reserved allocations
    prompt_tokens: int = 0
    response_reserve: int = 4000  # Reserve for model response
    structure_tokens: int = 1000  # Headers, formatting, etc.
    git_tokens: int = 0
    tenet_tokens: int = 0
    
    @property
    def available_for_files(self) -> int:
        """Calculate tokens available for file content."""
        return max(
            0,
            self.total_limit - self.prompt_tokens - self.response_reserve - 
            self.structure_tokens - self.git_tokens - self.tenet_tokens
        )
    
    @property
    def utilization(self) -> float:
        """Calculate budget utilization percentage."""
        used = self.prompt_tokens + self.structure_tokens + self.git_tokens + self.tenet_tokens
        return used / self.total_limit if self.total_limit > 0 else 0


class TokenOptimizer:
    """Optimizes token usage for maximum context value."""
    
    def __init__(self, config: TenetsConfig):
        """Initialize the optimizer.
        
        Args:
            config: Tenets configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
    def create_budget(
        self,
        model: Optional[str],
        max_tokens: Optional[int],
        prompt_tokens: int,
        has_git_context: bool = False,
        has_tenets: bool = False
    ) -> TokenBudget:
        """Create a token budget for context generation.
        
        Args:
            model: Target model name.
            max_tokens: Optional hard cap on total tokens; overrides model default.
            prompt_tokens: Tokens used by the prompt/instructions.
            has_git_context: Whether git context will be included.
            has_tenets: Whether tenets will be injected.
            
        Returns:
            TokenBudget: Configured budget with reserves.
        """
        # Determine total limit
        if max_tokens:
            total_limit = max_tokens
        elif model:
            limits = get_model_limits(model)
            total_limit = limits.max_context
        else:
            total_limit = self.config.max_tokens
            
        # Create budget
        budget = TokenBudget(
            total_limit=total_limit,
            model=model,
            prompt_tokens=prompt_tokens
        )
        
        # Adjust reserves based on model
        if model and 'gpt-4' in model.lower():
            budget.response_reserve = 4000
        elif model and 'claude' in model.lower():
            budget.response_reserve = 4000
        else:
            budget.response_reserve = 2000
            
        # Reserve for git context
        if has_git_context:
            budget.git_tokens = 500  # Rough estimate
            
        # Reserve for tenets
        if has_tenets:
            budget.tenet_tokens = 300  # Rough estimate
            
        self.logger.debug(
            f"Created token budget: {budget.available_for_files:,} available for files "
            f"(total: {total_limit:,}, reserved: {total_limit - budget.available_for_files:,})"
        )
        
        return budget
    
    def optimize_file_selection(
        self,
        files: List[FileAnalysis],
        budget: TokenBudget,
        strategy: str = 'balanced'
    ) -> List[Tuple[FileAnalysis, str]]:
        """Optimize file selection within budget.
        
        Uses different strategies to select which files to include
        and whether to summarize them.
        
        Args:
            files: Ranked files to consider
            budget: Token budget to work within
            strategy: Selection strategy (greedy, balanced, diverse)
            
        Returns:
            List of (file, action) tuples where action is 'full' or 'summary'
        """
        if strategy == 'greedy':
            return self._greedy_selection(files, budget)
        elif strategy == 'balanced':
            return self._balanced_selection(files, budget)
        elif strategy == 'diverse':
            return self._diverse_selection(files, budget)
        else:
            return self._balanced_selection(files, budget)
    
    def _greedy_selection(
        self,
        files: List[FileAnalysis],
        budget: TokenBudget
    ) -> List[Tuple[FileAnalysis, str]]:
        """Greedy selection - take highest relevance files that fit."""
        selected = []
        used_tokens = 0
        available = budget.available_for_files
        
        for file in files:
            file_tokens = count_tokens(file.content, budget.model)
            
            # Try full inclusion
            if used_tokens + file_tokens <= available:
                selected.append((file, 'full'))
                used_tokens += file_tokens
            else:
                # Try summarized version
                summary_tokens = min(file_tokens // 4, available - used_tokens)
                if summary_tokens > 100:  # Worth summarizing
                    selected.append((file, 'summary'))
                    used_tokens += summary_tokens
                    
            # Stop if we're close to the limit
            if used_tokens >= available * 0.95:
                break
                
        return selected
    
    def _balanced_selection(
        self,
        files: List[FileAnalysis],
        budget: TokenBudget
    ) -> List[Tuple[FileAnalysis, str]]:
        """Balanced selection - mix of full and summarized files."""
        selected = []
        used_tokens = 0
        available = budget.available_for_files
        
        # Phase 1: Include top files in full
        full_count = min(10, len(files) // 3)  # At most 10 full files
        
        for i, file in enumerate(files[:full_count]):
            file_tokens = count_tokens(file.content, budget.model)
            
            if used_tokens + file_tokens <= available * 0.6:  # Use at most 60% for full files
                selected.append((file, 'full'))
                used_tokens += file_tokens
            else:
                # Switch to summaries
                break
                
        # Phase 2: Add more files as summaries
        remaining_files = files[len(selected):]
        
        for file in remaining_files:
            if used_tokens >= available * 0.9:
                break
                
            # Estimate summary size
            file_tokens = count_tokens(file.content, budget.model)
            summary_tokens = min(
                file_tokens // 4,
                max(200, (available - used_tokens) // 10)  # At least 200 tokens per summary
            )
            
            if used_tokens + summary_tokens <= available:
                selected.append((file, 'summary'))
                used_tokens += summary_tokens
                
        return selected
    
    def _diverse_selection(
        self,
        files: List[FileAnalysis],
        budget: TokenBudget
    ) -> List[Tuple[FileAnalysis, str]]:
        """Diverse selection - ensure variety of file types and directories."""
        # Group files by directory and extension
        by_directory = {}
        by_extension = {}
        
        for file in files:
            dir_path = str(Path(file.path).parent)
            ext = Path(file.path).suffix
            
            by_directory.setdefault(dir_path, []).append(file)
            by_extension.setdefault(ext, []).append(file)
            
        selected = []
        used_tokens = 0
        available = budget.available_for_files
        
        # Take top file from each directory
        for dir_files in by_directory.values():
            if used_tokens >= available * 0.9:
                break
                
            top_file = max(dir_files, key=lambda f: f.relevance_score)
            file_tokens = count_tokens(top_file.content, budget.model)
            
            if used_tokens + file_tokens <= available * 0.7:
                selected.append((top_file, 'full'))
                used_tokens += file_tokens
            else:
                summary_tokens = min(file_tokens // 4, 300)
                if used_tokens + summary_tokens <= available:
                    selected.append((top_file, 'summary'))
                    used_tokens += summary_tokens
                    
        # Fill remaining space with high-relevance files
        selected_paths = {s[0].path for s in selected}
        remaining = [f for f in files if f.path not in selected_paths]
        
        for file in remaining:
            if used_tokens >= available * 0.95:
                break
                
            summary_tokens = 200  # Fixed small summaries
            if used_tokens + summary_tokens <= available:
                selected.append((file, 'summary'))
                used_tokens += summary_tokens
                
        return selected
    
    def estimate_tokens_for_git(
        self,
        git_context: Optional[Dict[str, Any]]
    ) -> int:
        """Estimate tokens needed for git context."""
        if not git_context:
            return 0
            
        tokens = 100  # Base overhead
        
        if 'recent_commits' in git_context:
            # ~50 tokens per commit
            tokens += len(git_context['recent_commits']) * 50
            
        if 'contributors' in git_context:
            # ~20 tokens per contributor  
            tokens += len(git_context['contributors']) * 20
            
        if 'recent_changes' in git_context:
            # ~30 tokens per file change entry
            tokens += len(git_context.get('recent_changes', [])) * 30
            
        return tokens
    
    def estimate_tokens_for_tenets(
        self,
        tenet_count: int,
        with_reinforcement: bool = False
    ) -> int:
        """Estimate tokens needed for tenet injection."""
        # ~30 tokens per tenet with formatting
        tokens = tenet_count * 30
        
        # Reinforcement section adds ~100 tokens
        if with_reinforcement and tenet_count > 3:
            tokens += 100
            
        return tokens