"""
ML Configuration for Ranking Strategies

This module provides configuration management for ML features in ranking,
particularly for controlling token limits and performance trade-offs.

Key features:
    - Configurable token limits for embeddings
    - Performance vs accuracy trade-offs
    - Adaptive token sizing based on available resources
    - Monitoring and warnings for performance issues

Author: Tenets Team
License: MIT
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import psutil
import logging
from pathlib import Path

from tenets.utils.logger import get_logger


@dataclass
class MLConfig:
    """
    Configuration for ML features in ranking.
    
    This class manages the trade-offs between performance and accuracy
    for ML-based ranking features, particularly token limits for embeddings.
    
    Attributes:
        enabled: Whether ML features are enabled
        token_limit: Maximum tokens per document for embedding (default: 1000)
        min_token_limit: Minimum acceptable token limit (default: 256)
        max_token_limit: Maximum token limit (default: 4096)
        batch_size: Batch size for embedding computation (default: 32)
        model_name: Name of the embedding model (default: all-MiniLM-L6-v2)
        cache_embeddings: Whether to cache computed embeddings (default: True)
        adaptive_sizing: Automatically adjust token limit based on resources
        performance_mode: "speed" | "balanced" | "quality"
    
    Example:
        >>> config = MLConfig(token_limit=512, performance_mode="speed")
        >>> config.validate()
        >>> actual_limit = config.get_adaptive_token_limit(num_files=100)
    """
    
    enabled: bool = True
    token_limit: int = 1000
    min_token_limit: int = 256
    max_token_limit: int = 4096
    batch_size: int = 32
    model_name: str = "all-MiniLM-L6-v2"
    cache_embeddings: bool = True
    adaptive_sizing: bool = True
    performance_mode: str = "balanced"  # "speed", "balanced", "quality"
    
    # Performance mode presets
    PERFORMANCE_PRESETS = {
        "speed": {
            "token_limit": 256,
            "batch_size": 64,
            "cache_embeddings": True,
            "model": "all-MiniLM-L6-v2"
        },
        "balanced": {
            "token_limit": 1000,
            "batch_size": 32,
            "cache_embeddings": True,
            "model": "all-MiniLM-L6-v2"
        },
        "quality": {
            "token_limit": 2048,
            "batch_size": 16,
            "cache_embeddings": True,
            "model": "all-mpnet-base-v2"
        }
    }
    
    def __post_init__(self):
        """Initialize and apply performance presets if specified."""
        self.logger = get_logger(__name__)
        
        # Apply performance preset
        if self.performance_mode in self.PERFORMANCE_PRESETS:
            preset = self.PERFORMANCE_PRESETS[self.performance_mode]
            self.token_limit = preset["token_limit"]
            self.batch_size = preset["batch_size"]
            self.cache_embeddings = preset["cache_embeddings"]
            if self.model_name == "all-MiniLM-L6-v2":
                self.model_name = preset["model"]
                
        self.validate()
        
    def validate(self):
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.token_limit < self.min_token_limit:
            raise ValueError(
                f"Token limit {self.token_limit} below minimum {self.min_token_limit}"
            )
            
        if self.token_limit > self.max_token_limit:
            raise ValueError(
                f"Token limit {self.token_limit} exceeds maximum {self.max_token_limit}"
            )
            
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
            
        if self.performance_mode not in ["speed", "balanced", "quality"]:
            raise ValueError(
                f"Invalid performance mode: {self.performance_mode}. "
                f"Choose from: speed, balanced, quality"
            )
            
    def get_adaptive_token_limit(
        self,
        num_files: int,
        available_memory_mb: Optional[float] = None
    ) -> int:
        """
        Calculate adaptive token limit based on resources.
        
        This method dynamically adjusts the token limit based on:
        - Number of files to process
        - Available system memory
        - Performance mode setting
        
        Args:
            num_files: Number of files to process
            available_memory_mb: Available memory in MB (auto-detected if None)
            
        Returns:
            Adjusted token limit
            
        Example:
            >>> config = MLConfig(adaptive_sizing=True)
            >>> limit = config.get_adaptive_token_limit(num_files=1000)
            >>> # Returns lower limit for many files to avoid memory issues
        """
        if not self.adaptive_sizing:
            return self.token_limit
            
        # Get available memory if not provided
        if available_memory_mb is None:
            try:
                mem = psutil.virtual_memory()
                available_memory_mb = mem.available / (1024 * 1024)
            except:
                available_memory_mb = 1024
                
        memory_per_file_mb = (self.token_limit * 0.002) + 0.0015
        total_memory_needed = num_files * memory_per_file_mb
        
        # Adjust token limit if memory constrained
        if total_memory_needed > available_memory_mb * 0.5:
            scale_factor = (available_memory_mb * 0.5) / total_memory_needed
            adjusted_limit = int(self.token_limit * scale_factor)
            
            adjusted_limit = max(self.min_token_limit, 
                               min(adjusted_limit, self.max_token_limit))
            
            if adjusted_limit < self.token_limit:
                self.logger.info(
                    f"Adaptive sizing: Reduced token limit from {self.token_limit} "
                    f"to {adjusted_limit} for {num_files} files"
                )
                
            return adjusted_limit
            
        if num_files > 1000 and self.performance_mode != "quality":
            if num_files > 5000:
                return self.min_token_limit
            elif num_files > 2000:
                return min(512, self.token_limit)
            else:
                return min(768, self.token_limit)
                
        return self.token_limit
        
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration for sentence-transformers.
        
        Returns:
            Dictionary with model configuration parameters
        """
        return {
            "model_name": self.model_name,
            "device": "cuda" if self._cuda_available() else "cpu",
            "normalize_embeddings": True,
            "batch_size": self.batch_size,
            "show_progress_bar": False
        }
        
    def _cuda_available(self) -> bool:
        """Check if CUDA is available for GPU acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def estimate_processing_time(
        self,
        num_files: int,
        avg_file_size: int = 5000
    ) -> Dict[str, float]:
        """
        Estimate processing time for ML operations.
        
        Args:
            num_files: Number of files to process
            avg_file_size: Average file size in characters
            
        Returns:
            Dictionary with time estimates in milliseconds
        """
        encoding_time_per_token = 0.01
        if self._cuda_available():
            encoding_time_per_token = 0.002
            
        tokens_per_file = min(self.token_limit, avg_file_size // 4)
        time_per_file = tokens_per_file * encoding_time_per_token
        
        batch_overhead = (num_files / self.batch_size) * 10
        
        total_time = (num_files * time_per_file) + batch_overhead
        
        return {
            "total_ms": total_time,
            "per_file_ms": time_per_file,
            "batch_overhead_ms": batch_overhead,
            "performance_mode": self.performance_mode,
            "token_limit": self.get_adaptive_token_limit(num_files)
        }
        
    def log_configuration(self):
        """Log current ML configuration for debugging."""
        self.logger.info("ML Configuration:")
        self.logger.info(f"  Enabled: {self.enabled}")
        self.logger.info(f"  Performance mode: {self.performance_mode}")
        self.logger.info(f"  Token limit: {self.token_limit}")
        self.logger.info(f"  Batch size: {self.batch_size}")
        self.logger.info(f"  Model: {self.model_name}")
        self.logger.info(f"  Adaptive sizing: {self.adaptive_sizing}")
        self.logger.info(f"  Cache embeddings: {self.cache_embeddings}")
        self.logger.info(f"  CUDA available: {self._cuda_available()}")
        
    @classmethod
    def from_config_file(cls, config_path: Path) -> "MLConfig":
        """
        Load ML configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            MLConfig instance
            
        Example config.yml:
            ml:
              enabled: true
              performance_mode: balanced
              token_limit: 1000
              adaptive_sizing: true
        """
        import yaml
        
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            
        ml_config = config_data.get("ml", {})
        
        return cls(
            enabled=ml_config.get("enabled", True),
            token_limit=ml_config.get("token_limit", 1000),
            performance_mode=ml_config.get("performance_mode", "balanced"),
            batch_size=ml_config.get("batch_size", 32),
            model_name=ml_config.get("model_name", "all-MiniLM-L6-v2"),
            cache_embeddings=ml_config.get("cache_embeddings", True),
            adaptive_sizing=ml_config.get("adaptive_sizing", True)
        )


def get_optimal_ml_config(
    num_files: int,
    performance_priority: str = "balanced"
) -> MLConfig:
    """
    Get optimal ML configuration for a given scenario.
    
    This function analyzes the workload and system resources to
    recommend the best ML configuration.
    
    Args:
        num_files: Number of files to process
        performance_priority: "speed" | "balanced" | "quality"
        
    Returns:
        Optimized MLConfig instance
        
    Example:
        >>> config = get_optimal_ml_config(num_files=500, performance_priority="speed")
        >>> # Returns config with token_limit=256 for speed
    """
    config = MLConfig(performance_mode=performance_priority)
    
    if num_files > 5000:
        config.performance_mode = "speed"
        config.token_limit = 256
        config.batch_size = 64
        logging.info(f"Large codebase ({num_files} files): Using speed mode")
        
    elif num_files < 100 and performance_priority == "quality":
        config.token_limit = 2048
        config.batch_size = 8
        logging.info(f"Small codebase ({num_files} files): Using quality mode")
        
    config.log_configuration()
    return config