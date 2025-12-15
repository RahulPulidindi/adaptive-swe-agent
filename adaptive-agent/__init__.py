"""
Adaptive Inference-Time Compute Scaling for AI Software Engineering Agents
"""

__version__ = "0.1.0"
__author__ = "Rahul Pulidindi, Aaryan Misal"
__email__ = "rp3254@columbia.edu"

from .base_agent import BaseAgent
from .adaptive_agent import AdaptiveAgent
from .fixed_agent import FixedAgent

__all__ = [
    "BaseAgent",
    "AdaptiveAgent", 
    "FixedAgent",
]