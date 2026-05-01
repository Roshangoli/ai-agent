"""
LLM Observability Module

Provides comprehensive observability for LLM-powered multi-agent systems.

Layers implemented:
- Layer 1: Infrastructure (correlation_id, latency, cost, errors)
- Layer 2: Model Behavior (prompt versions, token tracking)
- Layer 3: Output Quality (SQL/narrative scoring)
- Layer 5: Agent Specific (validation loops, handoffs)
- Layer 7: Business Impact (time saved, ROI)
- Layer 8: Drift & Regression (quality trends, CI/CD gates)
"""

from .tracer import Tracer, get_tracer
from .prompt_version import PromptVersion, PromptRegistry, get_prompt_registry
from .quality_scorer import (
    SQLQualityScorer,
    NarrativeQualityScorer,
    QualityRegistry,
    get_quality_registry
)

__all__ = [
    'Tracer', 'get_tracer',
    'PromptVersion', 'PromptRegistry', 'get_prompt_registry',
    'SQLQualityScorer', 'NarrativeQualityScorer', 'QualityRegistry', 'get_quality_registry'
]
