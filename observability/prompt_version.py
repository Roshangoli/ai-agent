"""
Prompt Version Tracking - Layer 2: Model Behavior

Captures:
- Prompt version hashes (SHA-256)
- Prompt change detection (drift alerts)
- Token efficiency metrics (tokens per query type)
- Prompt-to-cost correlation
"""

import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict


class PromptVersion:
    """
    Tracks prompt versions and detects changes.

    Uses SHA-256 hashing to create unique version IDs.
    Stores prompt metadata for drift detection.
    """

    def __init__(self, agent_name: str, prompt_text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a prompt version.

        Args:
            agent_name: Name of the agent using this prompt
            prompt_text: The actual prompt text (system message)
            metadata: Optional metadata (e.g., creation date, author, purpose)
        """
        self.agent_name = agent_name
        self.prompt_text = prompt_text
        self.metadata = metadata or {}

        # Generate version hash (SHA-256 of prompt text)
        self.version_hash = self._generate_hash(prompt_text)
        self.short_hash = self.version_hash[:8]  # First 8 chars for display

        # Timestamp
        self.created_at = datetime.utcnow().isoformat()

    def _generate_hash(self, text: str) -> str:
        """Generate SHA-256 hash of prompt text"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging"""
        return {
            "agent_name": self.agent_name,
            "version_hash": self.version_hash,
            "short_hash": self.short_hash,
            "created_at": self.created_at,
            "prompt_length": len(self.prompt_text),
            "metadata": self.metadata
        }


class PromptRegistry:
    """
    Registry for tracking all prompt versions across agents.

    Detects when prompts change (version drift).
    Calculates token efficiency per prompt version.
    """

    def __init__(self):
        # Store prompt versions by agent_name
        self.versions: Dict[str, PromptVersion] = {}

        # Track version history (all versions seen)
        self.version_history: Dict[str, List[PromptVersion]] = defaultdict(list)

        # Track metrics per version
        self.version_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "avg_tokens_per_request": 0.0,
            "avg_cost_per_request": 0.0
        })

    def register_prompt(
        self,
        agent_name: str,
        prompt_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PromptVersion:
        """
        Register a prompt version.

        Detects if this is a new version (prompt changed).

        Args:
            agent_name: Name of the agent
            prompt_text: The prompt text
            metadata: Optional metadata

        Returns:
            PromptVersion object
        """
        new_version = PromptVersion(agent_name, prompt_text, metadata)

        # Check if this agent already has a registered prompt
        if agent_name in self.versions:
            old_version = self.versions[agent_name]

            # DRIFT DETECTION: Prompt changed!
            if old_version.version_hash != new_version.version_hash:
                print(f"⚠️  PROMPT DRIFT DETECTED for {agent_name}")
                print(f"   Old version: {old_version.short_hash}")
                print(f"   New version: {new_version.short_hash}")
                print(f"   Prompt length change: {old_version.prompt_text.__len__()} → {new_version.prompt_text.__len__()} chars")

        # Store current version
        self.versions[agent_name] = new_version

        # Add to history
        self.version_history[agent_name].append(new_version)

        return new_version

    def get_version(self, agent_name: str) -> Optional[PromptVersion]:
        """Get current prompt version for an agent"""
        return self.versions.get(agent_name)

    def record_usage(
        self,
        agent_name: str,
        tokens: int,
        cost_usd: float
    ):
        """
        Record usage metrics for a prompt version.

        Args:
            agent_name: Name of the agent
            tokens: Total tokens used
            cost_usd: Cost in USD
        """
        version = self.get_version(agent_name)
        if not version:
            return

        version_hash = version.version_hash
        metrics = self.version_metrics[version_hash]

        # Update metrics
        metrics["total_requests"] += 1
        metrics["total_tokens"] += tokens
        metrics["total_cost_usd"] += cost_usd

        # Calculate averages
        metrics["avg_tokens_per_request"] = metrics["total_tokens"] / metrics["total_requests"]
        metrics["avg_cost_per_request"] = metrics["total_cost_usd"] / metrics["total_requests"]

    def get_version_metrics(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for current version of an agent's prompt"""
        version = self.get_version(agent_name)
        if not version:
            return None

        return self.version_metrics[version.version_hash]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all prompt versions"""
        result = {}

        for agent_name, version in self.versions.items():
            metrics = self.version_metrics[version.version_hash].copy()
            metrics["agent_name"] = agent_name
            metrics["version_hash"] = version.short_hash
            metrics["prompt_length"] = len(version.prompt_text)
            result[agent_name] = metrics

        return result

    def detect_inefficiencies(self) -> List[Dict[str, Any]]:
        """
        Detect token inefficiencies across agents.

        Returns:
            List of inefficiency warnings
        """
        warnings = []

        for agent_name, version in self.versions.items():
            metrics = self.version_metrics[version.version_hash]

            if metrics["total_requests"] == 0:
                continue

            avg_tokens = metrics["avg_tokens_per_request"]
            prompt_length = len(version.prompt_text)

            # Warning 1: Prompt too long (>2000 chars)
            if prompt_length > 2000:
                warnings.append({
                    "type": "long_prompt",
                    "agent": agent_name,
                    "prompt_length": prompt_length,
                    "message": f"Prompt for {agent_name} is {prompt_length} chars (consider shortening)"
                })

            # Warning 2: High token usage per request (>1000 tokens avg)
            if avg_tokens > 1000:
                warnings.append({
                    "type": "high_token_usage",
                    "agent": agent_name,
                    "avg_tokens": avg_tokens,
                    "message": f"{agent_name} uses {avg_tokens:.0f} tokens/request (optimize prompt or use gpt-4o-mini)"
                })

            # Warning 3: High cost per request (>$0.01)
            avg_cost = metrics["avg_cost_per_request"]
            if avg_cost > 0.01:
                warnings.append({
                    "type": "high_cost",
                    "agent": agent_name,
                    "avg_cost": avg_cost,
                    "message": f"{agent_name} costs ${avg_cost:.4f}/request (consider prompt optimization)"
                })

        return warnings


# Global registry instance
_global_registry: Optional[PromptRegistry] = None


def get_prompt_registry() -> PromptRegistry:
    """
    Get the global prompt registry (singleton).

    Returns:
        Global PromptRegistry instance
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = PromptRegistry()

    return _global_registry
