"""
Tracer - Layer 1 & 2 & 3: Infrastructure + Model Behavior + Output Quality

Layer 1 - Infrastructure:
- correlation_id (uuid4) for end-to-end request tracking
- Latency (p50/p95/p99) per agent
- Cost tracking (prompt tokens × price + completion tokens × price)
- Error rate and error messages
- Structured logs with consistent schema
- Mode tracking (Query vs Data Science)

Layer 2 - Model Behavior:
- Prompt version tracking (SHA-256 hash)
- Token efficiency metrics (tokens/query)
- Prompt drift detection

Layer 3 - Output Quality:
- SQL quality scoring (syntax, schema, best practices, optimization)
- Narrative quality scoring (relevance, clarity, hallucination detection)
- Quality trend tracking
"""

import uuid
import time
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class Trace:
    """Represents a single traced operation (agent call, tool execution, etc.)"""

    def __init__(
        self,
        correlation_id: str,
        operation_name: str,
        agent_name: Optional[str] = None,
        model_name: Optional[str] = None,
        mode: Optional[str] = None,
        prompt_version_hash: Optional[str] = None
    ):
        self.correlation_id = correlation_id
        self.operation_name = operation_name
        self.agent_name = agent_name
        self.model_name = model_name
        self.mode = mode
        self.prompt_version_hash = prompt_version_hash  # Layer 2: Prompt version tracking

        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.latency_ms: Optional[int] = None

        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.cost_usd: float = 0.0

        self.success: bool = True
        self.error_message: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def end(self, success: bool = True, error_message: Optional[str] = None):
        """Mark the trace as complete"""
        self.end_time = time.time()
        self.latency_ms = int((self.end_time - self.start_time) * 1000)
        self.success = success
        self.error_message = error_message

    def set_tokens(self, prompt_tokens: int, completion_tokens: int):
        """Set token counts and calculate cost"""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens

        # GPT-4o pricing (as of 2024/2025)
        # Prompt: $0.000005 per token
        # Completion: $0.000015 per token
        prompt_cost = prompt_tokens * 0.000005
        completion_cost = completion_tokens * 0.000015
        self.cost_usd = prompt_cost + completion_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for logging/storage"""
        return {
            "correlation_id": self.correlation_id,
            "operation_name": self.operation_name,
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "mode": self.mode,
            "prompt_version_hash": self.prompt_version_hash,  # Layer 2
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class Tracer:
    """
    Main tracing class for LLM observability.

    Tracks all operations with correlation IDs, captures latency, cost, and errors.
    Provides structured logging and metrics collection.
    """

    def __init__(self):
        self.current_correlation_id: Optional[str] = None
        self.traces: List[Trace] = []
        self.request_traces: Dict[str, List[Trace]] = defaultdict(list)

        # Metrics aggregation
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0,
            "total_cost_usd": 0.0,
            "latencies": [],  # For percentile calculations
            "costs_per_request": []
        }

    def start_request(self, mode: str = "query") -> str:
        """
        Start a new request trace with a unique correlation ID.

        Args:
            mode: "query" or "data_science"

        Returns:
            correlation_id (uuid4)
        """
        correlation_id = str(uuid.uuid4())
        self.current_correlation_id = correlation_id
        self.metrics["total_requests"] += 1

        logger.info(f"🔍 Starting request trace: {correlation_id} (mode: {mode})")

        return correlation_id

    @contextmanager
    def trace(
        self,
        operation_name: str,
        agent_name: Optional[str] = None,
        model_name: Optional[str] = "gpt-4o",
        mode: Optional[str] = None,
        prompt_version_hash: Optional[str] = None
    ):
        """
        Context manager for tracing an operation.

        Usage:
            with tracer.trace("sql_generation", agent_name="SQL_Generator", prompt_version_hash="abc123"):
                result = sql_agent.generate_reply(question)
                tracer.set_tokens(450, 120)  # Set after API call

        Args:
            operation_name: Name of the operation (e.g., "sql_generation")
            agent_name: Name of the agent (e.g., "SQL_Generator")
            model_name: LLM model name (default: "gpt-4o")
            mode: "query" or "data_science"
            prompt_version_hash: SHA-256 hash of prompt (Layer 2)
        """
        # Use current correlation_id or generate one
        correlation_id = self.current_correlation_id or self.start_request(mode or "query")

        # Create trace
        trace = Trace(
            correlation_id=correlation_id,
            operation_name=operation_name,
            agent_name=agent_name,
            model_name=model_name,
            mode=mode,
            prompt_version_hash=prompt_version_hash
        )

        # Store reference for token setting
        self._current_trace = trace

        try:
            logger.info(
                f"▶️  [{correlation_id[:8]}] Starting {operation_name} "
                f"(agent: {agent_name or 'N/A'})"
            )
            yield trace

            # Mark as successful
            trace.end(success=True)

            logger.info(
                f"✅ [{correlation_id[:8]}] Completed {operation_name} "
                f"in {trace.latency_ms}ms "
                f"(cost: ${trace.cost_usd:.6f}, tokens: {trace.total_tokens})"
            )

        except Exception as e:
            # Mark as failed
            trace.end(success=False, error_message=str(e))

            logger.error(
                f"❌ [{correlation_id[:8]}] Failed {operation_name} "
                f"after {trace.latency_ms}ms: {str(e)}"
            )

            self.metrics["failed_requests"] += 1
            raise

        finally:
            # Store trace
            self.traces.append(trace)
            self.request_traces[correlation_id].append(trace)

            # Update metrics
            if trace.success:
                self.metrics["successful_requests"] += 1

            if trace.latency_ms:
                self.metrics["total_latency_ms"] += trace.latency_ms
                self.metrics["latencies"].append(trace.latency_ms)

            self.metrics["total_cost_usd"] += trace.cost_usd

            # Log structured trace
            logger.debug(f"📊 Trace: {json.dumps(trace.to_dict(), indent=2)}")

    def set_tokens(self, prompt_tokens: int, completion_tokens: int):
        """
        Set token counts for the current trace.
        Call this after an LLM API call within a trace context.

        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        if hasattr(self, '_current_trace'):
            self._current_trace.set_tokens(prompt_tokens, completion_tokens)
        else:
            logger.warning("set_tokens() called outside of trace context")

    def end_request(self):
        """
        End the current request and calculate request-level metrics.
        """
        if not self.current_correlation_id:
            return

        request_traces = self.request_traces[self.current_correlation_id]

        # Calculate request-level metrics
        total_latency = sum(t.latency_ms or 0 for t in request_traces)
        total_cost = sum(t.cost_usd for t in request_traces)
        all_successful = all(t.success for t in request_traces)

        self.metrics["costs_per_request"].append(total_cost)

        logger.info(
            f"🏁 Request {self.current_correlation_id[:8]} completed: "
            f"{len(request_traces)} operations, "
            f"{total_latency}ms total, "
            f"${total_cost:.6f} cost, "
            f"{'✅ success' if all_successful else '❌ failed'}"
        )

        self.current_correlation_id = None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary with metrics including:
            - total_requests, successful_requests, failed_requests
            - avg_latency_ms, p50/p95/p99 latency
            - total_cost_usd, avg_cost_per_request
            - error_rate
        """
        total_requests = self.metrics["total_requests"]

        if total_requests == 0:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "error_rate": 0.0,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "total_cost_usd": 0.0,
                "avg_cost_per_request": 0.0
            }

        # Calculate percentiles
        latencies = sorted(self.metrics["latencies"])
        costs = self.metrics["costs_per_request"]

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < len(data):
                return data[f] * (1 - c) + data[f + 1] * c
            else:
                return data[f]

        return {
            "total_requests": total_requests,
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "error_rate": self.metrics["failed_requests"] / total_requests,

            "avg_latency_ms": int(self.metrics["total_latency_ms"] / len(latencies)) if latencies else 0,
            "p50_latency_ms": int(percentile(latencies, 0.50)) if latencies else 0,
            "p95_latency_ms": int(percentile(latencies, 0.95)) if latencies else 0,
            "p99_latency_ms": int(percentile(latencies, 0.99)) if latencies else 0,

            "total_cost_usd": round(self.metrics["total_cost_usd"], 6),
            "avg_cost_per_request": round(sum(costs) / len(costs), 6) if costs else 0.0
        }

    def get_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent traces.

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of trace dictionaries
        """
        return [t.to_dict() for t in self.traces[-limit:]]

    def get_request_trace(self, correlation_id: str) -> List[Dict[str, Any]]:
        """
        Get all traces for a specific request.

        Args:
            correlation_id: The correlation ID to look up

        Returns:
            List of trace dictionaries for that request
        """
        traces = self.request_traces.get(correlation_id, [])
        return [t.to_dict() for t in traces]


# Global tracer instance
_global_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """
    Get the global tracer instance (singleton).

    Returns:
        Global Tracer instance
    """
    global _global_tracer

    if _global_tracer is None:
        _global_tracer = Tracer()
        logger.info("🔍 Tracer initialized")

    return _global_tracer
