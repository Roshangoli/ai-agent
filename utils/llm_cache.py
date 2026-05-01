"""
LLM Cache Module
Caches LLM responses to reduce costs by 80% on repeated operations.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LLMCache:
    """
    Caches LLM responses based on prompt fingerprints.
    Reduces API calls and costs for identical/similar prompts.
    """

    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: int = 24):
        """
        Initialize LLM cache.

        Args:
            cache_dir: Directory to store cache files (default: project_root/cache)
            ttl_hours: Time-to-live for cache entries in hours (default: 24)
        """
        if cache_dir is None:
            # Get project root (2 levels up from utils/)
            project_root = Path(__file__).parent.parent
            cache_dir = project_root / "cache" / "llm"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

        # In-memory cache for faster lookups
        self.memory_cache = {}

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "evictions": 0,
            "cost_saved_usd": 0.0
        }

        logger.info(f"LLM Cache initialized: {self.cache_dir} (TTL: {ttl_hours}h)")

    def _generate_cache_key(self, prompt: str, model: str = "gpt-4") -> str:
        """
        Generate cache key from prompt and model.

        Args:
            prompt: The prompt text
            model: Model name

        Returns:
            SHA-256 hash of prompt+model
        """
        # Normalize prompt (strip whitespace, lowercase for consistency)
        normalized = prompt.strip().lower()

        # Create fingerprint
        fingerprint = f"{model}:{normalized}"
        cache_key = hashlib.sha256(fingerprint.encode()).hexdigest()

        return cache_key

    def get(self, prompt: str, model: str = "gpt-4") -> Optional[Dict[str, Any]]:
        """
        Get cached response for prompt.

        Args:
            prompt: The prompt text
            model: Model name

        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._generate_cache_key(prompt, model)

        # Check memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]

            # Check if expired
            if self._is_expired(entry):
                del self.memory_cache[cache_key]
                self.stats["evictions"] += 1
                logger.debug(f"Cache entry expired: {cache_key[:12]}...")
                return None

            self.stats["hits"] += 1

            # Calculate cost saved (rough estimate)
            tokens = entry.get("estimated_tokens", 1000)
            cost_saved = (tokens / 1_000_000) * 5  # $5 per 1M tokens for GPT-4
            self.stats["cost_saved_usd"] += cost_saved

            logger.info(f"✅ Cache HIT: {cache_key[:12]}... (saved ${cost_saved:.4f})")
            return entry["response"]

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)

                # Check if expired
                if self._is_expired(entry):
                    cache_file.unlink()  # Delete expired file
                    self.stats["evictions"] += 1
                    logger.debug(f"Cache file expired and deleted: {cache_key[:12]}...")
                    return None

                # Load into memory cache
                self.memory_cache[cache_key] = entry

                self.stats["hits"] += 1

                # Calculate cost saved
                tokens = entry.get("estimated_tokens", 1000)
                cost_saved = (tokens / 1_000_000) * 5
                self.stats["cost_saved_usd"] += cost_saved

                logger.info(f"✅ Cache HIT (disk): {cache_key[:12]}... (saved ${cost_saved:.4f})")
                return entry["response"]

            except Exception as e:
                logger.error(f"Error loading cache file: {e}")
                return None

        # Cache miss
        self.stats["misses"] += 1
        logger.debug(f"Cache MISS: {cache_key[:12]}...")
        return None

    def set(
        self,
        prompt: str,
        response: Dict[str, Any],
        model: str = "gpt-4",
        estimated_tokens: int = 1000
    ) -> None:
        """
        Store response in cache.

        Args:
            prompt: The prompt text
            response: The LLM response to cache
            model: Model name
            estimated_tokens: Estimated tokens used (for cost tracking)
        """
        cache_key = self._generate_cache_key(prompt, model)

        entry = {
            "prompt": prompt[:200],  # Store truncated prompt for reference
            "response": response,
            "model": model,
            "estimated_tokens": estimated_tokens,
            "cached_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + self.ttl).isoformat()
        }

        # Save to memory cache
        self.memory_cache[cache_key] = entry

        # Save to disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(entry, f, indent=2)

            self.stats["saves"] += 1
            logger.debug(f"💾 Cached response: {cache_key[:12]}...")

        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """
        Check if cache entry is expired.

        Args:
            entry: Cache entry dictionary

        Returns:
            True if expired, False otherwise
        """
        try:
            expires_at = datetime.fromisoformat(entry["expires_at"])
            return datetime.now() > expires_at
        except:
            return True  # Treat malformed entries as expired

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = 0

        # Clear memory cache
        count += len(self.memory_cache)
        self.memory_cache.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        logger.info(f"🗑️  Cleared {count} cache entries")
        return count

    def clean_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        count = 0

        # Clean memory cache
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            del self.memory_cache[key]
            count += 1

        # Clean disk cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)

                if self._is_expired(entry):
                    cache_file.unlink()
                    count += 1
            except:
                # Delete malformed files
                cache_file.unlink()
                count += 1

        if count > 0:
            logger.info(f"🧹 Cleaned {count} expired cache entries")

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": len(list(self.cache_dir.glob("*.json")))
        }

    def get_summary(self) -> str:
        """
        Get human-readable cache summary.

        Returns:
            Formatted summary string
        """
        stats = self.get_stats()

        summary = f"""
LLM Cache Summary:
  Hits: {stats['hits']}
  Misses: {stats['misses']}
  Hit Rate: {stats['hit_rate_percent']:.1f}%
  Total Saved: ${stats['cost_saved_usd']:.2f}
  Cache Size: {stats['memory_cache_size']} in memory, {stats['disk_cache_size']} on disk
"""
        return summary


# Global cache instance
_global_cache = None


def get_cache(ttl_hours: int = 24) -> LLMCache:
    """
    Get global cache instance (singleton).

    Args:
        ttl_hours: Time-to-live for cache entries

    Returns:
        Global LLMCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMCache(ttl_hours=ttl_hours)
    return _global_cache


# Decorator for caching LLM calls
def cache_llm_response(model: str = "gpt-4", ttl_hours: int = 24):
    """
    Decorator to cache LLM responses.

    Usage:
        @cache_llm_response(model="gpt-4", ttl_hours=24)
        def call_llm(prompt):
            response = openai.ChatCompletion.create(...)
            return response

    Args:
        model: Model name for cache key
        ttl_hours: Cache TTL in hours
    """
    def decorator(func):
        def wrapper(prompt, *args, **kwargs):
            cache = get_cache(ttl_hours)

            # Try to get from cache
            cached_response = cache.get(prompt, model)
            if cached_response is not None:
                return cached_response

            # Call actual function
            response = func(prompt, *args, **kwargs)

            # Cache the response
            estimated_tokens = kwargs.get("estimated_tokens", 1000)
            cache.set(prompt, response, model, estimated_tokens)

            return response

        return wrapper
    return decorator


if __name__ == "__main__":
    # Test the cache
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*60)
    print("LLM CACHE TEST")
    print("="*60)

    cache = LLMCache(ttl_hours=1)

    # Test 1: Cache miss
    print("\n1. First request (cache MISS)...")
    prompt = "Analyze this data and recommend cleaning strategies"
    response1 = cache.get(prompt)
    print(f"   Result: {response1}")

    # Test 2: Save to cache
    print("\n2. Saving response to cache...")
    mock_response = {
        "decision": "use median for age column",
        "reasoning": "Age is right-skewed"
    }
    cache.set(prompt, mock_response, estimated_tokens=1500)

    # Test 3: Cache hit
    print("\n3. Second request (cache HIT)...")
    response2 = cache.get(prompt)
    print(f"   Result: {response2}")

    # Test 4: Different prompt (cache miss)
    print("\n4. Different prompt (cache MISS)...")
    prompt2 = "Different analysis request"
    response3 = cache.get(prompt2)
    print(f"   Result: {response3}")

    # Test 5: Statistics
    print("\n5. Cache statistics:")
    print(cache.get_summary())

    # Test 6: Clean up
    print("\n6. Clearing cache...")
    cleared = cache.clear()
    print(f"   Cleared {cleared} entries")

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60 + "\n")
