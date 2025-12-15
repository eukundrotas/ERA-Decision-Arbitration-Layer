"""
Response Caching Module
Level 2 Upgrade: Cache LLM responses to reduce costs and latency

Features:
- In-memory LRU cache
- File-based persistent cache
- Semantic similarity matching
- TTL-based expiration
- Cache statistics
"""

import json
import hashlib
import logging
import time
import os
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry"""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float]
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """Update access time and hit count"""
        self.last_accessed = time.time()
        self.hit_count += 1


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": f"{self.hit_rate:.2%}"
        }


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = 3600  # 1 hour default
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats(max_size=max_size)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self.stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired():
                del self._cache[key]
                self.stats.misses += 1
                self.stats.size = len(self._cache)
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self.stats.hits += 1
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Set value in cache"""
        with self._lock:
            # Calculate expiration
            actual_ttl = ttl if ttl is not None else self.default_ttl
            expires_at = time.time() + actual_ttl if actual_ttl else None
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            # Update or insert
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                # Evict if necessary
                while len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                    self.stats.evictions += 1
            
            self._cache[key] = entry
            self.stats.size = len(self._cache)
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.stats.size = len(self._cache)
                return True
            return False
    
    def clear(self):
        """Clear all entries"""
        with self._lock:
            self._cache.clear()
            self.stats.size = 0
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() 
                if v.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            
            self.stats.size = len(self._cache)
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.stats.to_dict()


class ResponseCache:
    """
    Specialized cache for LLM responses.
    
    Supports:
    - Exact key matching
    - Fuzzy/semantic matching
    - Model-specific caching
    """
    
    def __init__(
        self,
        max_size: int = 500,
        default_ttl: int = 3600,
        storage_path: Optional[str] = None,
        enable_fuzzy: bool = True,
        fuzzy_threshold: float = 0.85
    ):
        self.cache = LRUCache(max_size=max_size, default_ttl=default_ttl)
        self.storage_path = Path(storage_path) if storage_path else None
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        
        # Index for fuzzy matching
        self._prompt_index: Dict[str, List[str]] = {}  # normalized_prefix -> [cache_keys]
        
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for comparison"""
        # Lowercase, remove extra whitespace
        normalized = prompt.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _get_prompt_prefix(self, prompt: str, length: int = 50) -> str:
        """Get normalized prefix for indexing"""
        normalized = self._normalize_prompt(prompt)
        return normalized[:length]
    
    def _generate_cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        temperature: float
    ) -> str:
        """Generate unique cache key"""
        key_data = {
            "system": self._normalize_prompt(system_prompt)[:200],
            "user": self._normalize_prompt(user_prompt),
            "model": model_id,
            "temp": round(temperature, 2)
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:32]
    
    def _calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate simple similarity between prompts"""
        p1 = set(self._normalize_prompt(prompt1).split())
        p2 = set(self._normalize_prompt(prompt2).split())
        
        if not p1 or not p2:
            return 0.0
        
        intersection = len(p1 & p2)
        union = len(p1 | p2)
        
        return intersection / union if union > 0 else 0.0
    
    def get(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        temperature: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response.
        
        Returns dict with 'response' and 'cache_info' if found.
        """
        # Try exact match first
        key = self._generate_cache_key(system_prompt, user_prompt, model_id, temperature)
        cached = self.cache.get(key)
        
        if cached:
            return {
                "response": cached,
                "cache_info": {
                    "match_type": "exact",
                    "key": key
                }
            }
        
        # Try fuzzy match if enabled
        if self.enable_fuzzy:
            prefix = self._get_prompt_prefix(user_prompt)
            
            if prefix in self._prompt_index:
                for candidate_key in self._prompt_index[prefix]:
                    entry = self.cache._cache.get(candidate_key)
                    if entry and not entry.is_expired():
                        stored_prompt = entry.metadata.get("user_prompt", "")
                        stored_model = entry.metadata.get("model_id", "")
                        
                        # Check model match and similarity
                        if stored_model == model_id:
                            similarity = self._calculate_similarity(user_prompt, stored_prompt)
                            if similarity >= self.fuzzy_threshold:
                                entry.touch()
                                self.cache.stats.hits += 1
                                return {
                                    "response": entry.value,
                                    "cache_info": {
                                        "match_type": "fuzzy",
                                        "similarity": similarity,
                                        "key": candidate_key
                                    }
                                }
        
        return None
    
    def set(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        temperature: float,
        response: str,
        ttl: Optional[int] = None
    ):
        """Cache a response"""
        key = self._generate_cache_key(system_prompt, user_prompt, model_id, temperature)
        
        metadata = {
            "system_prompt": system_prompt[:100],
            "user_prompt": user_prompt[:500],
            "model_id": model_id,
            "temperature": temperature,
            "response_length": len(response)
        }
        
        self.cache.set(key, response, ttl=ttl, metadata=metadata)
        
        # Update fuzzy index
        if self.enable_fuzzy:
            prefix = self._get_prompt_prefix(user_prompt)
            if prefix not in self._prompt_index:
                self._prompt_index[prefix] = []
            if key not in self._prompt_index[prefix]:
                self._prompt_index[prefix].append(key)
                # Limit index size per prefix
                if len(self._prompt_index[prefix]) > 10:
                    self._prompt_index[prefix] = self._prompt_index[prefix][-10:]
        
        # Persist to disk
        self._save_entry(key, response, metadata)
    
    def _save_entry(self, key: str, response: str, metadata: Dict[str, Any]):
        """Save entry to disk"""
        if not self.storage_path:
            return
        
        try:
            entry_data = {
                "key": key,
                "response": response,
                "metadata": metadata,
                "saved_at": datetime.now().isoformat()
            }
            
            file_path = self.storage_path / f"{key}.json"
            with open(file_path, 'w') as f:
                json.dump(entry_data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache entry: {e}")
    
    def _load_from_disk(self):
        """Load cached entries from disk"""
        if not self.storage_path:
            return
        
        loaded = 0
        for file in self.storage_path.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                
                key = data["key"]
                response = data["response"]
                metadata = data.get("metadata", {})
                
                self.cache.set(key, response, metadata=metadata)
                
                # Rebuild fuzzy index
                if self.enable_fuzzy and "user_prompt" in metadata:
                    prefix = self._get_prompt_prefix(metadata["user_prompt"])
                    if prefix not in self._prompt_index:
                        self._prompt_index[prefix] = []
                    self._prompt_index[prefix].append(key)
                
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load cache entry {file}: {e}")
        
        logger.info(f"Loaded {loaded} cached entries from disk")
    
    def invalidate_model(self, model_id: str) -> int:
        """Invalidate all cache entries for a specific model"""
        invalidated = 0
        keys_to_delete = []
        
        for key, entry in self.cache._cache.items():
            if entry.metadata.get("model_id") == model_id:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            self.cache.delete(key)
            invalidated += 1
        
        return invalidated
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache.get_stats()
        stats["fuzzy_index_size"] = len(self._prompt_index)
        stats["storage_enabled"] = self.storage_path is not None
        return stats
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self._prompt_index.clear()
        
        if self.storage_path:
            for file in self.storage_path.glob("*.json"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {file}: {e}")


class CachedProvider:
    """
    Wrapper that adds caching to an LLM provider.
    """
    
    def __init__(
        self,
        provider,
        cache: Optional[ResponseCache] = None,
        cache_config: Optional[Dict[str, Any]] = None
    ):
        self.provider = provider
        self.cache = cache or ResponseCache(**(cache_config or {}))
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_cache: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response with caching.
        
        Returns (response, metadata) tuple.
        """
        metadata = {"cached": False}
        
        # Try cache first
        if use_cache:
            cached = self.cache.get(system_prompt, user_prompt, model_id, temperature)
            if cached:
                metadata["cached"] = True
                metadata["cache_info"] = cached["cache_info"]
                return cached["response"], metadata
        
        # Generate fresh response
        response = self.provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Cache the response
        if use_cache:
            self.cache.set(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_id=model_id,
                temperature=temperature,
                response=response
            )
        
        return response, metadata
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()


# Singleton instances
response_cache = ResponseCache(max_size=500, default_ttl=3600)


def get_cached_response(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    temperature: float = 0.7
) -> Optional[Dict[str, Any]]:
    """Convenience function to get cached response"""
    return response_cache.get(system_prompt, user_prompt, model_id, temperature)


def cache_response(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    temperature: float,
    response: str,
    ttl: Optional[int] = None
):
    """Convenience function to cache a response"""
    response_cache.set(
        system_prompt, user_prompt, model_id, 
        temperature, response, ttl
    )
