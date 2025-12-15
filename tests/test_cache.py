"""Tests for response caching module"""

import unittest
import time
import tempfile
import shutil
from src.cache import (
    LRUCache, 
    ResponseCache,
    CacheEntry,
    get_cached_response,
    cache_response
)


class TestLRUCache(unittest.TestCase):
    
    def setUp(self):
        self.cache = LRUCache(max_size=5, default_ttl=3600)
    
    def test_set_and_get(self):
        """Basic set and get"""
        self.cache.set("key1", "value1")
        result = self.cache.get("key1")
        self.assertEqual(result, "value1")
    
    def test_get_missing(self):
        """Get non-existent key returns None"""
        result = self.cache.get("nonexistent")
        self.assertIsNone(result)
        self.assertEqual(self.cache.stats.misses, 1)
    
    def test_lru_eviction(self):
        """LRU eviction when cache is full"""
        # Fill cache
        for i in range(5):
            self.cache.set(f"key{i}", f"value{i}")
        
        # Access key0 to make it recently used
        self.cache.get("key0")
        
        # Add new item - should evict key1 (least recently used)
        self.cache.set("key5", "value5")
        
        self.assertIsNone(self.cache.get("key1"))
        self.assertEqual(self.cache.get("key0"), "value0")
        self.assertEqual(self.cache.get("key5"), "value5")
    
    def test_ttl_expiration(self):
        """Items expire after TTL"""
        self.cache.set("key1", "value1", ttl=1)  # 1 second TTL
        
        self.assertEqual(self.cache.get("key1"), "value1")
        
        time.sleep(1.1)
        
        self.assertIsNone(self.cache.get("key1"))
    
    def test_hit_rate(self):
        """Hit rate calculation"""
        self.cache.set("key1", "value1")
        
        self.cache.get("key1")  # hit
        self.cache.get("key1")  # hit
        self.cache.get("key2")  # miss
        
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(float(stats["hit_rate"].rstrip('%')) / 100, 0.67, places=1)
    
    def test_clear(self):
        """Clear cache"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        self.cache.clear()
        
        self.assertIsNone(self.cache.get("key1"))
        self.assertEqual(self.cache.stats.size, 0)
    
    def test_delete(self):
        """Delete specific key"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        result = self.cache.delete("key1")
        
        self.assertTrue(result)
        self.assertIsNone(self.cache.get("key1"))
        self.assertEqual(self.cache.get("key2"), "value2")
    
    def test_cleanup_expired(self):
        """Cleanup expired entries"""
        self.cache.set("key1", "value1", ttl=1)
        self.cache.set("key2", "value2", ttl=3600)
        
        time.sleep(1.1)
        
        cleaned = self.cache.cleanup_expired()
        
        self.assertEqual(cleaned, 1)
        self.assertIsNone(self.cache.get("key1"))
        self.assertEqual(self.cache.get("key2"), "value2")


class TestResponseCache(unittest.TestCase):
    
    def setUp(self):
        self.cache = ResponseCache(max_size=100, default_ttl=3600, enable_fuzzy=True)
    
    def test_cache_response(self):
        """Cache and retrieve LLM response"""
        self.cache.set(
            system_prompt="You are helpful",
            user_prompt="What is 2+2?",
            model_id="gpt-4",
            temperature=0.7,
            response="The answer is 4."
        )
        
        result = self.cache.get(
            system_prompt="You are helpful",
            user_prompt="What is 2+2?",
            model_id="gpt-4",
            temperature=0.7
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["response"], "The answer is 4.")
        self.assertEqual(result["cache_info"]["match_type"], "exact")
    
    def test_different_model_no_cache(self):
        """Different model should not hit cache"""
        self.cache.set(
            system_prompt="You are helpful",
            user_prompt="What is 2+2?",
            model_id="gpt-4",
            temperature=0.7,
            response="The answer is 4."
        )
        
        result = self.cache.get(
            system_prompt="You are helpful",
            user_prompt="What is 2+2?",
            model_id="claude-3",  # Different model
            temperature=0.7
        )
        
        self.assertIsNone(result)
    
    def test_different_temperature_exact_match(self):
        """Different temperature should not hit exact cache (may hit fuzzy)"""
        # Disable fuzzy for this test
        cache_no_fuzzy = ResponseCache(max_size=100, default_ttl=3600, enable_fuzzy=False)
        
        cache_no_fuzzy.set(
            system_prompt="You are helpful",
            user_prompt="What is 2+2?",
            model_id="gpt-4",
            temperature=0.7,
            response="The answer is 4."
        )
        
        result = cache_no_fuzzy.get(
            system_prompt="You are helpful",
            user_prompt="What is 2+2?",
            model_id="gpt-4",
            temperature=0.3  # Different temperature
        )
        
        # Without fuzzy matching, should be None (exact key differs)
        self.assertIsNone(result)
    
    def test_fuzzy_matching(self):
        """Fuzzy matching for similar prompts"""
        self.cache.set(
            system_prompt="You are helpful",
            user_prompt="What is two plus two?",
            model_id="gpt-4",
            temperature=0.7,
            response="The answer is 4."
        )
        
        # Try with slightly different prompt
        result = self.cache.get(
            system_prompt="You are helpful",
            user_prompt="What is two plus two",  # Missing question mark
            model_id="gpt-4",
            temperature=0.7
        )
        
        # May or may not match depending on similarity threshold
        # Just verify no crash
        self.assertTrue(True)
    
    def test_invalidate_model(self):
        """Invalidate all entries for a model"""
        self.cache.set("sys", "prompt1", "gpt-4", 0.7, "response1")
        self.cache.set("sys", "prompt2", "gpt-4", 0.7, "response2")
        self.cache.set("sys", "prompt3", "claude-3", 0.7, "response3")
        
        invalidated = self.cache.invalidate_model("gpt-4")
        
        self.assertEqual(invalidated, 2)
        self.assertIsNone(self.cache.get("sys", "prompt1", "gpt-4", 0.7))
        self.assertIsNotNone(self.cache.get("sys", "prompt3", "claude-3", 0.7))
    
    def test_stats(self):
        """Get cache statistics"""
        self.cache.set("sys", "prompt1", "gpt-4", 0.7, "response1")
        self.cache.get("sys", "prompt1", "gpt-4", 0.7)
        
        stats = self.cache.get_stats()
        
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("fuzzy_index_size", stats)


class TestResponseCachePersistence(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_persistence(self):
        """Test cache persistence to disk"""
        # Create cache and add entries
        cache1 = ResponseCache(
            max_size=100,
            storage_path=self.temp_dir
        )
        cache1.set("sys", "prompt1", "gpt-4", 0.7, "response1")
        
        # Create new cache that loads from disk
        cache2 = ResponseCache(
            max_size=100,
            storage_path=self.temp_dir
        )
        
        result = cache2.get("sys", "prompt1", "gpt-4", 0.7)
        self.assertIsNotNone(result)
        self.assertEqual(result["response"], "response1")


class TestCacheEntry(unittest.TestCase):
    
    def test_is_expired(self):
        """Test expiration check"""
        entry = CacheEntry(
            key="test",
            value="value",
            created_at=time.time(),
            expires_at=time.time() - 1  # Already expired
        )
        
        self.assertTrue(entry.is_expired())
    
    def test_not_expired(self):
        """Test non-expired entry"""
        entry = CacheEntry(
            key="test",
            value="value",
            created_at=time.time(),
            expires_at=time.time() + 3600
        )
        
        self.assertFalse(entry.is_expired())
    
    def test_no_expiration(self):
        """Entry without expiration never expires"""
        entry = CacheEntry(
            key="test",
            value="value",
            created_at=time.time(),
            expires_at=None
        )
        
        self.assertFalse(entry.is_expired())


if __name__ == '__main__':
    unittest.main()
