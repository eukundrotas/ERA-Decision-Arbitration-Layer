"""Тесты для утилит"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import safe_parse_json

class TestSafeParseJson(unittest.TestCase):
    
    def test_plain_json(self):
        result = safe_parse_json('{"key": "value"}')
        self.assertEqual(result["key"], "value")
    
    def test_json_with_markdown_wrapper(self):
        raw = '```json\n{"key": "value"}\n```'
        result = safe_parse_json(raw)
        self.assertEqual(result["key"], "value")
    
    def test_json_with_simple_wrapper(self):
        raw = '```\n{"key": "value"}\n```'
        result = safe_parse_json(raw)
        self.assertEqual(result["key"], "value")
    
    def test_json_with_whitespace(self):
        raw = '  \n  {"key": "value"}  \n  '
        result = safe_parse_json(raw)
        self.assertEqual(result["key"], "value")
    
    def test_invalid_json(self):
        with self.assertRaises(ValueError):
            safe_parse_json('not valid json')
    
    def test_nested_json(self):
        raw = '{"outer": {"inner": "value"}, "array": [1, 2, 3]}'
        result = safe_parse_json(raw)
        self.assertEqual(result["outer"]["inner"], "value")
        self.assertEqual(result["array"], [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
