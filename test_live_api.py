#!/usr/bin/env python3
"""
Live API Test Script for ERA DAL
Tests the system with a real OpenRouter API key
"""

import os
import sys
import json
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent))

def test_api_connection():
    """Test basic API connectivity"""
    from src.config import config
    from src.providers import provider
    
    print("=" * 60)
    print("🧪 ERA DAL - Live API Test")
    print("=" * 60)
    
    # Check API key
    if not config.openrouter_api_key:
        print("❌ OPENROUTER_API_KEY not set!")
        print("   Set it in .env file or export OPENROUTER_API_KEY=sk-...")
        return False
    
    print(f"✅ API Key configured: {config.openrouter_api_key[:12]}...")
    print(f"📡 Base URL: {config.openrouter_base_url}")
    
    # Test simple generation
    print("\n🔄 Testing API connection with a simple query...")
    try:
        response = provider.generate(
            system_prompt="You are a helpful assistant. Reply in one short sentence.",
            user_prompt="What is 2 + 2?",
            model_id="openai/gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=100
        )
        print(f"✅ API Response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False

def test_solver_pool():
    """Test solver pool with multiple models"""
    from src.solver_pool import SolverPool
    from src.config import config
    
    print("\n" + "=" * 60)
    print("🔄 Testing Solver Pool (science domain)")
    print("=" * 60)
    
    pool = SolverPool(domain="science")
    print(f"📊 Models in pool: {len(pool.solvers)}")
    for solver in pool.solvers:
        print(f"   - {solver.model_id} (temp={solver.temperature})")
    
    # Test with a simple science question
    problem = "Why is the sky blue? Give a brief scientific explanation."
    print(f"\n📝 Test problem: {problem}")
    print("⏳ Running solvers (this may take 30-60 seconds)...")
    
    try:
        results = pool.run(problem)
        print(f"\n✅ Got {len(results)} responses:")
        for i, result in enumerate(results, 1):
            answer_preview = result.answer[:100] + "..." if len(result.answer) > 100 else result.answer
            print(f"   {i}. [{result.model_id}] conf={result.confidence:.2f}")
            print(f"      {answer_preview}")
        return results
    except Exception as e:
        print(f"❌ Solver Pool Error: {e}")
        return None

def test_full_orchestration():
    """Test the full orchestration pipeline"""
    from src.orchestrator import Orchestrator
    
    print("\n" + "=" * 60)
    print("🎯 Testing Full Orchestration Pipeline")
    print("=" * 60)
    
    orchestrator = Orchestrator(
        pool_name="science",
        repeats=2,  # Меньше для теста
        consensus_topk=2,
        epsilon=0.1,
        enable_rebuttal=False  # Отключим для быстрого теста
    )
    
    problem = "What is photosynthesis? Explain briefly."
    print(f"📝 Test problem: {problem}")
    print("⏳ Running full pipeline (this may take 1-2 minutes)...")
    
    try:
        result = orchestrator.run(problem)
        
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"🎯 Final Answer: {result['final_answer'][:200]}...")
        print(f"📈 Majority Rate: {result['stability'].get('majority_rate', 'N/A')}")
        print(f"📉 CI Lower: {result['stability'].get('ci_lower', 'N/A')}")
        print(f"📈 CI Upper: {result['stability'].get('ci_upper', 'N/A')}")
        print(f"🔄 Total Runs: {len(result.get('runs', []))}")
        
        return result
    except Exception as e:
        print(f"❌ Orchestration Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("\n" + "🚀" * 30)
    print("ERA Decision & Arbitration Layer - Live Test Suite")
    print("🚀" * 30 + "\n")
    
    # Test 1: API Connection
    if not test_api_connection():
        print("\n⚠️  API connection failed. Please check your API key.")
        sys.exit(1)
    
    # Test 2: Solver Pool
    results = test_solver_pool()
    if not results:
        print("\n⚠️  Solver pool test failed.")
        sys.exit(1)
    
    # Test 3: Full Orchestration (optional, takes longer)
    response = input("\n🤔 Run full orchestration test? (takes 1-2 min) [y/N]: ")
    if response.lower() == 'y':
        test_full_orchestration()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
