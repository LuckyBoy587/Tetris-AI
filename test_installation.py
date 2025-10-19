"""
Test script to verify installation and basic functionality.
Run this after installing dependencies to ensure everything works.
"""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import pygame
        print(f"  ✓ pygame {pygame.version.ver}")
    except ImportError as e:
        print(f"  ✗ pygame failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"  ✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ⚠ matplotlib not found (optional): {e}")
    
    return True


def test_tetris_env():
    """Test that TetrisEnv works correctly."""
    print("\nTesting TetrisEnv...")
    
    try:
        from tetris_env import TetrisEnv
        
        # Create headless environment
        env = TetrisEnv(render_mode=False)
        print("  ✓ TetrisEnv created")
        
        # Test reset
        state = env.reset()
        assert state.shape == (20, 10), f"Invalid state shape: {state.shape}"
        print("  ✓ reset() works")
        
        # Test step
        state, reward, done, info = env.step(4)  # Hard drop
        assert isinstance(reward, (int, float)), "Invalid reward type"
        assert isinstance(done, bool), "Invalid done type"
        assert 'score' in info, "Missing score in info"
        print("  ✓ step() works")
        
        # Test features
        features = env.get_features()
        assert 'aggregate_height' in features, "Missing aggregate_height"
        assert 'holes' in features, "Missing holes"
        assert 'bumpiness' in features, "Missing bumpiness"
        assert 'completed_lines' in features, "Missing completed_lines"
        print("  ✓ get_features() works")
        
        # Play a few moves
        for _ in range(50):
            if done:
                break
            state, reward, done, info = env.step(4)
        
        print(f"  ✓ Played 50 steps (score: {info['score']})")
        
        env.close()
        print("  ✓ TetrisEnv test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ TetrisEnv test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_genetic_agent():
    """Test that GeneticAgent can be imported and created."""
    print("\nTesting GeneticAgent...")
    
    try:
        from genetic_agent import GeneticAgent, TetrisChromosome
        import numpy as np
        
        # Create chromosome
        genes = np.array([-0.5, -0.7, -0.3, 0.8])
        chromosome = TetrisChromosome(genes)
        assert len(chromosome.genes) == 4, "Invalid chromosome genes"
        print("  ✓ TetrisChromosome created")
        
        # Create agent
        agent = GeneticAgent(population_size=5)
        assert len(agent.population) == 5, "Invalid population size"
        print("  ✓ GeneticAgent created")
        
        # Test evaluation (quick, 1 game)
        from tetris_env import TetrisEnv
        env = TetrisEnv(render_mode=False)
        actions = agent.choose_action(env, chromosome)
        assert isinstance(actions, list), "Invalid actions type"
        assert len(actions) > 0, "No actions returned"
        print("  ✓ choose_action() works")
        
        env.close()
        print("  ✓ GeneticAgent test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ GeneticAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test that visualization module can be imported."""
    print("\nTesting visualization...")
    
    try:
        import visualize_training
        print("  ✓ visualize_training module imported")
        print("  ℹ Run after training to generate plots")
        return True
    except ImportError as e:
        print(f"  ⚠ visualize_training import warning: {e}")
        print("  ℹ matplotlib may not be installed (optional)")
        return True
    except Exception as e:
        print(f"  ✗ Visualization test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("TETRIS AI - INSTALLATION TEST")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    if not results[0][1]:
        print("\n" + "="*60)
        print("CRITICAL: Missing dependencies!")
        print("="*60)
        print("\nPlease install required packages:")
        print("  pip install pygame numpy")
        print("\nOptional (for visualization):")
        print("  pip install matplotlib")
        return False
    
    # Test components
    results.append(("TetrisEnv", test_tetris_env()))
    results.append(("GeneticAgent", test_genetic_agent()))
    results.append(("Visualization", test_visualization()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("  1. Run quick demo:  python demo.py")
        print("  2. Train an agent:  python genetic_agent.py --mode train")
        print("  3. See GA_USAGE_GUIDE.md for detailed instructions")
        return True
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
