"""
Run all tests and display results
"""
import pytest
import sys

if __name__ == "__main__":
    print("=" * 60)
    print("BLACKBOX TEST SUITE")
    print("=" * 60)
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--no-header",
        "-x"  # Stop on first failure
    ])
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    sys.exit(exit_code)
