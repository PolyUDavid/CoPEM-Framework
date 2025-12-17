#!/usr/bin/env python3
"""
CoPEM Framework Installation Validation Script

This script validates that the CoPEM framework is correctly installed
and all dependencies are available.

Usage:
    python scripts/validate_installation.py

Date: December 15, 2025
"""

import sys
import importlib
from typing import List, Tuple

def check_python_version() -> bool:
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
        return False

def check_dependencies() -> List[Tuple[str, bool]]:
    """Check required dependencies"""
    required_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("osqp", "OSQP"),
        ("tqdm", "tqdm"),
    ]
    
    results = []
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
            results.append((name, True))
        except ImportError:
            print(f"❌ {name} (not installed)")
            results.append((name, False))
    
    return results

def check_copem_modules() -> bool:
    """Check CoPEM modules"""
    try:
        import copem
        print(f"✅ CoPEM Framework v{copem.__version__}")
        
        from copem import CoESDRLAgent, EcoTESTransformer, CoPEM_API
        print("✅ Co-ESDRL Agent")
        print("✅ Eco-TES Transformer")
        print("✅ CoPEM API")
        
        return True
    except ImportError as e:
        print(f"❌ CoPEM Framework: {e}")
        return False

def check_cuda_availability():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available (CPU-only mode)")
    except:
        print("⚠️  Could not check CUDA availability")

def check_data_files():
    """Check experimental data files"""
    import os
    data_dir = "data/paper_data"
    
    required_files = [
        "copem_complete_experiment_results_20250714_151845.json",
        "copem_case3_fleet_cooperative_results_20250714_172108.json",
    ]
    
    all_present = True
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} (missing)")
            all_present = False
    
    return all_present

def main():
    """Main validation function"""
    print("=" * 70)
    print("  CoPEM Framework Installation Validation")
    print("  Date: December 15, 2025")
    print("=" * 70)
    print()
    
    # Check Python version
    print("📌 Checking Python version...")
    python_ok = check_python_version()
    print()
    
    # Check dependencies
    print("📌 Checking dependencies...")
    deps_results = check_dependencies()
    deps_ok = all(result[1] for result in deps_results)
    print()
    
    # Check CoPEM modules
    print("📌 Checking CoPEM modules...")
    copem_ok = check_copem_modules()
    print()
    
    # Check CUDA
    print("📌 Checking CUDA availability...")
    check_cuda_availability()
    print()
    
    # Check data files
    print("📌 Checking experimental data...")
    data_ok = check_data_files()
    print()
    
    # Summary
    print("=" * 70)
    print("  Validation Summary")
    print("=" * 70)
    
    if python_ok and deps_ok and copem_ok and data_ok:
        print("✅ All checks passed! CoPEM framework is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run Euro NCAP tests: python experiments/run_euro_ncap_tests.py")
        print("  2. Run fleet tests: python experiments/run_fleet_tests.py")
        print("  3. Generate figures: python scripts/generate_paper_figures.py")
        return 0
    else:
        print("❌ Some checks failed. Please install missing dependencies.")
        print()
        print("To install all dependencies:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

