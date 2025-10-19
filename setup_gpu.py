"""
Setup script for GPU-optimized training environment.
Run this to check your system and install necessary packages.
"""

import sys
import subprocess


def check_python_version():
    """Check if Python version is adequate."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} - Too old!")
        print("   Please upgrade to Python 3.8 or higher")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"   ✓ {package_name} installed")
        return True
    except ImportError:
        print(f"   ✗ {package_name} not installed")
        return False


def check_cuda():
    """Check if CUDA is available."""
    print("\n🔍 Checking CUDA availability...")
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            # Parse CUDA version from output
            output = result.stdout
            if 'release' in output:
                version_line = [line for line in output.split('\n') if 'release' in line][0]
                print(f"   ✓ CUDA detected: {version_line.strip()}")
                return True
        print("   ⚠ CUDA not detected (GPU acceleration unavailable)")
        print("   Training will still work using CPU parallelization")
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        print("   ⚠ CUDA not detected (GPU acceleration unavailable)")
        print("   Training will still work using CPU parallelization")
        return False


def check_cupy():
    """Check if CuPy is installed and working."""
    print("\n🔍 Checking CuPy (GPU acceleration)...")
    try:
        import cupy as cp
        # Try a simple operation
        arr = cp.array([1, 2, 3])
        result = cp.sum(arr)
        print(f"   ✓ CuPy installed and working")
        print(f"   ✓ GPU acceleration ENABLED")
        return True
    except ImportError:
        print("   ✗ CuPy not installed")
        print("   💡 GPU acceleration available but not installed")
        return False
    except Exception as e:
        print(f"   ⚠ CuPy installed but not working: {e}")
        print("   💡 Check CUDA installation")
        return False


def count_cpu_cores():
    """Count available CPU cores."""
    print("\n🔍 Checking CPU cores...")
    try:
        import multiprocessing
        cores = multiprocessing.cpu_count()
        print(f"   ✓ {cores} CPU cores detected")
        print(f"   ✓ Parallel processing will use all cores")
        return cores
    except Exception:
        print("   ⚠ Could not detect CPU cores")
        return 1


def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                      check=True)
        print("   ✓ Base requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("   ✗ Failed to install requirements")
        return False


def install_cupy():
    """Prompt and install CuPy if desired."""
    print("\n🎮 CuPy Installation")
    print("   CuPy provides GPU acceleration for numerical operations.")
    print("   It requires a compatible CUDA installation.")
    print()
    print("   CUDA version guide:")
    print("   - CUDA 12.x: pip install cupy-cuda12x")
    print("   - CUDA 11.x: pip install cupy-cuda11x")
    print("   - For other versions: https://docs.cupy.dev/en/stable/install.html")
    print()
    
    choice = input("   Install CuPy for CUDA 12.x? (y/n): ").lower()
    if choice == 'y':
        print("\n   Installing CuPy for CUDA 12.x...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'cupy-cuda12x'],
                          check=True)
            print("   ✓ CuPy installed successfully")
            print("   🎉 GPU acceleration enabled!")
            return True
        except subprocess.CalledProcessError:
            print("   ✗ Failed to install CuPy")
            print("   💡 You may need to install a different version for your CUDA")
            return False
    else:
        print("   ⊘ Skipping CuPy installation")
        print("   💡 You can install it later with: pip install cupy-cuda12x")
        return False


def main():
    """Main setup function."""
    print("="*70)
    print("🚀 TETRIS AI - GPU-OPTIMIZED TRAINING SETUP")
    print("="*70)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check existing packages
    print("\n🔍 Checking installed packages...")
    pygame_ok = check_package('pygame')
    numpy_ok = check_package('numpy')
    matplotlib_ok = check_package('matplotlib')
    
    # Install requirements if needed
    if not (pygame_ok and numpy_ok and matplotlib_ok):
        print("\n📦 Some packages are missing.")
        choice = input("   Install missing packages? (y/n): ").lower()
        if choice == 'y':
            install_requirements()
        else:
            print("   ⚠ Warning: Missing packages may cause errors")
    
    # Check CUDA
    cuda_available = check_cuda()
    
    # Check CuPy
    cupy_ok = check_cupy()
    
    # Offer to install CuPy if CUDA is available but CuPy is not
    if cuda_available and not cupy_ok:
        install_cupy()
    
    # Check CPU cores
    cores = count_cpu_cores()
    
    # Summary
    print("\n" + "="*70)
    print("📊 SYSTEM SUMMARY")
    print("="*70)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"CPU Cores: {cores}")
    print(f"CUDA Available: {'✓ Yes' if cuda_available else '✗ No'}")
    print(f"GPU Acceleration (CuPy): {'✓ Enabled' if check_package('cupy') else '✗ Disabled'}")
    print()
    
    if check_package('cupy'):
        print("🎉 Your system is FULLY OPTIMIZED for GPU training!")
        print()
        print("Recommended command:")
        print("  python demo_gpu.py --intensive")
        print()
        print("Or for maximum performance:")
        print("  python genetic_agent_gpu.py --mode train --population 500 --generations 100")
    else:
        print("✓ Your system is ready for CPU-PARALLEL training!")
        print()
        print("💡 For GPU acceleration, install CuPy:")
        print("  pip install cupy-cuda12x  # For CUDA 12.x")
        print()
        print("Recommended command:")
        print("  python demo_gpu.py --intensive")
        print()
        print("Expected speedup: 10-30x faster than sequential version")
    
    print("\n" + "="*70)
    print("📚 NEXT STEPS")
    print("="*70)
    print("1. Quick test (5 min):")
    print("   python demo_gpu.py")
    print()
    print("2. Intensive training (30-60 min):")
    print("   python demo_gpu.py --intensive")
    print()
    print("3. Benchmark your system:")
    print("   python demo_gpu.py --benchmark")
    print()
    print("4. Read the full guide:")
    print("   See GPU_TRAINING_GUIDE.md for detailed instructions")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error during setup: {e}")
        import traceback
        traceback.print_exc()
