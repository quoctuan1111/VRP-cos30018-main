import sys
import pkg_resources
import platform
import psutil
import numpy as np
import matplotlib.pyplot as plt

def check_system_requirements():
    print("\n=== System Requirements Check ===")
    
    # Check Python version
    print("\nPython Version:", sys.version.split()[0])
    python_ok = sys.version_info >= (3, 7)
    print("Python 3.7+ Required:", "✓" if python_ok else "✗")

    # Check required packages
    required_packages = {
        'numpy': '1.19.0',
        'pandas': '1.1.0',
        'matplotlib': '3.3.0',
        'scikit-learn': '0.23.0',
        'ortools': '9.0.0',
        'joblib': '0.17.0'
    }

    print("\nPackage Requirements:")
    packages_ok = True
    for package, min_version in required_packages.items():
        try:
            version = pkg_resources.get_distribution(package).version
            meets_req = pkg_resources.parse_version(version) >= pkg_resources.parse_version(min_version)
            print(f"{package}: {version} {'✓' if meets_req else '✗'}")
            packages_ok = packages_ok and meets_req
        except pkg_resources.DistributionNotFound:
            print(f"{package}: Not installed ✗")
            packages_ok = False

    # Check system resources
    print("\nSystem Resources:")
    
    # CPU
    cpu_count = psutil.cpu_count()
    cpu_ok = cpu_count >= 2
    print(f"CPU Cores: {cpu_count} {'✓' if cpu_ok else '✗'} (2+ recommended)")
    
    # Memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024 ** 3)
    memory_ok = memory_gb >= 4
    print(f"RAM: {memory_gb:.1f} GB {'✓' if memory_ok else '✗'} (4+ GB recommended)")
    
    # Disk
    disk = psutil.disk_usage('/')
    disk_gb = disk.free / (1024 ** 3)
    disk_ok = disk_gb >= 1
    print(f"Free Disk Space: {disk_gb:.1f} GB {'✓' if disk_ok else '✗'} (1+ GB recommended)")

    # Overall assessment
    print("\nOverall Assessment:")
    all_ok = python_ok and packages_ok and cpu_ok and memory_ok and disk_ok
    if all_ok:
        print("✓ Your system meets all requirements!")
    else:
        print("✗ Your system doesn't meet some requirements.")
        print("\nRecommended Actions:")
        if not python_ok:
            print("- Upgrade Python to version 3.7 or higher")
        if not packages_ok:
            print("- Install/upgrade required packages:")
            print("  pip install --upgrade numpy pandas matplotlib scikit-learn ortools joblib")
        if not cpu_ok:
            print("- Consider using a machine with more CPU cores")
        if not memory_ok:
            print("- Increase available RAM")
        if not disk_ok:
            print("- Free up disk space")

    # Test basic functionality
    print("\nTesting Basic Functionality:")
    try:
        # Test numpy
        arr = np.array([1, 2, 3])
        print("NumPy Array Operations: ✓")
        
        # Test matplotlib
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.close()
        print("Matplotlib Plotting: ✓")
        
    except Exception as e:
        print(f"Functionality Test Failed: {str(e)}")

if __name__ == "__main__":
    check_system_requirements()
