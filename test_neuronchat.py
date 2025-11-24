#!/usr/bin/env python3
"""
Test if NeuronChat is available
"""
try:
    import neuron_chat
    print("✓ NeuronChat is installed")
    print(f"  Version: {neuron_chat.__version__}")
except ImportError:
    print("✗ NeuronChat is NOT installed")
    print("\nTo install:")
    print("  pip install neuron-chat --break-system-packages")
    print("\nOr create separate environment:")
    print("  conda create -n neuronchat_env python=3.9")
    print("  conda activate neuronchat_env")
    print("  pip install neuron-chat scanpy")
