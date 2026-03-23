#!/usr/bin/env python3
"""
Minimal Test - Find the exact issue
==================================
This will test each import one by one to find what's causing the hang
"""

import sys
import time
import os

def clear_and_print(message):
    """Clear screen and print message"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print(message)
    sys.stdout.flush()
    time.sleep(0.1)

def test_import(module_name, import_statement):
    """Test a single import and report result"""
    print(f"Testing import: {module_name}... ", end="", flush=True)
    
    try:
        exec(import_statement)
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    clear_and_print("🔍 IMPORT TESTING - Finding the problematic import")
    print("=" * 60)
    print()
    
    # Test imports one by one
    imports_to_test = [
        ("pyrealsense2", "import pyrealsense2 as rs"),
        ("cv2", "import cv2"),
        ("numpy", "import numpy as np"),
        ("time", "import time"),
        ("datetime", "from datetime import datetime"),
        ("threading", "import threading"),
        ("queue", "import queue"),
        ("pathlib", "from pathlib import Path"),
        ("json", "import json"),
        ("platform", "import platform"),
    ]
    
    print("Testing basic imports:")
    for name, statement in imports_to_test:
        test_import(name, statement)
        time.sleep(0.1)
    
    print("\nTesting optional imports:")
    optional_imports = [
        ("scipy", "from scipy import ndimage; from scipy.ndimage import gaussian_filter"),
        ("skimage", "from skimage import segmentation, measure, filters"),
        ("matplotlib", "import matplotlib.pyplot as plt; from matplotlib import cm"),
        ("ultralytics", "from ultralytics import YOLO"),
        ("pygame", "import pygame; pygame.mixer.init()"),
        ("torch", "import torch"),
        ("tkinter", "import tkinter as tk; from tkinter import ttk"),
    ]
    
    for name, statement in optional_imports:
        test_import(name, statement)
        time.sleep(0.1)
    
    print("\n" + "=" * 60)
    print("✅ Import testing complete!")
    print()
    
    # Now test the menu display
    print("Testing menu display...")
    time.sleep(1)
    
    clear_and_print("🎯 MENU DISPLAY TEST")
    print("=" * 40)
    print()
    
    menu_items = [
        "1. Standard Mode",
        "2. FLIR Mode", 
        "3. Iron Mode",
        "4. Rainbow Mode",
        "5. High Performance",
        "6. Interactive Mode",
        "7. Custom Config",
        "8. Demo All",
        "9. Quick Start"
    ]
    
    for i, item in enumerate(menu_items):
        print(item)
        sys.stdout.flush()
        time.sleep(0.05)
    
    print()
    print("=" * 40)
    print()
    
    # Test input
    try:
        choice = input("👆 If you can see this menu, type any number (1-9): ").strip()
        print(f"✅ You entered: {choice}")
        
        if choice == '9':
            print("\n🚀 Testing quick camera connection...")
            test_quick_camera()
        else:
            print(f"✅ Menu and input working! Choice: {choice}")
            
    except Exception as e:
        print(f"❌ Input error: {e}")

def test_quick_camera():
    """Quick camera test"""
    try:
        import pyrealsense2 as rs
        import cv2
        import numpy as np
        
        print("📷 Testing RealSense connection...")
        
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("❌ No camera found")
            return
        
        print(f"✅ Found: {devices[0].get_info(rs.camera_info.name)}")
        
        # Quick test - just try to start pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        print("🚀 Starting camera pipeline...")
        
        profile = pipeline.start(config)
        
        print("✅ Camera started successfully!")
        print("Testing 10 frames...")
        
        for i in range(10):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame and depth_frame:
                print(f"  Frame {i+1}: ✅")
            else:
                print(f"  Frame {i+1}: ❌")
        
        pipeline.stop()
        print("✅ Camera test complete!")
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")
