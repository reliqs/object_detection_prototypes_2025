#!/usr/bin/env python3
"""
Debug Version - Predator Vision Menu Fix
========================================
This version will help us identify why the menu isn't showing up
"""

import sys
import time
import os

def clear_screen():
    """Clear screen for better visibility"""
    os.system('cls' if os.name == 'nt' else 'clear')

def debug_print(message, delay=0.1):
    """Print with forced flush and optional delay"""
    print(message)
    sys.stdout.flush()
    if delay > 0:
        time.sleep(delay)

def show_menu_debug():
    """Debug menu display"""
    
    # Clear screen first
    clear_screen()
    
    debug_print("=" * 80)
    debug_print("🎯 PREDATOR VISION SYSTEM - DEBUG VERSION")
    debug_print("=" * 80)
    debug_print("")
    
    debug_print("🔍 CHECKING CONSOLE OUTPUT...")
    debug_print("If you can see this text, console output is working!", 0.5)
    debug_print("")
    
    debug_print("📺 MENU DISPLAY TEST:")
    debug_print("=" * 50)
    
    menu_items = [
        "1. 🎯 Standard Predator Vision (YOLO + Thermal)",
        "2. 🌡️  FLIR Thermal Camera Style", 
        "3. 🔥 Military Iron Thermal Scope",
        "4. 🌈 Scientific Rainbow Thermal",
        "5. ⚡ High Performance Mode (GPU + Threading)",
        "6. 🎮 Interactive Mode (with Control Panel)",
        "7. 🔧 Custom Configuration",
        "8. 🎬 Demo All Modes",
        "9. 🚀 Quick Start (Skip Menu)",
        "",
        "d. 🐛 Debug Mode",
        "q. 👋 Quit"
    ]
    
    for item in menu_items:
        debug_print(item, 0.02)  # Small delay between lines
    
    debug_print("=" * 50)
    debug_print("")
    
    # Test different input methods
    debug_print("🎮 INPUT TEST:")
    debug_print("If you can see this menu, the issue was display buffering!")
    debug_print("")
    
    while True:
        try:
            debug_print("👆 Please enter your choice: ", 0)
            sys.stdout.flush()
            
            choice = input().strip().lower()
            
            if choice == 'q':
                debug_print("👋 Goodbye!")
                return None
            elif choice == 'd':
                run_debug_tests()
                continue
            elif choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                debug_print(f"✅ You selected: {choice}")
                return choice
            else:
                debug_print(f"❌ Invalid choice: '{choice}'. Please try again.")
                debug_print("")
                
        except KeyboardInterrupt:
            debug_print("\n👋 Interrupted by user")
            return None
        except EOFError:
            debug_print("\n❌ Input stream closed")
            return None

def run_debug_tests():
    """Run various debug tests"""
    debug_print("\n🐛 RUNNING DEBUG TESTS...")
    debug_print("=" * 40)
    
    # Test 1: Console capabilities
    debug_print("Test 1: Console capabilities")
    debug_print(f"  Terminal size: {os.get_terminal_size() if hasattr(os, 'get_terminal_size') else 'Unknown'}")
    debug_print(f"  OS: {os.name}")
    debug_print(f"  Platform: {sys.platform}")
    debug_print("")
    
    # Test 2: Unicode support
    debug_print("Test 2: Unicode support")
    try:
        debug_print("  🎯🌡️🔥🌈⚡🎮🔧🎬🚀 - Unicode icons")
        debug_print("  ✅ Unicode working!")
    except:
        debug_print("  ❌ Unicode issues detected")
    debug_print("")
    
    # Test 3: Input/Output
    debug_print("Test 3: Interactive input test")
    try:
        test_input = input("  Type 'test' and press Enter: ")
        debug_print(f"  You typed: '{test_input}'")
        debug_print("  ✅ Input/Output working!")
    except Exception as e:
        debug_print(f"  ❌ Input error: {e}")
    debug_print("")
    
    # Test 4: Buffering
    debug_print("Test 4: Output buffering test")
    for i in range(5):
        debug_print(f"  Line {i+1} (with flush)", 0)
        sys.stdout.flush()
        time.sleep(0.2)
    debug_print("  ✅ Buffering test complete")
    debug_print("")
    
    debug_print("🐛 DEBUG TESTS COMPLETE")
    debug_print("=" * 40)
    input("Press Enter to return to menu...")

def test_imports():
    """Test critical imports"""
    debug_print("🔍 TESTING IMPORTS...")
    debug_print("")
    
    imports_to_test = [
        ('pyrealsense2', 'rs'),
        ('cv2', 'cv2'),
        ('numpy', 'np'),
        ('ultralytics', 'YOLO'),
        ('pygame', 'pygame')
    ]
    
    import_results = {}
    
    for package, module in imports_to_test:
        try:
            if package == 'ultralytics':
                from ultralytics import YOLO
                debug_print(f"  ✅ {package}: Available")
                import_results[package] = True
            else:
                exec(f"import {module}")
                debug_print(f"  ✅ {package}: Available")
                import_results[package] = True
        except ImportError as e:
            debug_print(f"  ❌ {package}: Missing ({e})")
            import_results[package] = False
        except Exception as e:
            debug_print(f"  ⚠️  {package}: Error ({e})")
            import_results[package] = False
    
    debug_print("")
    return import_results

def simple_predator_start():
    """Simple predator vision start without complex menu"""
    debug_print("🚀 STARTING SIMPLE PREDATOR VISION...")
    debug_print("(This bypasses all complex initialization)")
    debug_print("")
    
    try:
        # Import the quick version components
        import pyrealsense2 as rs
        import cv2
        import numpy as np
        
        debug_print("✅ Core imports successful")
        
        # Try to start camera
        debug_print("📷 Testing camera connection...")
        
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            debug_print("❌ No RealSense camera found")
            return False
        
        debug_print(f"✅ Found camera: {devices[0].get_info(rs.camera_info.name)}")
        
        # Simple thermal vision
        debug_print("🎯 Starting basic thermal vision...")
        debug_print("(Press 'q' in the camera window to quit)")
        
        # Configure pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        align = rs.align(rs.stream.color)
        
        debug_print("✅ Camera started - look for window!")
        
        frame_count = 0
        
        try:
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Create simple thermal effect
                depth_meters = depth_image * depth_scale
                depth_clipped = np.clip(depth_meters, 0.5, 4.0)
                depth_inverted = 4.0 - depth_clipped
                depth_normalized = ((depth_inverted / 3.5) * 255).astype(np.uint8)
                
                # Apply colormap
                thermal_image = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
                
                # Simple crosshair
                h, w = thermal_image.shape[:2]
                center_x, center_y = w // 2, h // 2
                cv2.line(thermal_image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
                cv2.line(thermal_image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
                
                # Add simple text
                cv2.putText(thermal_image, "BASIC THERMAL VISION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(thermal_image, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Show
                cv2.imshow('Basic Predator Vision - DEBUG', thermal_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            debug_print("🔄 Interrupted by user")
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            debug_print("✅ Camera stopped")
            
        return True
        
    except Exception as e:
        debug_print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    
    debug_print("🐛 PREDATOR VISION DEBUG TOOL")
    debug_print("=" * 50)
    debug_print("This tool will help diagnose menu display issues")
    debug_print("")
    
    # Test imports first
    import_results = test_imports()
    
    # Show menu
    choice = show_menu_debug()
    
    if choice is None:
        return
    
    if choice == '9':  # Quick start
        success = simple_predator_start()
        if success:
            debug_print("✅ Basic thermal vision test completed successfully!")
        else:
            debug_print("❌ Basic thermal vision test failed")
    
    elif choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
        debug_print(f"🚀 You selected mode {choice}")
        debug_print("This would normally start the full Predator Vision system")
        debug_print("For now, let's test the basic version first...")
        debug_print("")
        
        if input("Start basic thermal vision test? (y/n): ").lower().startswith('y'):
            simple_predator_start()
    
    debug_print("\n🐛 DEBUG SESSION COMPLETE")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
