#!/usr/bin/env python3
"""
Fixed Predator Vision System
============================
This version initializes step-by-step with progress feedback to avoid hangs
"""

import pyrealsense2 as rs
import cv2
import numpy as np
import time
import json
import sys
import os
import platform
from datetime import datetime
from pathlib import Path

# Test optional imports with feedback
print("🔍 Loading optional features...")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("  ✅ YOLO available")
except ImportError:
    YOLO_AVAILABLE = False
    print("  ⚠️  YOLO not available")

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
    print("  ✅ SciPy available")
except ImportError:
    SCIPY_AVAILABLE = False
    print("  ⚠️  SciPy not available")

try:
    import pygame
    # Don't initialize pygame mixer here - do it later when needed
    PYGAME_AVAILABLE = True
    print("  ✅ PyGame available")
except ImportError:
    PYGAME_AVAILABLE = False
    print("  ⚠️  PyGame not available")

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available() if 'torch' in locals() else False
    if CUDA_AVAILABLE:
        print("  🚀 CUDA GPU available")
    else:
        print("  💻 CPU only")
except ImportError:
    CUDA_AVAILABLE = False
    print("  💻 CPU only")

try:
    import tkinter as tk
    from tkinter import ttk
    TKINTER_AVAILABLE = True
    print("  ✅ tkinter available")
except ImportError:
    TKINTER_AVAILABLE = False
    print("  ⚠️  tkinter not available")

print("✅ Optional imports complete\n")

class FixedPredatorVision:
    """Fixed Predator Vision with step-by-step initialization"""
    
    def __init__(self, 
                 thermal_style='predator',
                 segmentation_method='yolo_seg',
                 depth_range=(0.5, 5.0),
                 enable_gpu=True,
                 confidence_threshold=0.5):
        
        print("🎯 Initializing Predator Vision System...")
        
        # Basic settings
        self.thermal_style = thermal_style
        self.segmentation_method = segmentation_method
        self.depth_range = depth_range
        self.confidence_threshold = confidence_threshold
        self.enable_gpu = enable_gpu and CUDA_AVAILABLE
        
        print(f"  Mode: {thermal_style}")
        print(f"  Segmentation: {segmentation_method}")
        print(f"  Depth range: {depth_range[0]:.1f}-{depth_range[1]:.1f}m")
        print(f"  GPU: {'Enabled' if self.enable_gpu else 'Disabled'}")
        
        # Initialize components step by step
        self.initialize_camera()
        self.initialize_models()
        self.setup_thermal_visualization()
        self.setup_effects()
        
        print("✅ Predator Vision System ready!")
    
    def initialize_camera(self):
        """Initialize RealSense camera with progress feedback"""
        print("\n📷 Initializing camera...")
        
        try:
            # Check for devices
            print("  Scanning for devices...")
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found!")
            
            device_name = devices[0].get_info(rs.camera_info.name)
            print(f"  Found: {device_name}")
            
            # Configure pipeline
            print("  Configuring pipeline...")
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable streams
            print("  Enabling color stream...")
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            print("  Enabling depth stream...")
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            print("  Starting pipeline...")
            profile = self.pipeline.start(self.config)
            
            # Get depth scale
            print("  Getting depth sensor settings...")
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Create alignment
            print("  Setting up frame alignment...")
            self.align = rs.align(rs.stream.color)
            
            print("  ✅ Camera initialized successfully")
            
        except Exception as e:
            print(f"  ❌ Camera initialization failed: {e}")
            raise
    
    def initialize_models(self):
        """Initialize AI models with progress feedback"""
        print("\n🤖 Loading AI models...")
        
        if self.segmentation_method == 'yolo_seg' and YOLO_AVAILABLE:
            try:
                print("  Loading YOLO segmentation model...")
                self.model = YOLO('yolov8n-seg.pt')
                
                if self.enable_gpu:
                    print("  Moving model to GPU...")
                    self.model.to('cuda')
                
                self.use_yolo = True
                print("  ✅ YOLO model loaded")
                
            except Exception as e:
                print(f"  ⚠️  YOLO loading failed: {e}")
                print("  Falling back to basic detection...")
                self.use_yolo = False
                self.segmentation_method = 'contour'
        else:
            print(f"  Using {self.segmentation_method} method")
            self.use_yolo = False
        
        # Initialize background subtractor for motion detection
        if self.segmentation_method == 'motion':
            print("  Setting up background subtractor...")
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        
        print("  ✅ Models initialized")
    
    def setup_thermal_visualization(self):
        """Setup thermal colormap with progress feedback"""
        print("\n🌡️  Setting up thermal visualization...")
        
        print(f"  Creating {self.thermal_style} colormap...")
        
        if self.thermal_style == 'predator':
            self.colormap = self.create_predator_colormap()
        else:
            colormap_dict = {
                'flir': cv2.COLORMAP_INFERNO,
                'iron': cv2.COLORMAP_HOT,
                'rainbow': cv2.COLORMAP_RAINBOW,
                'hot': cv2.COLORMAP_HOT,
                'plasma': cv2.COLORMAP_PLASMA,
                'viridis': cv2.COLORMAP_VIRIDIS
            }
            self.colormap = colormap_dict.get(self.thermal_style, cv2.COLORMAP_INFERNO)
        
        print("  ✅ Thermal visualization ready")
    
    def setup_effects(self):
        """Setup visual effects"""
        print("\n✨ Setting up visual effects...")
        
        # Effect settings
        self.scan_line_pos = 0
        self.scan_direction = 1
        self.thermal_noise_intensity = 0.1
        self.outline_glow = True
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []
        
        # Object tracking
        self.tracked_objects = {}
        self.next_track_id = 0
        
        print("  ✅ Effects initialized")
    
    def create_predator_colormap(self):
        """Create custom Predator-style thermal colormap"""
        colors = np.array([
            [0, 0, 0],        # Black (cold)
            [0, 0, 128],      # Dark blue
            [0, 128, 255],    # Blue
            [0, 255, 255],    # Cyan
            [0, 255, 0],      # Green
            [255, 255, 0],    # Yellow
            [255, 128, 0],    # Orange
            [255, 0, 0],      # Red
            [255, 255, 255]   # White (hot)
        ], dtype=np.uint8)
        
        # Create 256-color palette
        indices = np.linspace(0, len(colors) - 1, 256)
        colormap = np.zeros((256, 1, 3), dtype=np.uint8)
        
        for i in range(256):
            idx = indices[i]
            lower_idx = int(np.floor(idx))
            upper_idx = min(int(np.ceil(idx)), len(colors) - 1)
            alpha = idx - lower_idx
            
            color = (1 - alpha) * colors[lower_idx] + alpha * colors[upper_idx]
            colormap[i, 0] = color.astype(np.uint8)
        
        return colormap
    
    def get_frame(self):
        """Get camera frame"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"Frame error: {e}")
            return None, None
    
    def create_thermal_image(self, depth_image):
        """Create thermal visualization from depth"""
        # Convert to meters
        depth_meters = depth_image * self.depth_scale
        
        # Smooth depth data
        if SCIPY_AVAILABLE:
            depth_smooth = gaussian_filter(depth_meters, sigma=1.0)
        else:
            depth_smooth = cv2.GaussianBlur(depth_meters, (5, 5), 1.0)
        
        # Clip and invert (closer = hotter)
        depth_clipped = np.clip(depth_smooth, self.depth_range[0], self.depth_range[1])
        depth_inverted = self.depth_range[1] - depth_clipped
        
        # Normalize to 0-255
        depth_normalized = ((depth_inverted - 0) / (self.depth_range[1] - self.depth_range[0]) * 255)
        
        # Add thermal noise
        if self.thermal_noise_intensity > 0:
            noise = np.random.normal(0, self.thermal_noise_intensity * 255, depth_normalized.shape)
            depth_normalized = np.clip(depth_normalized + noise, 0, 255)
        
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Apply thermal colormap
        if self.thermal_style == 'predator':
            thermal_image = cv2.applyColorMap(depth_normalized, self.colormap)
        else:
            thermal_image = cv2.applyColorMap(depth_normalized, self.colormap)
        
        return thermal_image
    
    def detect_objects(self, color_image):
        """Detect objects using available method"""
        if self.use_yolo:
            try:
                results = self.model(color_image, conf=self.confidence_threshold)
                return results[0] if results else None
            except Exception as e:
                print(f"YOLO detection error: {e}")
                return None
        
        elif self.segmentation_method == 'motion':
            # Motion detection
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            fg_mask = self.bg_subtractor.apply(gray)
            
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area
            valid_contours = [c for c in contours if cv2.contourArea(c) > 800]
            return valid_contours
        
        else:
            # Edge detection fallback
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area
            valid_contours = [c for c in contours if cv2.contourArea(c) > 1000]
            return valid_contours
    
    def draw_segmentation(self, thermal_image, detections):
        """Draw object outlines on thermal image"""
        result = thermal_image.copy()
        
        if self.use_yolo and detections and hasattr(detections, 'masks') and detections.masks is not None:
            # YOLO segmentation
            masks = detections.masks.data.cpu().numpy()
            boxes = detections.boxes
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # Resize mask
                mask_resized = cv2.resize(mask, (thermal_image.shape[1], thermal_image.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Get class info
                class_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                class_name = self.model.names[class_id]
                
                # Draw glowing outline
                self.draw_glowing_outline(result, mask_binary, class_name, confidence)
        
        elif isinstance(detections, list) and detections:
            # Contour detection
            for i, contour in enumerate(detections):
                # Create mask from contour
                mask = np.zeros(thermal_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                
                # Draw outline
                self.draw_glowing_outline(result, mask, f"Target_{i}", 0.8)
        
        return result
    
    def draw_glowing_outline(self, image, mask, label, confidence):
        """Draw glowing outline effect"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return
        
        # Predator-style colors
        outline_color = (0, 255, 255)  # Bright cyan
        glow_color = (0, 255, 0)       # Bright green
        
        for contour in contours:
            if self.outline_glow:
                # Draw glow effect
                for thickness in range(8, 2, -1):
                    alpha = 0.3 * (8 - thickness) / 6
                    glow_img = image.copy()
                    cv2.drawContours(glow_img, [contour], -1, glow_color, thickness)
                    cv2.addWeighted(image, 1 - alpha, glow_img, alpha, 0, image)
            
            # Main outline
            cv2.drawContours(image, [contour], -1, outline_color, 2)
            
            # Label
            x, y, w, h = cv2.boundingRect(contour)
            label_text = f"{label}: {confidence:.2f}"
            
            # Label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), (0, 0, 0), -1)
            cv2.rectangle(image, (x - 2, y - text_h - 12), (x + text_w + 2, y + 2), outline_color, 1)
            
            # Label text
            cv2.putText(image, label_text, (x, y - 5), font, font_scale, outline_color, thickness)
    
    def add_predator_effects(self, image):
        """Add Predator-style visual effects"""
        result = image.copy()
        height, width = result.shape[:2]
        
        # Scanning line
        scan_y = int(self.scan_line_pos)
        scan_color = (0, 255, 255)
        
        cv2.line(result, (0, scan_y), (width, scan_y), scan_color, 2)
        cv2.line(result, (0, scan_y - 1), (width, scan_y - 1), scan_color, 1)
        cv2.line(result, (0, scan_y + 1), (width, scan_y + 1), scan_color, 1)
        
        # Update scan position
        self.scan_line_pos += self.scan_direction * 3
        if self.scan_line_pos >= height or self.scan_line_pos <= 0:
            self.scan_direction *= -1
        
        # Crosshair
        center_x, center_y = width // 2, height // 2
        reticle_size = 30
        reticle_color = (0, 255, 0)
        
        cv2.line(result, (center_x - reticle_size, center_y), (center_x + reticle_size, center_y), reticle_color, 2)
        cv2.line(result, (center_x, center_y - reticle_size), (center_x, center_y + reticle_size), reticle_color, 2)
        
        # Corner brackets
        bracket_size = 15
        corners = [
            (center_x - reticle_size, center_y - reticle_size),
            (center_x + reticle_size, center_y - reticle_size),
            (center_x - reticle_size, center_y + reticle_size),
            (center_x + reticle_size, center_y + reticle_size)
        ]
        
        for corner in corners:
            cv2.line(result, corner, (corner[0] + bracket_size * (1 if corner[0] < center_x else -1), corner[1]), reticle_color, 2)
            cv2.line(result, corner, (corner[0], corner[1] + bracket_size * (1 if corner[1] < center_y else -1)), reticle_color, 2)
        
        return result
    
    def add_hud(self, image):
        """Add HUD overlay"""
        # Performance metrics
        current_time = time.time()
        runtime = current_time - self.start_time
        fps = self.frame_count / runtime if runtime > 0 else 0
        
        # Update FPS history
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # HUD info
        info_lines = [
            "PREDATOR VISION ACTIVE",
            f"MODE: {self.thermal_style.upper()}",
            f"DETECTION: {self.segmentation_method.upper()}",
            f"FPS: {avg_fps:.1f}",
            f"DEPTH: {self.depth_range[0]:.1f}-{self.depth_range[1]:.1f}M",
            f"FRAME: {self.frame_count:06d}"
        ]
        
        # Draw HUD panel with smaller text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # Reduced from 0.6 - make this smaller for tinier text
        thickness = 1     # Reduced from 2 for thinner text
        hud_color = (0, 255, 0)
        
        # Adjust panel size for smaller text
        panel_width = 240   # Reduced from 300
        panel_height = len(info_lines) * 18 + 15  # Reduced spacing from 25 to 18
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
        
        # Panel border
        cv2.rectangle(image, (10, 10), (panel_width, panel_height), hud_color, 1)  # Thinner border
        
        # Draw info lines with smaller spacing
        for i, line in enumerate(info_lines):
            y_pos = 25 + i * 18  # Reduced from 35 + i * 25
            cv2.putText(image, line, (15, y_pos), font, font_scale, hud_color, thickness)
        
        # Smaller timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(image, timestamp, (image.shape[1] - 80, image.shape[0] - 15), 
                   font, 0.4, hud_color, 1)  # Smaller timestamp
        
        return image
    
    def run(self, save_video=False, video_filename='predator_vision.avi'):
        """Main processing loop"""
        print(f"\n🎯 PREDATOR VISION ACTIVATED")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save frame")
        print("  'p' - Pause/Resume")
        print("  '+/-' - Adjust depth range")
        print("  'g' - Toggle glow effects")
        print("  'n' - Toggle thermal noise")
        print("=" * 50)
        
        # Video writer
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))
            print(f"📹 Recording to: {video_filename}")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    # Get frame
                    color_image, depth_image = self.get_frame()
                    
                    if color_image is None:
                        continue
                    
                    # Create thermal image
                    thermal_image = self.create_thermal_image(depth_image)
                    
                    # Detect objects
                    detections = self.detect_objects(color_image)
                    
                    # Draw segmentation
                    result = self.draw_segmentation(thermal_image, detections)
                    
                    # Add effects
                    result = self.add_predator_effects(result)
                    
                    # Add HUD
                    result = self.add_hud(result)
                    
                    # Display
                    cv2.imshow('🎯 PREDATOR VISION', result)
                    
                    # Save video
                    if save_video and video_writer:
                        video_writer.write(result)
                    
                    self.frame_count += 1
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"predator_capture_{timestamp}.jpg"
                    cv2.imwrite(filename, result)
                    print(f"🎯 Frame saved: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    print("⏸️ Paused" if paused else "▶️ Resumed")
                elif key == ord('+') or key == ord('='):
                    self.depth_range = (self.depth_range[0], min(15.0, self.depth_range[1] + 0.5))
                    print(f"📏 Depth range: {self.depth_range[0]:.1f}-{self.depth_range[1]:.1f}m")
                elif key == ord('-'):
                    self.depth_range = (self.depth_range[0], max(1.0, self.depth_range[1] - 0.5))
                    print(f"📏 Depth range: {self.depth_range[0]:.1f}-{self.depth_range[1]:.1f}m")
                elif key == ord('g'):
                    self.outline_glow = not self.outline_glow
                    print(f"🌟 Glow effects: {'ON' if self.outline_glow else 'OFF'}")
                elif key == ord('n'):
                    self.thermal_noise_intensity = 0.2 if self.thermal_noise_intensity == 0 else 0
                    print(f"📡 Thermal noise: {'ON' if self.thermal_noise_intensity > 0 else 'OFF'}")
                    
        except KeyboardInterrupt:
            print("\n🎯 System terminated by user")
        finally:
            # Cleanup
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
            if video_writer:
                video_writer.release()
                print(f"📹 Video saved: {video_filename}")
            
            # Stats
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            print(f"\n📊 Session Statistics:")
            print(f"   Runtime: {total_time:.1f}s")
            print(f"   Frames: {self.frame_count}")
            print(f"   Average FPS: {avg_fps:.1f}")
            print("🎯 PREDATOR VISION DEACTIVATED")

def clear_and_menu():
    """Display menu with proper clearing"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🎯 PREDATOR VISION SYSTEM - Fixed Version")
    print("=" * 60)
    print()
    
    menu_items = [
        "1. 🎯 Standard Predator Vision (YOLO + Thermal)",
        "2. 🌡️  FLIR Thermal Camera Style", 
        "3. 🔥 Military Iron Thermal Scope",
        "4. 🌈 Scientific Rainbow Thermal", 
        "5. ⚡ High Performance Mode",
        "6. 🎮 Motion Detection Mode",
        "7. 🔧 Edge Detection Mode",
        "8. 🚀 Quick Start (Basic Mode)"
    ]
    
    for item in menu_items:
        print(item)
        time.sleep(0.02)
    
    print()
    print("=" * 60)
    print()
    
    return input("👆 Enter your choice (1-8, or 'q' to quit): ").strip()

def main():
    """Main function"""
    try:
        choice = clear_and_menu()
        
        if choice.lower() in ['q', 'quit', 'exit']:
            print("👋 Goodbye!")
            return
        
        # Configuration mapping
        configs = {
            '1': {'thermal_style': 'predator', 'segmentation_method': 'yolo_seg', 'filename': 'predator_standard.avi'},
            '2': {'thermal_style': 'flir', 'segmentation_method': 'yolo_seg', 'filename': 'flir_thermal.avi'},
            '3': {'thermal_style': 'iron', 'segmentation_method': 'motion', 'filename': 'military_iron.avi'},
            '4': {'thermal_style': 'rainbow', 'segmentation_method': 'yolo_seg', 'filename': 'scientific_rainbow.avi'},
            '5': {'thermal_style': 'predator', 'segmentation_method': 'yolo_seg', 'filename': 'high_performance.avi'},
            '6': {'thermal_style': 'predator', 'segmentation_method': 'motion', 'filename': 'motion_detection.avi'},
            '7': {'thermal_style': 'predator', 'segmentation_method': 'contour', 'filename': 'edge_detection.avi'},
            '8': {'thermal_style': 'predator', 'segmentation_method': 'contour', 'filename': 'quick_start.avi'}
        }
        
        if choice in configs:
            config = configs[choice]
            
            print(f"\n🚀 Initializing mode {choice}...")
            
            # Create system with step-by-step initialization
            system = FixedPredatorVision(
                thermal_style=config['thermal_style'],
                segmentation_method=config['segmentation_method'],
                enable_gpu=CUDA_AVAILABLE
            )
            
            # Run system
            system.run(save_video=True, video_filename=config['filename'])
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
