#!/usr/bin/env python3
"""
Quick Start Predator Vision Script
==================================
Simplified version that starts immediately with default settings
"""

import pyrealsense2 as rs
import cv2
import numpy as np
import time
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO available")
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ YOLO not available - using basic detection")

class QuickPredatorVision:
    def __init__(self):
        print("🎯 Quick Predator Vision Starting...")
        
        # Initialize camera first
        self.initialize_camera()
        
        # Initialize YOLO if available
        if YOLO_AVAILABLE:
            try:
                print("Loading YOLO model...")
                self.model = YOLO('yolov8n-seg.pt')
                self.use_yolo = True
                print("✅ YOLO segmentation loaded")
            except Exception as e:
                print(f"YOLO failed: {e}, using basic detection")
                self.use_yolo = False
        else:
            self.use_yolo = False
        
        # Settings
        self.depth_range = (0.5, 4.0)
        self.thermal_style = 'predator'
        self.confidence_threshold = 0.5
        
        # Create predator colormap
        self.colormap = self.create_predator_colormap()
        
        # Effects
        self.scan_line_pos = 0
        self.scan_direction = 1
        
        print("✅ System ready!")
    
    def initialize_camera(self):
        """Initialize RealSense camera"""
        try:
            print("📷 Connecting to RealSense camera...")
            
            # Check for devices
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                raise RuntimeError("No RealSense camera found!")
            
            print(f"Found RealSense: {devices[0].get_info(rs.camera_info.name)}")
            
            # Configure pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable streams
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Create alignment
            self.align = rs.align(rs.stream.color)
            
            print("✅ Camera connected successfully")
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            print("\nTroubleshooting:")
            print("1. Check RealSense D455 is connected to USB 3.0")
            print("2. Install Intel RealSense SDK")
            print("3. Test with Intel RealSense Viewer")
            raise
    
    def create_predator_colormap(self):
        """Create Predator-style thermal colormap"""
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
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image
    
    def create_thermal_image(self, depth_image):
        """Create thermal visualization from depth"""
        # Convert to meters
        depth_meters = depth_image * self.depth_scale
        
        # Clip and invert (closer = hotter)
        depth_clipped = np.clip(depth_meters, self.depth_range[0], self.depth_range[1])
        depth_inverted = self.depth_range[1] - depth_clipped
        
        # Normalize to 0-255
        depth_normalized = ((depth_inverted - 0) / (self.depth_range[1] - self.depth_range[0]) * 255).astype(np.uint8)
        
        # Apply thermal colormap
        thermal_image = cv2.applyColorMap(depth_normalized, self.colormap)
        
        return thermal_image
    
    def detect_objects(self, color_image):
        """Detect objects using available method"""
        if self.use_yolo:
            try:
                results = self.model(color_image, conf=self.confidence_threshold)
                return results[0] if results else None
            except:
                return None
        else:
            # Basic edge detection fallback
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area
            valid_contours = [c for c in contours if cv2.contourArea(c) > 1000]
            return valid_contours
    
    def draw_segmentation(self, thermal_image, detections, color_image):
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
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Glow effect
                for thickness in range(8, 2, -1):
                    alpha = 0.3 * (8 - thickness) / 6
                    glow_img = result.copy()
                    cv2.drawContours(glow_img, contours, -1, (0, 255, 0), thickness)
                    cv2.addWeighted(result, 1 - alpha, glow_img, alpha, 0, result)
                
                # Main outline
                cv2.drawContours(result, contours, -1, (0, 255, 255), 2)
                
                # Label
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        elif isinstance(detections, list):
            # Contour detection
            for i, contour in enumerate(detections):
                # Glow effect
                for thickness in range(6, 1, -1):
                    alpha = 0.2 * (6 - thickness) / 5
                    glow_img = result.copy()
                    cv2.drawContours(glow_img, [contour], -1, (0, 255, 0), thickness)
                    cv2.addWeighted(result, 1 - alpha, glow_img, alpha, 0, result)
                
                # Main outline
                cv2.drawContours(result, [contour], -1, (0, 255, 255), 2)
                
                # Label
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(result, f"Target_{i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return result
    
    def add_predator_effects(self, image):
        """Add Predator-style visual effects"""
        result = image.copy()
        height, width = result.shape[:2]
        
        # Scanning line
        scan_y = int(self.scan_line_pos)
        cv2.line(result, (0, scan_y), (width, scan_y), (0, 255, 255), 2)
        
        # Update scan position
        self.scan_line_pos += self.scan_direction * 3
        if self.scan_line_pos >= height or self.scan_line_pos <= 0:
            self.scan_direction *= -1
        
        # Crosshair
        center_x, center_y = width // 2, height // 2
        reticle_size = 30
        
        cv2.line(result, (center_x - reticle_size, center_y), (center_x + reticle_size, center_y), (0, 255, 0), 2)
        cv2.line(result, (center_x, center_y - reticle_size), (center_x, center_y + reticle_size), (0, 255, 0), 2)
        
        # Corner brackets
        bracket_size = 15
        corners = [
            (center_x - reticle_size, center_y - reticle_size),
            (center_x + reticle_size, center_y - reticle_size),
            (center_x - reticle_size, center_y + reticle_size),
            (center_x + reticle_size, center_y + reticle_size)
        ]
        
        for corner in corners:
            cv2.line(result, corner, (corner[0] + bracket_size * (1 if corner[0] < center_x else -1), corner[1]), (0, 255, 0), 2)
            cv2.line(result, corner, (corner[0], corner[1] + bracket_size * (1 if corner[1] < center_y else -1)), (0, 255, 0), 2)
        
        return result
    
    def add_hud(self, image, fps=0):
        """Add HUD overlay"""
        # System info
        info_lines = [
            "PREDATOR VISION ACTIVE",
            f"THERMAL: {self.thermal_style.upper()}",
            f"FPS: {fps:.1f}",
            f"DEPTH: {self.depth_range[0]:.1f}-{self.depth_range[1]:.1f}M",
            "STATUS: HUNTING"
        ]
        
        # Draw info panel
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.rectangle(image, (10, y_pos - 20), (300, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(image, line, (15, y_pos), font, 0.6, (0, 255, 0), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(image, timestamp, (image.shape[1] - 100, image.shape[0] - 20), font, 0.5, (0, 255, 0), 1)
        
        return image
    
    def run(self):
        """Main processing loop"""
        print("\n🎯 PREDATOR VISION ACTIVATED")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save frame")
        print("  'p' - Pause/Resume")
        print("  '+/-' - Adjust depth range")
        print("=" * 50)
        
        frame_count = 0
        start_time = time.time()
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
                    result = self.draw_segmentation(thermal_image, detections, color_image)
                    
                    # Add effects
                    result = self.add_predator_effects(result)
                    
                    # Add HUD
                    fps = frame_count / (time.time() - start_time) if frame_count > 0 else 0
                    result = self.add_hud(result, fps)
                    
                    # Display
                    cv2.imshow('🎯 PREDATOR VISION', result)
                    
                    frame_count += 1
                
                # Handle keys
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
                    
        except KeyboardInterrupt:
            print("\n🎯 System terminated by user")
        finally:
            # Cleanup
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
            # Stats
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"\n📊 Session complete:")
            print(f"   Runtime: {total_time:.1f}s")
            print(f"   Frames: {frame_count}")
            print(f"   Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    try:
        print("🎯 PREDATOR VISION - Quick Start")
        print("================================")
        
        # Create and run system
        system = QuickPredatorVision()
        system.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure RealSense D455 is connected")
        print("2. Install: pip install ultralytics pyrealsense2 opencv-python")
        print("3. Test camera with Intel RealSense Viewer")
        input("Press Enter to exit...")
