# Predator Vision - Thermal Depth Segmentation System

## Overview
This system creates a "Predator vision" effect by combining RealSense depth information with object segmentation to produce thermal/FLIR camera-style visualization. Objects are outlined with precise segmentation masks while the entire scene is rendered with depth-based thermal coloring.

## Prerequisites

```bash
# Core dependencies
pip install ultralytics
pip install pyrealsense2
pip install opencv-python
pip install numpy
pip install matplotlib

# Optional for advanced effects
pip install scipy
pip install scikit-image
```

## Complete Predator Vision Implementation

```python
import pyrealsense2 as rs
import cv2
import numpy as np
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage
import threading

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available - using basic segmentation")

class PredatorVisionSystem:
    def __init__(self, 
                 segmentation_method='yolo_seg',
                 confidence_threshold=0.5,
                 thermal_style='predator',
                 depth_range=(0.5, 5.0)):
        """
        Initialize Predator Vision System
        
        Args:
            segmentation_method: 'yolo_seg', 'contour', 'motion'
            confidence_threshold: Detection confidence threshold
            thermal_style: 'predator', 'flir', 'iron', 'rainbow', 'hot'
            depth_range: (min_depth, max_depth) in meters for thermal mapping
        """
        self.segmentation_method = segmentation_method
        self.confidence_threshold = confidence_threshold
        self.thermal_style = thermal_style
        self.depth_range = depth_range
        
        # Initialize models
        self.initialize_models()
        
        # Initialize camera
        self.initialize_camera()
        
        # Thermal visualization settings
        self.setup_thermal_visualization()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Visual effects settings
        self.scan_line_pos = 0
        self.scan_direction = 1
        self.thermal_noise_intensity = 0.1
        self.outline_thickness = 2
        self.outline_glow = True
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        
        # Audio effects (optional)
        self.audio_enabled = False
        
    def initialize_models(self):
        """Initialize segmentation models"""
        if self.segmentation_method == 'yolo_seg' and YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n-seg.pt')
                print("✓ YOLO segmentation model loaded")
            except Exception as e:
                print(f"Failed to load YOLO: {e}, falling back to contour method")
                self.segmentation_method = 'contour'
        else:
            print(f"Using {self.segmentation_method} method")
    
    def initialize_camera(self):
        """Initialize RealSense camera"""
        try:
            # Configure RealSense pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable color and depth streams
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Get depth sensor for depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Create alignment object
            self.align = rs.align(rs.stream.color)
            
            # Get camera intrinsics
            color_profile = profile.get_stream(rs.stream.color)
            self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            print("✓ RealSense camera initialized")
            
        except Exception as e:
            print(f"Failed to initialize RealSense: {e}")
            raise RuntimeError("RealSense camera required for Predator Vision")
    
    def setup_thermal_visualization(self):
        """Setup thermal colormap and effects"""
        self.thermal_colormaps = {
            'predator': self.create_predator_colormap(),
            'flir': cv2.COLORMAP_INFERNO,
            'iron': cv2.COLORMAP_HOT,
            'rainbow': cv2.COLORMAP_RAINBOW,
            'hot': cv2.COLORMAP_HOT,
            'plasma': cv2.COLORMAP_PLASMA,
            'viridis': cv2.COLORMAP_VIRIDIS
        }
        
        # Create custom predator-style colormap
        if self.thermal_style == 'predator':
            self.colormap = self.thermal_colormaps['predator']
        else:
            self.colormap = self.thermal_colormaps.get(self.thermal_style, cv2.COLORMAP_INFERNO)
    
    def create_predator_colormap(self):
        """Create custom Predator-style colormap"""
        # Create a custom colormap with the iconic Predator thermal colors
        colors = np.array([
            [0, 0, 0],        # Black (cold/far)
            [0, 0, 128],      # Dark blue
            [0, 128, 255],    # Blue
            [0, 255, 255],    # Cyan
            [0, 255, 0],      # Green
            [255, 255, 0],    # Yellow
            [255, 128, 0],    # Orange
            [255, 0, 0],      # Red (hot/close)
            [255, 255, 255]   # White (very hot/very close)
        ], dtype=np.uint8)
        
        # Interpolate to create 256 color palette
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
        """Get aligned color and depth frames"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image, depth_frame
            
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None, None, None
    
    def create_thermal_depth_image(self, depth_image):
        """Convert depth image to thermal-style visualization"""
        # Convert depth to meters
        depth_meters = depth_image * self.depth_scale
        
        # Clip to depth range and normalize
        depth_clipped = np.clip(depth_meters, self.depth_range[0], self.depth_range[1])
        
        # Invert depth (closer = hotter/brighter)
        depth_inverted = self.depth_range[1] - depth_clipped
        
        # Normalize to 0-255 range
        depth_normalized = ((depth_inverted - 0) / (self.depth_range[1] - self.depth_range[0]) * 255).astype(np.uint8)
        
        # Add thermal noise for realism
        if self.thermal_noise_intensity > 0:
            noise = np.random.normal(0, self.thermal_noise_intensity * 255, depth_normalized.shape)
            depth_normalized = np.clip(depth_normalized + noise, 0, 255).astype(np.uint8)
        
        # Apply thermal colormap
        if self.thermal_style == 'predator':
            # Apply custom predator colormap
            thermal_image = cv2.applyColorMap(depth_normalized, self.colormap)
        else:
            # Apply OpenCV colormap
            thermal_image = cv2.applyColorMap(depth_normalized, self.colormap)
        
        return thermal_image, depth_normalized
    
    def segment_objects(self, color_image):
        """Perform object segmentation"""
        if self.segmentation_method == 'yolo_seg' and hasattr(self, 'model'):
            return self.segment_yolo(color_image)
        elif self.segmentation_method == 'motion':
            return self.segment_motion(color_image)
        else:
            return self.segment_contour(color_image)
    
    def segment_yolo(self, image):
        """YOLO segmentation"""
        results = self.model(image, conf=self.confidence_threshold)
        return results[0] if results else None
    
    def segment_motion(self, image):
        """Motion-based segmentation"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(image)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area
                valid_contours.append(contour)
        
        return valid_contours
    
    def segment_contour(self, image):
        """Edge-based contour segmentation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        valid_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                valid_contours.append(contour)
        
        return valid_contours
    
    def draw_segmentation_outlines(self, thermal_image, segmentations, color_image=None):
        """Draw segmentation outlines on thermal image"""
        result = thermal_image.copy()
        
        if self.segmentation_method == 'yolo_seg' and segmentations and segmentations.masks is not None:
            # YOLO format
            masks = segmentations.masks.data.cpu().numpy()
            boxes = segmentations.boxes
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (thermal_image.shape[1], thermal_image.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Get class info
                class_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                class_name = self.model.names[class_id]
                
                # Draw outline with glow effect
                self.draw_glowing_outline(result, mask_binary, class_name, confidence)
        
        elif isinstance(segmentations, list):
            # Contour format
            for i, contour in enumerate(segmentations):
                # Create mask from contour
                mask = np.zeros(thermal_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                
                # Draw outline with glow effect
                self.draw_glowing_outline(result, mask, f"Object_{i}", 0.8)
        
        return result
    
    def draw_glowing_outline(self, image, mask, label, confidence):
        """Draw glowing outline effect"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return
        
        # Predator-style colors (bright green/yellow for outlines)
        outline_color = (0, 255, 255)  # Bright cyan/yellow
        glow_color = (0, 255, 0)       # Bright green
        
        for contour in contours:
            if self.outline_glow:
                # Draw glow effect (multiple thick outlines)
                for thickness in range(8, 2, -1):
                    alpha = 0.3 * (8 - thickness) / 6
                    glow_img = image.copy()
                    cv2.drawContours(glow_img, [contour], -1, glow_color, thickness)
                    cv2.addWeighted(image, 1 - alpha, glow_img, alpha, 0, image)
            
            # Draw main outline
            cv2.drawContours(image, [contour], -1, outline_color, self.outline_thickness)
            
            # Add label with glow effect
            x, y, w, h = cv2.boundingRect(contour)
            label_text = f"{label}: {confidence:.2f}"
            
            # Label background with glow
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), (0, 0, 0), -1)
            cv2.rectangle(image, (x - 2, y - text_h - 12), (x + text_w + 2, y + 2), outline_color, 1)
            
            # Draw label text
            cv2.putText(image, label_text, (x, y - 5), font, font_scale, outline_color, thickness)
    
    def add_predator_effects(self, image):
        """Add Predator-style visual effects"""
        result = image.copy()
        height, width = result.shape[:2]
        
        # Scanning line effect
        scan_y = int(self.scan_line_pos)
        
        # Draw scanning line
        scan_color = (0, 255, 255)  # Bright cyan
        cv2.line(result, (0, scan_y), (width, scan_y), scan_color, 2)
        cv2.line(result, (0, scan_y - 1), (width, scan_y - 1), scan_color, 1)
        cv2.line(result, (0, scan_y + 1), (width, scan_y + 1), scan_color, 1)
        
        # Update scan line position
        self.scan_line_pos += self.scan_direction * 3
        if self.scan_line_pos >= height or self.scan_line_pos <= 0:
            self.scan_direction *= -1
        
        # Add crosshair/targeting reticle
        center_x, center_y = width // 2, height // 2
        reticle_size = 30
        reticle_color = (0, 255, 0)  # Bright green
        
        # Crosshair
        cv2.line(result, (center_x - reticle_size, center_y), (center_x + reticle_size, center_y), reticle_color, 2)
        cv2.line(result, (center_x, center_y - reticle_size), (center_x, center_y + reticle_size), reticle_color, 2)
        
        # Corner brackets
        bracket_size = 15
        cv2.line(result, (center_x - reticle_size, center_y - reticle_size), 
                (center_x - reticle_size + bracket_size, center_y - reticle_size), reticle_color, 2)
        cv2.line(result, (center_x - reticle_size, center_y - reticle_size), 
                (center_x - reticle_size, center_y - reticle_size + bracket_size), reticle_color, 2)
        
        cv2.line(result, (center_x + reticle_size, center_y - reticle_size), 
                (center_x + reticle_size - bracket_size, center_y - reticle_size), reticle_color, 2)
        cv2.line(result, (center_x + reticle_size, center_y - reticle_size), 
                (center_x + reticle_size, center_y - reticle_size + bracket_size), reticle_color, 2)
        
        cv2.line(result, (center_x - reticle_size, center_y + reticle_size), 
                (center_x - reticle_size + bracket_size, center_y + reticle_size), reticle_color, 2)
        cv2.line(result, (center_x - reticle_size, center_y + reticle_size), 
                (center_x - reticle_size, center_y + reticle_size - bracket_size), reticle_color, 2)
        
        cv2.line(result, (center_x + reticle_size, center_y + reticle_size), 
                (center_x + reticle_size - bracket_size, center_y + reticle_size), reticle_color, 2)
        cv2.line(result, (center_x + reticle_size, center_y + reticle_size), 
                (center_x + reticle_size, center_y + reticle_size - bracket_size), reticle_color, 2)
        
        return result
    
    def add_hud_overlay(self, image):
        """Add HUD (Heads-Up Display) elements"""
        height, width = image.shape[:2]
        
        # Performance metrics
        current_time = time.time()
        fps = self.frame_count / (current_time - self.start_time) if self.frame_count > 0 else 0
        
        # HUD color
        hud_color = (0, 255, 0)  # Bright green
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # System info
        info_lines = [
            f"THERMAL VISION ACTIVE",
            f"MODE: {self.thermal_style.upper()}",
            f"DEPTH RANGE: {self.depth_range[0]:.1f}-{self.depth_range[1]:.1f}m",
            f"FPS: {fps:.1f}",
            f"FRAME: {self.frame_count:06d}",
            f"TARGETS: SCANNING..."
        ]
        
        # Draw HUD background
        hud_height = len(info_lines) * 25 + 20
        cv2.rectangle(image, (10, 10), (300, hud_height), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (300, hud_height), hud_color, 2)
        
        # Draw info lines
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 25
            cv2.putText(image, line, (20, y_pos), font, font_scale, hud_color, thickness)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(image, timestamp, (width - 200, height - 20), font, 0.5, hud_color, 1)
        
        # Add distance readout at crosshair
        center_x, center_y = width // 2, height // 2
        # This would show distance to center point if depth frame available
        cv2.putText(image, "DIST: --.-m", (center_x + 40, center_y), font, 0.5, hud_color, 1)
        
        return image
    
    def process_frame(self, color_image, depth_image, depth_frame):
        """Process a single frame with all effects"""
        # Create thermal depth visualization
        thermal_image, depth_normalized = self.create_thermal_depth_image(depth_image)
        
        # Perform object segmentation
        segmentations = self.segment_objects(color_image)
        
        # Draw segmentation outlines
        result = self.draw_segmentation_outlines(thermal_image, segmentations, color_image)
        
        # Add Predator-style effects
        result = self.add_predator_effects(result)
        
        # Add HUD overlay
        result = self.add_hud_overlay(result)
        
        return result
    
    def start_predator_vision(self, save_video=False, video_filename='predator_vision.avi'):
        """Start Predator Vision system"""
        print("🎯 PREDATOR VISION SYSTEM ACTIVATED")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'p' - Pause/Resume")
        print("  '1-7' - Switch thermal style")
        print("  '+/-' - Adjust depth range")
        print("  'g' - Toggle glow effect")
        print("  'n' - Toggle thermal noise")
        print("=" * 50)
        
        # Video writer setup
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))
        
        paused = False
        
        try:
            while True:
                if not paused:
                    # Get frame
                    color_image, depth_image, depth_frame = self.get_frame()
                    
                    if color_image is None or depth_image is None:
                        continue
                    
                    # Process frame
                    result = self.process_frame(color_image, depth_image, depth_frame)
                    
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
                    filename = f"predator_vision_{timestamp}.jpg"
                    cv2.imwrite(filename, result)
                    print(f"🎯 Target acquired and saved: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    print("🔄 PAUSED" if paused else "▶️  RESUMED")
                elif key == ord('1'):
                    self.switch_thermal_style('predator')
                elif key == ord('2'):
                    self.switch_thermal_style('flir')
                elif key == ord('3'):
                    self.switch_thermal_style('iron')
                elif key == ord('4'):
                    self.switch_thermal_style('rainbow')
                elif key == ord('5'):
                    self.switch_thermal_style('hot')
                elif key == ord('6'):
                    self.switch_thermal_style('plasma')
                elif key == ord('7'):
                    self.switch_thermal_style('viridis')
                elif key == ord('+') or key == ord('='):
                    self.adjust_depth_range(0.5)
                elif key == ord('-'):
                    self.adjust_depth_range(-0.5)
                elif key == ord('g'):
                    self.outline_glow = not self.outline_glow
                    print(f"🌟 Glow effect: {'ON' if self.outline_glow else 'OFF'}")
                elif key == ord('n'):
                    self.thermal_noise_intensity = 0.2 if self.thermal_noise_intensity == 0 else 0
                    print(f"📡 Thermal noise: {'ON' if self.thermal_noise_intensity > 0 else 'OFF'}")
                    
        except KeyboardInterrupt:
            print("\n🎯 PREDATOR VISION DEACTIVATED")
        except Exception as e:
            print(f"❌ System error: {e}")
        finally:
            # Cleanup
            self.pipeline.stop()
            cv2.destroyAllWindows()
            if video_writer:
                video_writer.release()
            print("🔒 All systems secured")
    
    def switch_thermal_style(self, style):
        """Switch thermal visualization style"""
        self.thermal_style = style
        self.setup_thermal_visualization()
        print(f"🌡️  Thermal mode: {style.upper()}")
    
    def adjust_depth_range(self, delta):
        """Adjust depth range for thermal mapping"""
        new_max = self.depth_range[1] + delta
        if new_max > 1.0:  # Minimum 1 meter range
            self.depth_range = (self.depth_range[0], new_max)
            print(f"📏 Depth range: {self.depth_range[0]:.1f}-{self.depth_range[1]:.1f}m")

# Demonstration modes
class PredatorVisionDemo:
    def __init__(self):
        self.thermal_styles = ['predator', 'flir', 'iron', 'rainbow', 'hot', 'plasma', 'viridis']
        self.segmentation_methods = ['yolo_seg', 'motion', 'contour']
    
    def demo_thermal_styles(self):
        """Demonstrate different thermal styles"""
        print("🎯 THERMAL STYLE DEMONSTRATION")
        print("=" * 50)
        
        for style in self.thermal_styles:
            print(f"\n🌡️  Testing thermal style: {style.upper()}")
            input("Press Enter to start demo...")
            
            try:
                system = PredatorVisionSystem(
                    thermal_style=style,
                    segmentation_method='yolo_seg'
                )
                
                print(f"Starting {style} thermal vision...")
                system.start_predator_vision(
                    save_video=True,
                    video_filename=f'predator_{style}_demo.avi'
                )
                
            except Exception as e:
                print(f"Demo failed for {style}: {e}")
    
    def demo_segmentation_methods(self):
        """Demonstrate different segmentation methods"""
        print("🎯 SEGMENTATION METHOD DEMONSTRATION")
        print("=" * 50)
        
        for method in self.segmentation_methods:
            print(f"\n🔍 Testing segmentation: {method.upper()}")
            input("Press Enter to start demo...")
            
            try:
                system = PredatorVisionSystem(
                    thermal_style='predator',
                    segmentation_method=method
                )
                
                print(f"Starting {method} segmentation...")
                system.start_predator_vision(
                    save_video=True,
                    video_filename=f'predator_{method}_demo.avi'
                )
                
            except Exception as e:
                print(f"Demo failed for {method}: {e}")

# Usage Examples
if __name__ == "__main__":
    print("🎯 PREDATOR VISION SYSTEM")
    print("=" * 50)
    print("Select mode:")
    print("1. Standard Predator Vision")
    print("2. FLIR Thermal Style")
    print("3. Demo All Thermal Styles")
    print("4. Demo All Segmentation Methods")
    print("5. Custom Configuration")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    try:
        if choice == '1':
            # Standard Predator Vision
            system = PredatorVisionSystem(
                thermal_style='predator',
                segmentation_method='yolo_seg',
                depth_range=(0.5, 4.0)
            )
            system.start_predator_vision(save_video=True, video_filename='predator_vision_standard.avi')
            
        elif choice == '2':
            # FLIR Style
            system = PredatorVisionSystem(
                thermal_style='flir',
                segmentation_method='yolo_seg',
                depth_range=(0.5, 5.0)
            )
            system.start_predator_vision(save_video=True, video_filename='flir_thermal_vision.avi')
            
        elif choice == '3':
            # Demo thermal styles
            demo = PredatorVisionDemo()
            demo.demo_thermal_styles()
            
        elif choice == '4':
            # Demo segmentation methods
            demo = PredatorVisionDemo()
            demo.demo_segmentation_methods()
            
        elif choice == '5':
            # Custom configuration
            print("\n🎯 CUSTOM CONFIGURATION")
            print("=" * 30)
            
            # Thermal style selection
            print("Thermal styles:")
            styles = ['predator', 'flir', 'iron', 'rainbow', 'hot', 'plasma', 'viridis']
            for i, style in enumerate(styles, 1):
                print(f"  {i}. {style.upper()}")
            
            style_choice = input("Select thermal style (1-7): ").strip()
            thermal_style = styles[int(style_choice) - 1] if style_choice.isdigit() and 1 <= int(style_choice) <= 7 else 'predator'
            
            # Segmentation method selection
            print("\nSegmentation methods:")
            methods = ['yolo_seg', 'motion', 'contour']
            for i, method in enumerate(methods, 1):
                print(f"  {i}. {method.upper()}")
            
            method_choice = input("Select segmentation method (1-3): ").strip()
            seg_method = methods[int(method_choice) - 1] if method_choice.isdigit() and 1 <= int(method_choice) <= 3 else 'yolo_seg'
            
            # Depth range configuration
            print(f"\nCurrent depth range: 0.5-4.0 meters")
            max_depth = input("Enter max depth in meters (default 4.0): ").strip()
            max_depth = float(max_depth) if max_depth else 4.0
            
            # Initialize custom system
            system = PredatorVisionSystem(
                thermal_style=thermal_style,
                segmentation_method=seg_method,
                depth_range=(0.5, max_depth),
                confidence_threshold=0.5
            )
            
            filename = f'predator_{thermal_style}_{seg_method}_custom.avi'
            system.start_predator_vision(save_video=True, video_filename=filename)
            
        else:
            print("Invalid choice, starting standard Predator Vision...")
            system = PredatorVisionSystem()
            system.start_predator_vision(save_video=True)
            
    except KeyboardInterrupt:
        print("\n🎯 System terminated by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure RealSense D455 is connected")
        print("2. Install dependencies: pip install ultralytics pyrealsense2 opencv-python")
        print("3. Check camera permissions")
```

## Advanced Features & Customization

### Thermal Color Schemes

```python
# Additional thermal effects you can add to the class:

def create_advanced_thermal_effects(self):
    """Create advanced thermal visualization effects"""
    
    # Heat signature simulation
    def heat_signature_effect(self, depth_image, color_image):
        """Simulate realistic heat signatures"""
        # Convert to grayscale for heat calculation
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Simulate body heat (detect warm objects)
        warm_mask = gray > 100  # Assume brighter areas are warmer
        
        # Enhance thermal response for warm objects
        enhanced_depth = depth_image.copy()
        enhanced_depth[warm_mask] = enhanced_depth[warm_mask] * 0.7  # Make warm objects appear closer
        
        return enhanced_depth
    
    # Motion heat trails
    def add_motion_trails(self, thermal_image, motion_mask):
        """Add heat trails for moving objects"""
        if not hasattr(self, 'motion_history'):
            self.motion_history = []
        
        # Store motion mask with timestamp
        self.motion_history.append({
            'mask': motion_mask.copy(),
            'timestamp': time.time(),
            'intensity': 255
        })
        
        # Decay old motion trails
        current_time = time.time()
        for trail in self.motion_history[:]:
            age = current_time - trail['timestamp']
            if age > 2.0:  # Remove trails older than 2 seconds
                self.motion_history.remove(trail)
            else:
                # Fade trail intensity
                trail['intensity'] = int(255 * (1 - age / 2.0))
        
        # Apply motion trails to thermal image
        for trail in self.motion_history:
            trail_color = (0, trail['intensity'], trail['intensity'])
            thermal_image[trail['mask'] > 0] = trail_color
        
        return thermal_image

# Audio effects integration
def add_audio_effects(self):
    """Add Predator-style audio effects"""
    try:
        import pygame
        pygame.mixer.init()
        
        # Load sound effects (you would need to provide these files)
        sounds = {
            'scan': 'predator_scan.wav',
            'target_lock': 'target_lock.wav',
            'vision_activate': 'vision_activate.wav'
        }
        
        self.audio_enabled = True
        self.sounds = {}
        
        for name, filename in sounds.items():
            try:
                self.sounds[name] = pygame.mixer.Sound(filename)
            except:
                print(f"Audio file {filename} not found")
                
    except ImportError:
        print("pygame not available for audio effects")
        self.audio_enabled = False

# Real-time parameter adjustment
def create_control_panel(self):
    """Create real-time parameter adjustment panel"""
    import tkinter as tk
    from tkinter import ttk
    
    def update_parameters():
        # This would run in a separate thread to update parameters
        pass
    
    # Create control window
    control_window = tk.Tk()
    control_window.title("Predator Vision Controls")
    control_window.geometry("300x400")
    
    # Thermal style selector
    ttk.Label(control_window, text="Thermal Style:").pack(pady=5)
    style_var = tk.StringVar(value=self.thermal_style)
    style_combo = ttk.Combobox(control_window, textvariable=style_var,
                              values=['predator', 'flir', 'iron', 'rainbow'])
    style_combo.pack(pady=5)
    
    # Depth range sliders
    ttk.Label(control_window, text="Max Depth (m):").pack(pady=5)
    depth_var = tk.DoubleVar(value=self.depth_range[1])
    depth_scale = ttk.Scale(control_window, from_=1.0, to=10.0, 
                           variable=depth_var, orient='horizontal')
    depth_scale.pack(pady=5, fill='x', padx=20)
    
    # Confidence threshold
    ttk.Label(control_window, text="Detection Confidence:").pack(pady=5)
    conf_var = tk.DoubleVar(value=self.confidence_threshold)
    conf_scale = ttk.Scale(control_window, from_=0.1, to=1.0,
                          variable=conf_var, orient='horizontal')
    conf_scale.pack(pady=5, fill='x', padx=20)
    
    return control_window
```

## Performance Optimization Tips

### GPU Acceleration
```python
# Enable GPU processing for better performance
def enable_gpu_acceleration(self):
    """Enable GPU acceleration where possible"""
    try:
        import torch
        if torch.cuda.is_available():
            print("🚀 GPU acceleration available")
            self.device = 'cuda'
        else:
            print("💻 Using CPU processing")
            self.device = 'cpu'
    except ImportError:
        self.device = 'cpu'

# Optimize frame processing
def optimize_processing(self):
    """Optimize frame processing pipeline"""
    # Use threading for parallel processing
    import threading
    import queue
    
    # Create processing queue
    self.frame_queue = queue.Queue(maxsize=5)
    self.result_queue = queue.Queue(maxsize=5)
    
    # Start processing thread
    self.processing_thread = threading.Thread(target=self.process_frames_threaded)
    self.processing_thread.daemon = True
    self.processing_thread.start()
```

## Installation & Setup Guide

### Complete Dependencies
```bash
# Core requirements
pip install ultralytics pyrealsense2 opencv-python numpy

# Advanced features
pip install scipy scikit-image matplotlib

# Audio effects (optional)
pip install pygame

# GPU acceleration (optional)
pip install torch torchvision

# Control panel (optional)
pip install tkinter  # Usually included with Python
```

### Hardware Requirements
- **RealSense D455 camera** (required for depth data)
- **USB 3.0 port** (for optimal performance)
- **8GB+ RAM** (recommended)
- **GPU** (optional but recommended for real-time processing)

### System Setup Checklist
1. ✅ Install Intel RealSense SDK
2. ✅ Test camera with RealSense Viewer
3. ✅ Install Python dependencies
4. ✅ Run camera diagnostics
5. ✅ Test thermal visualization

## Usage Examples

### Basic Predator Vision
```python
# Simple setup
system = PredatorVisionSystem(thermal_style='predator')
system.start_predator_vision(save_video=True)
```

### Advanced FLIR-style
```python
# FLIR thermal camera simulation
system = PredatorVisionSystem(
    thermal_style='flir',
    segmentation_method='yolo_seg',
    depth_range=(0.3, 8.0),
    confidence_threshold=0.6
)
system.start_predator_vision()
```

### Custom Military-style
```python
# Military thermal scope simulation
system = PredatorVisionSystem(
    thermal_style='iron',
    segmentation_method='motion',
    depth_range=(1.0, 15.0)
)
system.thermal_noise_intensity = 0.15  # Add realistic noise
system.outline_glow = True              # Enable target highlighting
system.start_predator_vision()
```

## Key Features Summary

🎯 **Thermal Vision Effects:**
- Multiple thermal color schemes (Predator, FLIR, Iron, Rainbow, etc.)
- Depth-based thermal mapping with realistic heat signatures
- Customizable thermal noise and effects

🔍 **Object Segmentation:**
- YOLO-based precise object outlines
- Motion detection for moving targets
- Contour-based edge detection fallback

🌟 **Visual Effects:**
- Scanning line animation
- Glowing object outlines
- HUD overlay with system information
- Crosshair targeting reticle

⚡ **Real-time Controls:**
- Switch thermal styles on-the-fly (1-7 keys)
- Adjust depth range (+/- keys)
- Toggle effects (g for glow, n for noise)
- Save frames and video recording

This creates an authentic "Predator vision" experience that combines the thermal imaging effect with precise object segmentation, giving you that iconic sci-fi thermal sight with real depth information from your RealSense camera!
            