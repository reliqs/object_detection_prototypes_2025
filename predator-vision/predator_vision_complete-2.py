#!/usr/bin/env python3
"""
Predator Vision - Complete Thermal Depth Segmentation System
============================================================

Advanced thermal vision system with object segmentation, performance optimization,
and real-time effects. Combines RealSense depth data with AI-powered object detection
to create an authentic Predator/FLIR thermal vision experience.

Requirements:
    pip install ultralytics pyrealsense2 opencv-python numpy scipy scikit-image matplotlib pygame torch

Hardware:
    - Intel RealSense D455 camera
    - USB 3.0 port
    - 8GB+ RAM recommended
    - GPU optional but recommended

Author: AI Assistant
Version: 2.0
"""

import pyrealsense2 as rs
import cv2
import numpy as np
import time
import json
from datetime import datetime
import threading
import queue
from pathlib import Path
import sys
import platform

# Advanced features imports
try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available - advanced filtering disabled")

try:
    from skimage import segmentation, measure, filters
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️  scikit-image not available - advanced segmentation disabled")

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - custom colormaps disabled")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available - using basic segmentation")

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not available - audio effects disabled")

try:
    import torch
    if torch.cuda.is_available():
        CUDA_AVAILABLE = True
        print("🚀 CUDA GPU acceleration available")
    else:
        CUDA_AVAILABLE = False
        print("💻 Using CPU processing")
except ImportError:
    CUDA_AVAILABLE = False
    print("💻 PyTorch not available - using CPU only")

try:
    import tkinter as tk
    from tkinter import ttk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("⚠️  tkinter not available - control panel disabled")


class PredatorVisionSystem:
    """Complete Predator Vision System with advanced features and optimizations"""
    
    def __init__(self, 
                 segmentation_method='yolo_seg',
                 confidence_threshold=0.5,
                 thermal_style='predator',
                 depth_range=(0.5, 5.0),
                 enable_gpu=True,
                 enable_threading=True,
                 enable_audio=False):
        """
        Initialize Predator Vision System
        
        Args:
            segmentation_method: 'yolo_seg', 'motion', 'contour', 'watershed', 'adaptive'
            confidence_threshold: Detection confidence threshold (0.0-1.0)
            thermal_style: 'predator', 'flir', 'iron', 'rainbow', 'hot', 'plasma', 'viridis'
            depth_range: (min_depth, max_depth) in meters for thermal mapping
            enable_gpu: Use GPU acceleration if available
            enable_threading: Use multi-threading for performance
            enable_audio: Enable audio effects
        """
        print("🎯 Initializing Predator Vision System...")
        
        # Core settings
        self.segmentation_method = segmentation_method
        self.confidence_threshold = confidence_threshold
        self.thermal_style = thermal_style
        self.depth_range = depth_range
        self.enable_gpu = enable_gpu and CUDA_AVAILABLE
        self.enable_threading = enable_threading
        self.enable_audio = enable_audio and PYGAME_AVAILABLE
        
        # Performance optimization
        self.device = 'cuda' if self.enable_gpu else 'cpu'
        
        # Initialize components
        self.initialize_models()
        self.initialize_camera()
        self.setup_thermal_visualization()
        self.setup_performance_optimization()
        self.setup_audio_effects()
        
        # Visual effects settings
        self.scan_line_pos = 0
        self.scan_direction = 1
        self.thermal_noise_intensity = 0.1
        self.outline_thickness = 2
        self.outline_glow = True
        self.motion_trails_enabled = True
        self.heat_signature_simulation = True
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []
        self.processing_times = []
        
        # Advanced features
        self.motion_history = []
        self.target_tracking_enabled = True
        self.tracked_objects = {}
        self.next_track_id = 0
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, 
            varThreshold=50,
            history=500
        )
        
        # Control panel
        self.control_panel = None
        self.control_vars = {}
        
        print("✅ Predator Vision System initialized successfully")
    
    def initialize_models(self):
        """Initialize AI models with GPU support"""
        print("🤖 Loading AI models...")
        
        if self.segmentation_method == 'yolo_seg' and YOLO_AVAILABLE:
            try:
                self.model = YOLO('yolov8n-seg.pt')
                if self.enable_gpu:
                    self.model.to('cuda')
                print("✅ YOLO segmentation model loaded with GPU support")
            except Exception as e:
                print(f"❌ Failed to load YOLO: {e}, falling back to contour method")
                self.segmentation_method = 'contour'
        
        # Initialize additional models for advanced segmentation
        if self.segmentation_method == 'adaptive' and SKIMAGE_AVAILABLE:
            print("✅ Adaptive segmentation enabled")
        
        print(f"🔧 Using segmentation method: {self.segmentation_method}")
    
    def initialize_camera(self):
        """Initialize RealSense camera with optimized settings"""
        print("📷 Initializing RealSense camera...")
        
        try:
            # Configure pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable streams with optimal settings
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Get depth sensor settings
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Optimize depth sensor settings
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # Create alignment object
            self.align = rs.align(rs.stream.color)
            
            # Get camera intrinsics
            color_profile = profile.get_stream(rs.stream.color)
            self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            # Create depth filters for better quality
            self.setup_depth_filters()
            
            print("✅ RealSense camera initialized with optimizations")
            
        except Exception as e:
            print(f"❌ Failed to initialize RealSense: {e}")
            raise RuntimeError("RealSense camera required for Predator Vision")
    
    def setup_depth_filters(self):
        """Setup depth post-processing filters"""
        self.depth_filters = []
        
        # Decimation filter
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 1)
        self.depth_filters.append(decimation)
        
        # Spatial filter
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.depth_filters.append(spatial)
        
        # Temporal filter
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal.set_option(rs.option.filter_smooth_delta, 20)
        self.depth_filters.append(temporal)
        
        # Hole filling filter
        hole_filling = rs.hole_filling_filter()
        self.depth_filters.append(hole_filling)
        
        print("✅ Depth filters configured")
    
    def setup_thermal_visualization(self):
        """Setup thermal colormap and advanced visualization"""
        print("🌡️  Setting up thermal visualization...")
        
        self.thermal_colormaps = {
            'predator': self.create_predator_colormap(),
            'flir': cv2.COLORMAP_INFERNO,
            'iron': cv2.COLORMAP_HOT,
            'rainbow': cv2.COLORMAP_RAINBOW,
            'hot': cv2.COLORMAP_HOT,
            'plasma': cv2.COLORMAP_PLASMA,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'turbo': cv2.COLORMAP_TURBO
        }
        
        # Set current colormap
        if self.thermal_style == 'predator':
            self.colormap = self.thermal_colormaps['predator']
        else:
            self.colormap = self.thermal_colormaps.get(self.thermal_style, cv2.COLORMAP_INFERNO)
        
        # Advanced thermal effects
        self.thermal_history = []
        self.heat_accumulation = None
        
        print("✅ Thermal visualization configured")
    
    def setup_performance_optimization(self):
        """Setup multi-threading and performance optimizations"""
        if self.enable_threading:
            print("⚡ Setting up performance optimizations...")
            
            # Frame processing queues
            self.frame_queue = queue.Queue(maxsize=3)
            self.result_queue = queue.Queue(maxsize=3)
            self.depth_queue = queue.Queue(maxsize=3)
            
            # Processing threads
            self.processing_active = False
            self.processing_threads = []
            
            # Create worker threads
            for i in range(2):  # 2 worker threads
                thread = threading.Thread(target=self.process_frames_worker, daemon=True)
                self.processing_threads.append(thread)
            
            print("✅ Multi-threading configured")
    
    def setup_audio_effects(self):
        """Setup audio effects system"""
        if self.enable_audio and PYGAME_AVAILABLE:
            print("🔊 Setting up audio effects...")
            
            # Create simple beep sounds programmatically
            self.audio_effects = {
                'scan': self.generate_scan_sound(),
                'target_lock': self.generate_target_lock_sound(),
                'activation': self.generate_activation_sound()
            }
            
            print("✅ Audio effects configured")
        else:
            self.audio_effects = {}
    
    def generate_scan_sound(self):
        """Generate scan sound effect"""
        if not PYGAME_AVAILABLE:
            return None
        
        duration = 0.1
        sample_rate = 22050
        frames = int(duration * sample_rate)
        
        # Generate sweep sound
        arr = np.zeros(frames)
        for i in range(frames):
            freq = 800 + (i / frames) * 400
            arr[i] = np.sin(2 * np.pi * freq * i / sample_rate) * 0.3
        
        # Convert to pygame sound
        sound_array = (arr * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(sound_array)
        return sound
    
    def generate_target_lock_sound(self):
        """Generate target lock sound effect"""
        if not PYGAME_AVAILABLE:
            return None
        
        duration = 0.2
        sample_rate = 22050
        frames = int(duration * sample_rate)
        
        # Generate double beep
        arr = np.zeros(frames)
        for i in range(frames):
            if i < frames // 3 or (i > frames * 2 // 3 and i < frames):
                freq = 1200
                arr[i] = np.sin(2 * np.pi * freq * i / sample_rate) * 0.5
        
        sound_array = (arr * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(sound_array)
        return sound
    
    def generate_activation_sound(self):
        """Generate activation sound effect"""
        if not PYGAME_AVAILABLE:
            return None
        
        duration = 0.5
        sample_rate = 22050
        frames = int(duration * sample_rate)
        
        # Generate rising tone
        arr = np.zeros(frames)
        for i in range(frames):
            freq = 400 + (i / frames) * 800
            envelope = (i / frames) * np.exp(-(i / frames) * 2)
            arr[i] = np.sin(2 * np.pi * freq * i / sample_rate) * envelope * 0.4
        
        sound_array = (arr * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(sound_array)
        return sound
    
    def create_predator_colormap(self):
        """Create custom Predator-style colormap"""
        colors = np.array([
            [0, 0, 0],        # Black (cold/far)
            [0, 0, 64],       # Dark blue
            [0, 0, 128],      # Blue
            [0, 64, 192],     # Blue-cyan
            [0, 128, 255],    # Cyan
            [0, 255, 255],    # Bright cyan
            [0, 255, 128],    # Cyan-green
            [0, 255, 0],      # Green
            [128, 255, 0],    # Yellow-green
            [255, 255, 0],    # Yellow
            [255, 192, 0],    # Orange-yellow
            [255, 128, 0],    # Orange
            [255, 64, 0],     # Red-orange
            [255, 0, 0],      # Red
            [255, 128, 128],  # Light red
            [255, 255, 255]   # White (very hot/very close)
        ], dtype=np.uint8)
        
        # Interpolate to create 256-color palette
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
        """Get processed frame with depth filtering"""
        try:
            frames = self.pipeline.wait_for_frames()
            
            # Apply depth filters
            depth_frame = frames.get_depth_frame()
            for filter_func in self.depth_filters:
                depth_frame = filter_func.process(depth_frame)
            
            # Align frames
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            
            if not color_frame or not depth_frame:
                return None, None, None
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image, depth_frame
            
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None, None, None
    
    def create_thermal_depth_image(self, depth_image, color_image=None):
        """Create advanced thermal visualization from depth data"""
        start_time = time.time()
        
        # Convert depth to meters
        depth_meters = depth_image * self.depth_scale
        
        # Advanced depth processing
        if SCIPY_AVAILABLE:
            # Apply Gaussian smoothing for better thermal effect
            depth_smooth = gaussian_filter(depth_meters, sigma=1.0)
        else:
            depth_smooth = cv2.GaussianBlur(depth_meters, (5, 5), 1.0)
        
        # Heat signature simulation
        if self.heat_signature_simulation and color_image is not None:
            depth_smooth = self.simulate_heat_signatures(depth_smooth, color_image)
        
        # Clip to depth range and invert (closer = hotter)
        depth_clipped = np.clip(depth_smooth, self.depth_range[0], self.depth_range[1])
        depth_inverted = self.depth_range[1] - depth_clipped
        
        # Normalize to 0-255 range
        depth_normalized = ((depth_inverted - 0) / (self.depth_range[1] - self.depth_range[0]) * 255)
        
        # Add thermal noise for realism
        if self.thermal_noise_intensity > 0:
            noise = np.random.normal(0, self.thermal_noise_intensity * 255, depth_normalized.shape)
            depth_normalized = np.clip(depth_normalized + noise, 0, 255)
        
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Heat accumulation over time
        if self.heat_accumulation is None:
            self.heat_accumulation = depth_normalized.astype(np.float32)
        else:
            # Accumulate heat over time with decay
            self.heat_accumulation = 0.95 * self.heat_accumulation + 0.05 * depth_normalized.astype(np.float32)
            depth_normalized = np.clip(self.heat_accumulation, 0, 255).astype(np.uint8)
        
        # Apply thermal colormap
        if self.thermal_style == 'predator':
            thermal_image = cv2.applyColorMap(depth_normalized, self.colormap)
        else:
            thermal_image = cv2.applyColorMap(depth_normalized, self.colormap)
        
        # Add motion trails
        if self.motion_trails_enabled:
            thermal_image = self.add_motion_trails(thermal_image)
        
        # Record processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        return thermal_image, depth_normalized
    
    def simulate_heat_signatures(self, depth_image, color_image):
        """Simulate realistic heat signatures based on color content"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Detect potential warm objects (brighter areas, skin tones, etc.)
        warm_mask = gray > 120  # Bright objects tend to be warmer
        
        # Detect skin tones (simplified)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, (0, 30, 60), (20, 150, 255))
        
        # Combine masks
        heat_sources = warm_mask | (skin_mask > 0)
        
        # Enhance thermal response for heat sources
        enhanced_depth = depth_image.copy()
        enhanced_depth[heat_sources] = enhanced_depth[heat_sources] * 0.8  # Make them appear closer/hotter
        
        return enhanced_depth
    
    def add_motion_trails(self, thermal_image):
        """Add heat trails for moving objects"""
        current_time = time.time()
        
        # Detect motion using background subtractor
        gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        motion_mask = self.bg_subtractor.apply(gray)
        
        # Store motion with timestamp
        if np.any(motion_mask > 0):
            self.motion_history.append({
                'mask': motion_mask.copy(),
                'timestamp': current_time,
                'intensity': 255
            })
        
        # Remove old trails and update intensities
        for trail in self.motion_history[:]:
            age = current_time - trail['timestamp']
            if age > 3.0:  # Remove trails older than 3 seconds
                self.motion_history.remove(trail)
            else:
                # Fade trail intensity over time
                fade_factor = 1.0 - (age / 3.0)
                trail['intensity'] = int(255 * fade_factor)
        
        # Apply motion trails to thermal image
        for trail in self.motion_history:
            if trail['intensity'] > 50:  # Only show visible trails
                trail_color = (0, trail['intensity'], trail['intensity'])
                thermal_image[trail['mask'] > 0] = trail_color
        
        return thermal_image
    
    def segment_objects(self, color_image):
        """Advanced object segmentation with multiple methods"""
        if self.segmentation_method == 'yolo_seg' and hasattr(self, 'model'):
            return self.segment_yolo(color_image)
        elif self.segmentation_method == 'adaptive' and SKIMAGE_AVAILABLE:
            return self.segment_adaptive(color_image)
        elif self.segmentation_method == 'watershed':
            return self.segment_watershed(color_image)
        elif self.segmentation_method == 'motion':
            return self.segment_motion(color_image)
        else:
            return self.segment_contour(color_image)
    
    def segment_yolo(self, image):
        """YOLO segmentation with GPU acceleration"""
        try:
            results = self.model(image, conf=self.confidence_threshold, device=self.device)
            return results[0] if results else None
        except Exception as e:
            print(f"YOLO segmentation error: {e}")
            return None
    
    def segment_adaptive(self, image):
        """Adaptive segmentation using scikit-image"""
        if not SKIMAGE_AVAILABLE:
            return self.segment_contour(image)
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = filters.threshold_local(gray, block_size=35, offset=10)
            binary = gray > thresh
            
            # Label connected components
            labels = measure.label(binary)
            
            # Convert to contours
            contours = []
            for region in measure.regionprops(labels):
                if region.area > 1000:  # Minimum area threshold
                    # Convert region to contour format
                    coords = region.coords
                    contour = np.array([[coord[1], coord[0]] for coord in coords], dtype=np.int32)
                    
                    # Get convex hull for smoother contour
                    hull = cv2.convexHull(contour)
                    contours.append(hull)
            
            return contours
            
        except Exception as e:
            print(f"Adaptive segmentation error: {e}")
            return self.segment_contour(image)
    
    def segment_watershed(self, image):
        """Watershed segmentation for blob-like objects"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            
            # Unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Apply watershed
            markers = cv2.watershed(image, markers)
            
            # Extract contours from segments
            contours = []
            for label in np.unique(markers):
                if label <= 1:  # Skip background and borders
                    continue
                
                mask = (markers == label).astype(np.uint8) * 255
                segment_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if segment_contours:
                    largest_contour = max(segment_contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 500:
                        contours.append(largest_contour)
            
            return contours
            
        except Exception as e:
            print(f"Watershed segmentation error: {e}")
            return self.segment_contour(image)
    
    def segment_motion(self, image):
        """Motion-based segmentation with improved filtering"""
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(image)
            
            # Advanced morphological operations
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            
            # Remove noise
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
            # Fill holes
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 800:  # Minimum area
                    # Check aspect ratio to filter out noise
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio
                        valid_contours.append(contour)
            
            return valid_contours
            
        except Exception as e:
            print(f"Motion segmentation error: {e}")
            return []
    
    def segment_contour(self, image):
        """Enhanced edge-based contour segmentation"""
        try:
            # Multi-scale edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Multi-scale Canny edge detection
            edges1 = cv2.Canny(filtered, 50, 100)
            edges2 = cv2.Canny(filtered, 100, 200)
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Morphological operations to close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and refine contours
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    # Smooth contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    smoothed = cv2.approxPolyDP(contour, epsilon, True)
                    valid_contours.append(smoothed)
            
            return valid_contours
            
        except Exception as e:
            print(f"Contour segmentation error: {e}")
            return []
    
    def track_objects(self, segmentations):
        """Advanced object tracking across frames"""
        if not self.target_tracking_enabled:
            return segmentations
        
        current_objects = []
        
        if self.segmentation_method == 'yolo_seg' and segmentations and segmentations.boxes is not None:
            # YOLO format tracking
            boxes = segmentations.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)
                class_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                
                current_objects.append({
                    'center': center,
                    'bbox': [x1, y1, x2, y2],
                    'area': area,
                    'class_id': class_id,
                    'confidence': confidence
                })
        
        elif isinstance(segmentations, list):
            # Contour format tracking
            for i, contour in enumerate(segmentations):
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                area = cv2.contourArea(contour)
                
                current_objects.append({
                    'center': center,
                    'bbox': [x, y, x + w, y + h],
                    'area': area,
                    'class_id': 0,
                    'confidence': 0.8
                })
        
        # Update tracking
        self.update_object_tracking(current_objects)
        return segmentations
    
    def update_object_tracking(self, current_objects):
        """Update object tracking with new detections"""
        # Simple distance-based tracking
        max_distance = 100  # Maximum distance for object association
        
        # Update existing tracks
        for track_id, tracked_obj in list(self.tracked_objects.items()):
            best_match = None
            best_distance = float('inf')
            
            for obj in current_objects:
                distance = np.sqrt((obj['center'][0] - tracked_obj['center'][0])**2 + 
                                 (obj['center'][1] - tracked_obj['center'][1])**2)
                
                if distance < best_distance and distance < max_distance:
                    best_distance = distance
                    best_match = obj
            
            if best_match:
                # Update existing track
                self.tracked_objects[track_id].update({
                    'center': best_match['center'],
                    'bbox': best_match['bbox'],
                    'area': best_match['area'],
                    'last_seen': time.time(),
                    'track_length': tracked_obj.get('track_length', 0) + 1
                })
                current_objects.remove(best_match)
            else:
                # Mark as lost if not seen for too long
                if time.time() - tracked_obj.get('last_seen', 0) > 2.0:
                    del self.tracked_objects[track_id]
        
        # Create new tracks for unmatched objects
        for obj in current_objects:
            if obj['area'] > 1000:  # Only track significant objects
                self.tracked_objects[self.next_track_id] = {
                    'center': obj['center'],
                    'bbox': obj['bbox'],
                    'area': obj['area'],
                    'class_id': obj['class_id'],
                    'confidence': obj['confidence'],
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'track_length': 1
                }
                self.next_track_id += 1
    
    def draw_segmentation_outlines(self, thermal_image, segmentations, color_image=None):
        """Draw advanced segmentation outlines with tracking info"""
        result = thermal_image.copy()
        
        # Track objects first
        segmentations = self.track_objects(segmentations)
        
        if self.segmentation_method == 'yolo_seg' and segmentations and segmentations.masks is not None:
            # YOLO format with masks
            masks = segmentations.masks.data.cpu().numpy()
            boxes = segmentations.boxes
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (thermal_image.shape[1], thermal_image.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Get object info
                class_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                class_name = self.model.names[class_id]
                
                # Draw outline with advanced effects
                self.draw_advanced_outline(result, mask_binary, class_name, confidence, class_id)
        
        elif isinstance(segmentations, list):
            # Contour format
            for i, contour in enumerate(segmentations):
                # Create mask from contour
                mask = np.zeros(thermal_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                
                # Draw outline
                self.draw_advanced_outline(result, mask, f"Target_{i}", 0.8, i)
        
        # Draw tracking information
        self.draw_tracking_info(result)
        
        return result
    
    def draw_advanced_outline(self, image, mask, label, confidence, class_id):
        """Draw advanced glowing outline with animations"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return
        
        # Dynamic colors based on confidence and class
        base_hue = (class_id * 30) % 180  # Different hue for each class
        confidence_factor = max(0.5, confidence)
        
        # Predator-style colors with confidence-based intensity
        outline_color = (
            int(confidence_factor * 255),  # Blue component
            int(255 * confidence_factor),  # Green component  
            int(255 * min(1.0, confidence_factor * 1.5))  # Red component
        )
        
        glow_color = (
            int(confidence_factor * 128),
            int(255 * confidence_factor * 0.8),
            0
        )
        
        # Animated pulsing effect
        pulse_factor = 0.8 + 0.2 * np.sin(time.time() * 4 + class_id)
        outline_thickness = int(self.outline_thickness * pulse_factor)
        
        for contour in contours:
            if self.outline_glow:
                # Multi-layer glow effect
                for thickness in range(12, 2, -2):
                    alpha = 0.15 * (12 - thickness) / 10 * pulse_factor
                    glow_img = image.copy()
                    cv2.drawContours(glow_img, [contour], -1, glow_color, thickness)
                    cv2.addWeighted(image, 1 - alpha, glow_img, alpha, 0, image)
            
            # Main outline with thickness variation
            cv2.drawContours(image, [contour], -1, outline_color, outline_thickness)
            
            # Add corner markers for enhanced targeting look
            x, y, w, h = cv2.boundingRect(contour)
            corner_size = 15
            corner_thickness = 3
            
            # Top-left corner
            cv2.line(image, (x, y), (x + corner_size, y), outline_color, corner_thickness)
            cv2.line(image, (x, y), (x, y + corner_size), outline_color, corner_thickness)
            
            # Top-right corner
            cv2.line(image, (x + w, y), (x + w - corner_size, y), outline_color, corner_thickness)
            cv2.line(image, (x + w, y), (x + w, y + corner_size), outline_color, corner_thickness)
            
            # Bottom-left corner
            cv2.line(image, (x, y + h), (x + corner_size, y + h), outline_color, corner_thickness)
            cv2.line(image, (x, y + h), (x, y + h - corner_size), outline_color, corner_thickness)
            
            # Bottom-right corner
            cv2.line(image, (x + w, y + h), (x + w - corner_size, y + h), outline_color, corner_thickness)
            cv2.line(image, (x + w, y + h), (x + w, y + h - corner_size), outline_color, corner_thickness)
            
            # Enhanced label with background
            label_text = f"{label}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Animated label background
            bg_alpha = 0.7 + 0.3 * np.sin(time.time() * 2 + class_id)
            label_bg = np.zeros_like(image)
            cv2.rectangle(label_bg, (x, y - text_h - 15), (x + text_w + 10, y), outline_color, -1)
            cv2.addWeighted(image, 1 - bg_alpha * 0.5, label_bg, bg_alpha * 0.5, 0, image)
            
            # Label border
            cv2.rectangle(image, (x - 2, y - text_h - 17), (x + text_w + 12, y + 2), outline_color, 2)
            
            # Label text with glow
            cv2.putText(image, label_text, (x + 5, y - 8), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(image, label_text, (x + 5, y - 8), font, font_scale, outline_color, thickness)
    
    def draw_tracking_info(self, image):
        """Draw object tracking information"""
        if not self.target_tracking_enabled or not self.tracked_objects:
            return
        
        # Draw tracking trails
        for track_id, tracked_obj in self.tracked_objects.items():
            center = tracked_obj['center']
            track_length = tracked_obj.get('track_length', 0)
            
            if track_length > 5:  # Only show established tracks
                # Draw tracking cross
                cross_size = 20
                track_color = (0, 255, 255)  # Cyan for tracking
                
                cv2.line(image, (center[0] - cross_size, center[1]), 
                        (center[0] + cross_size, center[1]), track_color, 2)
                cv2.line(image, (center[0], center[1] - cross_size), 
                        (center[0], center[1] + cross_size), track_color, 2)
                
                # Track ID
                cv2.putText(image, f"T{track_id}", (center[0] + 25, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2)
    
    def add_predator_effects(self, image):
        """Add comprehensive Predator-style visual effects"""
        result = image.copy()
        height, width = result.shape[:2]
        
        # Animated scanning line with multiple effects
        scan_y = int(self.scan_line_pos)
        scan_color = (0, 255, 255)  # Bright cyan
        
        # Main scanning line with glow
        for offset in range(-3, 4):
            alpha = 1.0 - abs(offset) * 0.2
            line_color = tuple(int(c * alpha) for c in scan_color)
            cv2.line(result, (0, scan_y + offset), (width, scan_y + offset), line_color, 1)
        
        # Scanning line data readout effect
        if scan_y % 20 == 0:  # Periodic data bursts
            for i in range(0, width, 50):
                cv2.circle(result, (i, scan_y), 2, scan_color, -1)
        
        # Update scan position with variable speed
        scan_speed = 2 + int(np.sin(time.time()) * 2)
        self.scan_line_pos += self.scan_direction * scan_speed
        
        if self.scan_line_pos >= height or self.scan_line_pos <= 0:
            self.scan_direction *= -1
            # Play scan sound effect
            if self.enable_audio and 'scan' in self.audio_effects:
                try:
                    self.audio_effects['scan'].play()
                except:
                    pass
        
        # Advanced targeting reticle
        center_x, center_y = width // 2, height // 2
        reticle_size = 40
        reticle_color = (0, 255, 0)  # Bright green
        
        # Animated reticle with rotation
        angle = time.time() * 30  # Rotation speed
        
        # Main crosshair
        cv2.line(result, (center_x - reticle_size, center_y), 
                (center_x + reticle_size, center_y), reticle_color, 3)
        cv2.line(result, (center_x, center_y - reticle_size), 
                (center_x, center_y + reticle_size), reticle_color, 3)
        
        # Rotating outer ring
        for i in range(0, 360, 30):
            angle_rad = np.radians(i + angle)
            x1 = center_x + int((reticle_size + 10) * np.cos(angle_rad))
            y1 = center_y + int((reticle_size + 10) * np.sin(angle_rad))
            x2 = center_x + int((reticle_size + 20) * np.cos(angle_rad))
            y2 = center_y + int((reticle_size + 20) * np.sin(angle_rad))
            cv2.line(result, (x1, y1), (x2, y2), reticle_color, 2)
        
        # Corner brackets with animation
        bracket_size = 20
        bracket_anim = int(5 * np.sin(time.time() * 4))  # Animated size
        animated_size = bracket_size + bracket_anim
        
        corners = [
            (center_x - reticle_size, center_y - reticle_size),  # Top-left
            (center_x + reticle_size, center_y - reticle_size),  # Top-right
            (center_x - reticle_size, center_y + reticle_size),  # Bottom-left
            (center_x + reticle_size, center_y + reticle_size)   # Bottom-right
        ]
        
        bracket_offsets = [
            [(animated_size, 0), (0, animated_size)],           # Top-left
            [(-animated_size, 0), (0, animated_size)],          # Top-right
            [(animated_size, 0), (0, -animated_size)],          # Bottom-left
            [(-animated_size, 0), (0, -animated_size)]          # Bottom-right
        ]
        
        for corner, offsets in zip(corners, bracket_offsets):
            for offset in offsets:
                cv2.line(result, corner, (corner[0] + offset[0], corner[1] + offset[1]), 
                        reticle_color, 3)
        
        # Distance measurement lines (simulated)
        for i in range(4):
            line_y = center_y + (i - 1.5) * 20
            cv2.line(result, (center_x + reticle_size + 30, line_y), 
                    (center_x + reticle_size + 50, line_y), reticle_color, 1)
            cv2.putText(result, f"{2.5 + i * 0.5:.1f}m", 
                       (center_x + reticle_size + 55, line_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, reticle_color, 1)
        
        return result
    
    def add_hud_overlay(self, image):
        """Add comprehensive HUD (Heads-Up Display) elements"""
        height, width = image.shape[:2]
        
        # Calculate performance metrics
        current_time = time.time()
        runtime = current_time - self.start_time
        fps = self.frame_count / runtime if runtime > 0 else 0
        
        # Update FPS history for smoothing
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Calculate processing performance
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        # HUD color scheme
        hud_color = (0, 255, 0)  # Bright green
        warning_color = (0, 255, 255)  # Yellow for warnings
        critical_color = (0, 0, 255)  # Red for critical
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Main system status panel
        info_lines = [
            "PREDATOR VISION SYSTEM",
            f"STATUS: {'ACTIVE' if not hasattr(self, '_paused') or not self._paused else 'STANDBY'}",
            f"THERMAL MODE: {self.thermal_style.upper()}",
            f"DETECTION: {self.segmentation_method.upper()}",
            f"DEPTH RANGE: {self.depth_range[0]:.1f}-{self.depth_range[1]:.1f}M",
            f"FPS: {avg_fps:.1f} ({'GPU' if self.enable_gpu else 'CPU'})",
            f"TARGETS: {len(self.tracked_objects)} TRACKED",
            f"PROC TIME: {avg_processing_time*1000:.1f}MS"
        ]
        
        # Draw main HUD panel
        panel_width = 320
        panel_height = len(info_lines) * 25 + 30
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
        
        # Panel border with corner details
        cv2.rectangle(image, (10, 10), (panel_width, panel_height), hud_color, 2)
        
        # Corner decorations
        corner_size = 15
        corners = [(10, 10), (panel_width, 10), (10, panel_height), (panel_width, panel_height)]
        for corner in corners:
            cv2.line(image, corner, (corner[0] + corner_size * (1 if corner[0] == 10 else -1), corner[1]), hud_color, 3)
            cv2.line(image, corner, (corner[0], corner[1] + corner_size * (1 if corner[1] == 10 else -1)), hud_color, 3)
        
        # Draw info lines with color coding
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 25
            
            # Color code based on content
            if "FPS" in line and avg_fps < 15:
                color = warning_color
            elif "PROC TIME" in line and avg_processing_time > 0.05:
                color = warning_color
            elif i == 0:  # Title
                color = (0, 255, 255)  # Cyan
            else:
                color = hud_color
            
            cv2.putText(image, line, (20, y_pos), font, font_scale, color, thickness)
        
        # System performance graph (mini)
        if len(self.fps_history) > 5:
            graph_x = panel_width - 100
            graph_y = 40
            graph_w = 80
            graph_h = 40
            
            # Graph background
            cv2.rectangle(image, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
            cv2.rectangle(image, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), hud_color, 1)
            
            # Draw FPS graph
            max_fps = 60  # Scale
            for i in range(1, len(self.fps_history)):
                x1 = graph_x + int((i - 1) * graph_w / len(self.fps_history))
                y1 = graph_y + graph_h - int(self.fps_history[i - 1] * graph_h / max_fps)
                x2 = graph_x + int(i * graph_w / len(self.fps_history))
                y2 = graph_y + graph_h - int(self.fps_history[i] * graph_h / max_fps)
                
                cv2.line(image, (x1, y1), (x2, y2), hud_color, 1)
        
        # Timestamp and coordinates
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(image, timestamp, (width - 250, height - 40), font, 0.5, hud_color, 1)
        
        # GPS coordinates (simulated)
        cv2.putText(image, "GPS: 34.0522°N 118.2437°W", (width - 250, height - 20), font, 0.5, hud_color, 1)
        
        # Target information panel (right side)
        if self.tracked_objects:
            target_panel_x = width - 300
            target_panel_y = 10
            target_panel_w = 280
            target_lines = ["TARGET ACQUISITION:"] + [
                f"T{tid}: {obj.get('class_id', 'UNK')} ({obj.get('track_length', 0)}f)"
                for tid, obj in list(self.tracked_objects.items())[:8]  # Show max 8 targets
            ]
            
            target_panel_h = len(target_lines) * 20 + 20
            
            # Target panel background
            overlay = image.copy()
            cv2.rectangle(overlay, (target_panel_x, target_panel_y), 
                         (target_panel_x + target_panel_w, target_panel_y + target_panel_h), (0, 0, 0), -1)
            cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
            
            # Target panel border
            cv2.rectangle(image, (target_panel_x, target_panel_y), 
                         (target_panel_x + target_panel_w, target_panel_y + target_panel_h), hud_color, 2)
            
            # Draw target list
            for i, line in enumerate(target_lines):
                y_pos = target_panel_y + 25 + i * 20
                color = (0, 255, 255) if i == 0 else hud_color  # Cyan for header
                cv2.putText(image, line, (target_panel_x + 10, y_pos), font, 0.5, color, 1)
        
        # Threat assessment (simulated)
        threat_level = "LOW"
        threat_color = hud_color
        
        if len(self.tracked_objects) > 3:
            threat_level = "MEDIUM"
            threat_color = warning_color
        elif len(self.tracked_objects) > 6:
            threat_level = "HIGH" 
            threat_color = critical_color
        
        cv2.putText(image, f"THREAT LEVEL: {threat_level}", (width // 2 - 100, height - 20), 
                   font, 0.7, threat_color, 2)
        
        # System mode indicators
        mode_indicators = []
        if self.thermal_noise_intensity > 0:
            mode_indicators.append("NOISE")
        if self.outline_glow:
            mode_indicators.append("GLOW")
        if self.motion_trails_enabled:
            mode_indicators.append("TRAILS")
        if self.target_tracking_enabled:
            mode_indicators.append("TRACK")
        
        if mode_indicators:
            mode_text = " | ".join(mode_indicators)
            cv2.putText(image, mode_text, (20, height - 20), font, 0.5, hud_color, 1)
        
        return image
    
    def process_frames_worker(self):
        """Worker thread for processing frames"""
        while self.processing_active:
            try:
                if not self.frame_queue.empty():
                    frame_data = self.frame_queue.get(timeout=0.1)
                    if frame_data is None:  # Shutdown signal
                        break
                    
                    color_image, depth_image, depth_frame = frame_data
                    
                    # Process frame
                    processed_result = self.process_frame(color_image, depth_image, depth_frame)
                    
                    # Put result in output queue
                    if not self.result_queue.full():
                        self.result_queue.put(processed_result)
                    
                    self.frame_queue.task_done()
                else:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker thread error: {e}")
    
    def process_frame(self, color_image, depth_image, depth_frame):
        """Process a single frame with all effects"""
        start_time = time.time()
        
        # Create thermal depth visualization
        thermal_image, depth_normalized = self.create_thermal_depth_image(depth_image, color_image)
        
        # Perform object segmentation
        segmentations = self.segment_objects(color_image)
        
        # Draw segmentation outlines
        result = self.draw_segmentation_outlines(thermal_image, segmentations, color_image)
        
        # Add Predator-style effects
        result = self.add_predator_effects(result)
        
        # Add HUD overlay
        result = self.add_hud_overlay(result)
        
        # Record total processing time
        total_time = time.time() - start_time
        self.processing_times.append(total_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        return result
    
    def create_control_panel(self):
        """Create real-time parameter adjustment panel"""
        if not TKINTER_AVAILABLE:
            print("⚠️  Control panel not available - tkinter required")
            return None
        
        def update_thermal_style():
            self.thermal_style = self.control_vars['thermal_style'].get()
            self.setup_thermal_visualization()
        
        def update_depth_range():
            max_depth = self.control_vars['max_depth'].get()
            self.depth_range = (self.depth_range[0], max_depth)
        
        def update_confidence():
            self.confidence_threshold = self.control_vars['confidence'].get()
        
        def toggle_effects():
            self.outline_glow = self.control_vars['glow'].get()
            self.motion_trails_enabled = self.control_vars['trails'].get()
            self.target_tracking_enabled = self.control_vars['tracking'].get()
        
        def update_noise():
            self.thermal_noise_intensity = self.control_vars['noise'].get()
        
        # Create control window
        control_window = tk.Tk()
        control_window.title("🎯 Predator Vision Controls")
        control_window.geometry("400x600")
        control_window.configure(bg='black')
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background='black', foreground='lime')
        style.configure('TFrame', background='black')
        
        # Main frame
        main_frame = ttk.Frame(control_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Thermal style selection
        ttk.Label(main_frame, text="🌡️ THERMAL STYLE", font=('Arial', 12, 'bold')).pack(pady=5)
        self.control_vars['thermal_style'] = tk.StringVar(value=self.thermal_style)
        thermal_combo = ttk.Combobox(main_frame, textvariable=self.control_vars['thermal_style'],
                                   values=['predator', 'flir', 'iron', 'rainbow', 'hot', 'plasma', 'viridis'],
                                   command=update_thermal_style, width=30)
        thermal_combo.pack(pady=5)
        thermal_combo.bind('<<ComboboxSelected>>', lambda e: update_thermal_style())
        
        # Depth range control
        ttk.Label(main_frame, text="📏 MAX DEPTH (meters)", font=('Arial', 10, 'bold')).pack(pady=(20,5))
        self.control_vars['max_depth'] = tk.DoubleVar(value=self.depth_range[1])
        depth_scale = ttk.Scale(main_frame, from_=1.0, to=15.0, 
                              variable=self.control_vars['max_depth'], orient='horizontal',
                              command=lambda x: update_depth_range())
        depth_scale.pack(pady=5, fill='x')
        
        # Confidence threshold
        ttk.Label(main_frame, text="🎯 DETECTION CONFIDENCE", font=('Arial', 10, 'bold')).pack(pady=(20,5))
        self.control_vars['confidence'] = tk.DoubleVar(value=self.confidence_threshold)
        conf_scale = ttk.Scale(main_frame, from_=0.1, to=1.0,
                             variable=self.control_vars['confidence'], orient='horizontal',
                             command=lambda x: update_confidence())
        conf_scale.pack(pady=5, fill='x')
        
        # Thermal noise control
        ttk.Label(main_frame, text="📡 THERMAL NOISE", font=('Arial', 10, 'bold')).pack(pady=(20,5))
        self.control_vars['noise'] = tk.DoubleVar(value=self.thermal_noise_intensity)
        noise_scale = ttk.Scale(main_frame, from_=0.0, to=0.5,
                              variable=self.control_vars['noise'], orient='horizontal',
                              command=lambda x: update_noise())
        noise_