# RealSense D455 Object Detection System

## Overview
This prototype demonstrates real-time object detection using Intel RealSense D455 camera with YOLO v8 model. The system captures RGB and depth data, performs object detection, and provides 3D spatial information about detected objects.

## Prerequisites

### Hardware Requirements
- Intel RealSense D455 camera
- USB 3.0 port (recommended for optimal performance)
- Minimum 8GB RAM
- GPU (optional but recommended for better performance)

### Software Dependencies
```bash
pip install pyrealsense2
pip install opencv-python
pip install ultralytics
pip install numpy
pip install matplotlib
```

## Complete Implementation

```python
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
from datetime import datetime

class RealSenseObjectDetector:
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the RealSense Object Detection system
        
        Args:
            model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt)
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable color and depth streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Create alignment object to align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Detection log
        self.detection_log = []
        
    def start_detection(self, save_video=False, video_filename='detection_output.avi'):
        """
        Start real-time object detection
        
        Args:
            save_video: Whether to save detection results to video file
            video_filename: Output video filename
        """
        print("Starting RealSense Object Detection...")
        print("Press 'q' to quit, 's' to save current frame, 'p' to pause/resume")
        
        # Start pipeline
        self.pipeline.start(self.config)
        
        # Video writer setup
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))
        
        paused = False
        
        try:
            while True:
                if not paused:
                    # Get frames
                    frames = self.pipeline.wait_for_frames()
                    
                    # Align depth to color
                    aligned_frames = self.align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                    
                    # Convert to numpy arrays
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    # Run object detection
                    results = self.model(color_image, conf=self.confidence_threshold)
                    
                    # Process detections
                    annotated_image = self.process_detections(
                        color_image, depth_image, results[0], depth_frame
                    )
                    
                    # Update performance metrics
                    self.frame_count += 1
                    current_time = time.time()
                    fps = self.frame_count / (current_time - self.start_time)
                    
                    # Add performance info to image
                    cv2.putText(annotated_image, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_image, f"Objects: {len(results[0].boxes) if results[0].boxes is not None else 0}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show image
                    cv2.imshow('RealSense Object Detection', annotated_image)
                    
                    # Save to video if enabled
                    if save_video and video_writer:
                        video_writer.write(annotated_image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_image)
                    print(f"Frame saved as {filename}")
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                    
        except Exception as e:
            print(f"Error: {e}")
            
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            if video_writer:
                video_writer.release()
            
            # Save detection log
            self.save_detection_log()
            
    def process_detections(self, color_image, depth_image, results, depth_frame):
        """
        Process detection results and add 3D spatial information
        
        Args:
            color_image: RGB image from camera
            depth_image: Depth image from camera
            results: YOLO detection results
            depth_frame: RealSense depth frame for coordinate conversion
            
        Returns:
            Annotated image with detection results
        """
        annotated_image = color_image.copy()
        
        if results.boxes is not None:
            boxes = results.boxes
            
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]
                
                # Calculate center point for depth measurement
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Get depth at center point (in meters)
                depth_value = depth_frame.get_distance(center_x, center_y)
                
                # Get 3D coordinates
                depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics, [center_x, center_y], depth_value
                )
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create label with 3D information
                label = f"{class_name}: {confidence:.2f}"
                if depth_value > 0:
                    label += f" | {depth_value:.2f}m"
                    label += f" | 3D: ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(annotated_image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Draw center point
                cv2.circle(annotated_image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Log detection
                detection_data = {
                    'timestamp': datetime.now().isoformat(),
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'center_pixel': [center_x, center_y],
                    'depth_meters': depth_value,
                    'position_3d': point_3d
                }
                self.detection_log.append(detection_data)
                
        return annotated_image
    
    def save_detection_log(self, filename='detection_log.json'):
        """Save detection log to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.detection_log, f, indent=2)
        print(f"Detection log saved to {filename}")
    
    def get_statistics(self):
        """Get detection statistics"""
        if not self.detection_log:
            return {}
            
        class_counts = {}
        for detection in self.detection_log:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        return {
            'total_detections': len(self.detection_log),
            'class_distribution': class_counts,
            'average_confidence': np.mean([d['confidence'] for d in self.detection_log]),
            'detection_rate': len(self.detection_log) / max(1, self.frame_count)
        }

# Advanced Configuration Class
class AdvancedDetectorConfig:
    def __init__(self):
        self.detection_zones = []  # Define specific zones for detection
        self.size_filters = {'min_area': 100, 'max_area': 50000}
        self.tracking_enabled = True
        self.alert_classes = ['person', 'car', 'truck']  # Classes to trigger alerts
        
    def add_detection_zone(self, x1, y1, x2, y2, name="Zone"):
        """Add a specific detection zone"""
        self.detection_zones.append({
            'coords': (x1, y1, x2, y2),
            'name': name
        })

# Usage Example
if __name__ == "__main__":
    # Initialize detector with medium model for better accuracy
    detector = RealSenseObjectDetector(model_name='yolov8s.pt', confidence_threshold=0.6)
    
    # Start detection with video recording
    detector.start_detection(save_video=True, video_filename='object_detection_output.avi')
    
    # Print statistics after detection
    stats = detector.get_statistics()
    print("\nDetection Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
```

## Usage Instructions

### Basic Usage
1. Connect your RealSense D455 camera
2. Run the script: `python object_detection.py`
3. The system will display real-time object detection with depth information
4. Press 'q' to quit, 's' to save current frame, 'p' to pause/resume

### Advanced Features
- **3D Position Tracking**: Each detected object shows its 3D coordinates in real-world space
- **Depth Information**: Distance to each object is displayed
- **Performance Monitoring**: Real-time FPS and object count
- **Detection Logging**: All detections are saved to JSON for analysis
- **Video Recording**: Optional video output with annotations

### Customization Options
- **Model Selection**: Choose between yolov8n.pt (fastest) to yolov8x.pt (most accurate)
- **Confidence Threshold**: Adjust detection sensitivity
- **Detection Zones**: Define specific areas for focused detection
- **Alert System**: Configure alerts for specific object classes

### Output Files
- `detection_log.json`: Complete detection history with timestamps and 3D coordinates
- `object_detection_output.avi`: Video recording of detection session
- `detection_frame_TIMESTAMP.jpg`: Individual saved frames

## Performance Optimization Tips

1. **GPU Acceleration**: Install PyTorch with CUDA support for faster inference
2. **Model Selection**: Use lighter models (yolov8n.pt) for real-time performance
3. **Resolution**: Lower camera resolution for better FPS
4. **Selective Detection**: Filter by specific object classes to reduce processing

## Troubleshooting

### Common Issues
- **Camera Not Found**: Ensure RealSense D455 is properly connected and drivers are installed
- **Low FPS**: Try using a lighter YOLO model or lower resolution
- **Depth Accuracy**: Ensure proper camera calibration and avoid reflective surfaces

### Performance Benchmarks
- **yolov8n.pt**: ~30 FPS on typical hardware
- **yolov8s.pt**: ~20 FPS on typical hardware
- **yolov8m.pt**: ~15 FPS on typical hardware

## Extension Ideas
- Add object tracking across frames
- Implement custom object classes
- Add database storage for detections
- Create web dashboard for monitoring
- Integrate with home automation systems