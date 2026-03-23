import pyrealsense2 as rs
import cv2
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
import sys

# Try importing advanced segmentation models
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available")

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  SAM not available")

try:
    from skimage import segmentation, measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️  scikit-image not available")

class RealSenseSegmentationDetector:
    def __init__(self, segmentation_method='yolo_seg', confidence_threshold=0.5):
        """
        Initialize segmentation-based object detection
        
        Args:
            segmentation_method: 'yolo_seg', 'sam', 'watershed', 'grabcut', 'contour'
            confidence_threshold: Minimum confidence for detections
        """
        self.segmentation_method = segmentation_method
        self.confidence_threshold = confidence_threshold
        self.using_realsense = False
        self.camera_source = None
        
        # Initialize segmentation model
        self.initialize_segmentation_model()
        
        # Initialize camera
        self.initialize_camera()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_log = []
        
        # Visualization settings
        self.colors = self.generate_colors(80)  # 80 COCO classes
        self.overlay_alpha = 0.6
        
    def generate_colors(self, num_classes):
        """Generate distinct colors for each class"""
        np.random.seed(42)  # For consistent colors
        colors = []
        for i in range(num_classes):
            color = (
                np.random.randint(50, 255),
                np.random.randint(50, 255), 
                np.random.randint(50, 255)
            )
            colors.append(color)
        return colors
    
    def initialize_segmentation_model(self):
        """Initialize the segmentation model based on method"""
        print(f"Initializing segmentation method: {self.segmentation_method}")
        
        if self.segmentation_method == 'yolo_seg':
            if YOLO_AVAILABLE:
                try:
                    # Use YOLOv8 segmentation model
                    self.model = YOLO('yolov8n-seg.pt')  # Downloads automatically
                    print("✓ YOLO segmentation model loaded")
                    return
                except Exception as e:
                    print(f"Failed to load YOLO segmentation: {e}")
            
            # Fallback to detection + contour method
            print("Falling back to contour-based segmentation")
            self.segmentation_method = 'contour'
            
        elif self.segmentation_method == 'sam':
            if SAM_AVAILABLE:
                try:
                    # Download SAM model if needed
                    model_path = self.download_sam_model()
                    sam = sam_model_registry["vit_b"](checkpoint=model_path)
                    self.sam_predictor = SamPredictor(sam)
                    
                    # Also need object detection for SAM prompts
                    if YOLO_AVAILABLE:
                        self.detection_model = YOLO('yolov8n.pt')
                    
                    print("✓ SAM model loaded")
                    return
                except Exception as e:
                    print(f"Failed to load SAM: {e}")
                    self.segmentation_method = 'contour'
            else:
                print("SAM not available, using contour method")
                self.segmentation_method = 'contour'
        
        # Traditional computer vision methods don't need model loading
        print(f"Using traditional CV method: {self.segmentation_method}")
    
    def download_sam_model(self):
        """Download SAM model if not present"""
        model_path = "sam_vit_b_01ec64.pth"
        if not Path(model_path).exists():
            print("Downloading SAM model...")
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, model_path)
            print("✓ SAM model downloaded")
        return model_path
    
    def initialize_camera(self):
        """Initialize camera (same as before)"""
        print("Initializing camera...")
        
        # Try RealSense first
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) > 0:
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                try:
                    self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    self.has_depth = True
                    self.align = rs.align(rs.stream.color)
                except:
                    self.has_depth = False
                
                # Test pipeline
                profile = self.pipeline.start(self.config)
                self.pipeline.stop()
                
                self.using_realsense = True
                print("✓ RealSense camera initialized")
                return
        except:
            pass
        
        # Fallback to webcam
        for i in range(4):
            self.camera_source = cv2.VideoCapture(i)
            if self.camera_source.isOpened():
                self.camera_source.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera_source.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"✓ Webcam initialized at index {i}")
                return
        
        raise RuntimeError("No camera available")
    
    def get_frame(self):
        """Get frame from camera"""
        if self.using_realsense:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    return None, None
                
                color_image = np.asanyarray(color_frame.get_data())
                
                depth_image = None
                if self.has_depth:
                    try:
                        aligned_frames = self.align.process(frames)
                        depth_frame = aligned_frames.get_depth_frame()
                        depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
                    except:
                        pass
                
                return color_image, depth_image
            except:
                return None, None
        else:
            ret, frame = self.camera_source.read()
            return frame if ret else None, None
    
    def segment_yolo(self, image):
        """YOLO segmentation"""
        results = self.model(image, conf=self.confidence_threshold)
        return results[0] if results else None
    
    def segment_sam(self, image):
        """SAM segmentation with YOLO prompts"""
        # First get object detections for prompts
        detections = self.detection_model(image, conf=self.confidence_threshold)
        
        if not detections or not detections[0].boxes:
            return []
        
        # Set image for SAM
        self.sam_predictor.set_image(image)
        
        segmentations = []
        boxes = detections[0].boxes
        
        for box in boxes:
            # Get bounding box as prompt
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            input_box = np.array([x1, y1, x2, y2])
            
            # Generate mask
            masks, scores, _ = self.sam_predictor.predict(
                box=input_box,
                multimask_output=False
            )
            
            if len(masks) > 0:
                segmentations.append({
                    'mask': masks[0],
                    'bbox': [x1, y1, x2, y2],
                    'confidence': box.conf[0].item(),
                    'class_id': int(box.cls[0].item()),
                    'class_name': self.detection_model.names[int(box.cls[0].item())]
                })
        
        return segmentations
    
    def segment_watershed(self, image):
        """Watershed segmentation"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Extract segments
        segmentations = []
        for label in np.unique(markers):
            if label <= 1:  # Skip background and borders
                continue
            
            mask = (markers == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 500:  # Minimum area
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    segmentations.append({
                        'mask': mask,
                        'contour': largest_contour,
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.8,
                        'class_name': 'object'
                    })
        
        return segmentations
    
    def segment_grabcut(self, image):
        """GrabCut segmentation on detected objects"""
        # First detect objects using background subtraction or edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        segmentations = []
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Minimum area
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Expand slightly for GrabCut
                margin = 10
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(image.shape[1] - x, w + 2*margin)
                h = min(image.shape[0] - y, h + 2*margin)
                
                # Initialize GrabCut
                mask = np.zeros(gray.shape[:2], np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Define rectangle
                rect = (x, y, w, h)
                
                try:
                    # Apply GrabCut
                    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
                    
                    # Create final mask
                    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                    
                    segmentations.append({
                        'mask': mask2 * 255,
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.7,
                        'class_name': 'object'
                    })
                except:
                    continue  # Skip if GrabCut fails
        
        return segmentations
    
    def segment_contour(self, image):
        """Simple contour-based segmentation"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        segmentations = []
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 500:  # Minimum area
                # Create mask from contour
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                segmentations.append({
                    'mask': mask,
                    'contour': contour,
                    'bbox': [x, y, x+w, y+h],
                    'confidence': 0.7,
                    'class_id': i,
                    'class_name': 'object'
                })
        
        return segmentations
    
    def apply_segmentation(self, image):
        """Apply the selected segmentation method"""
        if self.segmentation_method == 'yolo_seg':
            return self.segment_yolo(image)
        elif self.segmentation_method == 'sam':
            return self.segment_sam(image)
        elif self.segmentation_method == 'watershed':
            return self.segment_watershed(image)
        elif self.segmentation_method == 'grabcut':
            return self.segment_grabcut(image)
        elif self.segmentation_method == 'contour':
            return self.segment_contour(image)
        else:
            return []
    
    def draw_segmentation_masks(self, image, segmentations):
        """Draw segmentation masks on image"""
        overlay = image.copy()
        
        if self.segmentation_method == 'yolo_seg' and segmentations and segmentations.masks is not None:
            # YOLO segmentation format
            masks = segmentations.masks.data.cpu().numpy()
            boxes = segmentations.boxes
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Get class info
                class_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                class_name = self.model.names[class_id]
                
                # Apply colored mask
                color = self.colors[class_id % len(self.colors)]
                colored_mask = np.zeros_like(image)
                colored_mask[mask_binary == 1] = color
                
                # Blend with image
                overlay = cv2.addWeighted(overlay, 1, colored_mask, self.overlay_alpha, 0)
                
                # Draw contour
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
                
                # Add label
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(overlay, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        elif isinstance(segmentations, list):
            # Traditional CV methods format
            for i, seg in enumerate(segmentations):
                mask = seg['mask']
                confidence = seg.get('confidence', 0.0)
                class_name = seg.get('class_name', 'object')
                
                # Create colored mask
                color = self.colors[i % len(self.colors)]
                colored_mask = np.zeros_like(image)
                
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)
                
                colored_mask[mask > 0] = color
                
                # Blend with image
                overlay = cv2.addWeighted(overlay, 1, colored_mask, self.overlay_alpha, 0)
                
                # Draw contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(overlay, contours, -1, color, 2)
                    
                    # Add label
                    x, y, w, h = cv2.boundingRect(contours[0])
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(overlay, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return overlay
    
    def add_info_overlay(self, image):
        """Add system information overlay"""
        camera_type = "RealSense" if self.using_realsense else "Webcam"
        
        # Performance metrics
        current_time = time.time()
        fps = self.frame_count / (current_time - self.start_time) if self.frame_count > 0 else 0
        
        # Info text
        info_texts = [
            f"Camera: {camera_type}",
            f"Method: {self.segmentation_method.upper()}",
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}"
        ]
        
        # Draw info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        for i, text in enumerate(info_texts):
            y_pos = 30 + i * 25
            
            # Background
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            cv2.rectangle(image, (10, y_pos - 20), (10 + text_size[0] + 10, y_pos + 5), 
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(image, text, (10, y_pos), font, font_scale, (255, 255, 255), thickness)
    
    def start_detection(self, save_video=False, video_filename='segmentation_output.avi'):
        """Start segmentation detection"""
        print(f"\nStarting Segmentation Detection ({self.segmentation_method})...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'p' - Pause/Resume")
        print("  '1-5' - Switch segmentation method")
        print("  '+/-' - Adjust overlay transparency")
        
        # Start camera
        if self.using_realsense:
            self.pipeline.start(self.config)
        
        # Video writer
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))
        
        paused = False
        
        try:
            while True:
                if not paused:
                    # Get frame
                    color_image, depth_image = self.get_frame()
                    
                    if color_image is None:
                        continue
                    
                    # Apply segmentation
                    segmentations = self.apply_segmentation(color_image)
                    
                    # Draw segmentation masks
                    result_image = self.draw_segmentation_masks(color_image, segmentations)
                    
                    # Add info overlay
                    self.add_info_overlay(result_image)
                    
                    # Show image
                    cv2.imshow('Object Segmentation', result_image)
                    
                    # Save video
                    if save_video and video_writer:
                        video_writer.write(result_image)
                    
                    self.frame_count += 1
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"segmentation_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, result_image)
                    print(f"Frame saved: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('1'):
                    self.switch_method('yolo_seg')
                elif key == ord('2'):
                    self.switch_method('sam')
                elif key == ord('3'):
                    self.switch_method('watershed')
                elif key == ord('4'):
                    self.switch_method('grabcut')
                elif key == ord('5'):
                    self.switch_method('contour')
                elif key == ord('+') or key == ord('='):
                    self.overlay_alpha = min(1.0, self.overlay_alpha + 0.1)
                    print(f"Overlay alpha: {self.overlay_alpha:.1f}")
                elif key == ord('-'):
                    self.overlay_alpha = max(0.0, self.overlay_alpha - 0.1)
                    print(f"Overlay alpha: {self.overlay_alpha:.1f}")
                    
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Detection error: {e}")
        finally:
            # Cleanup
            if self.using_realsense:
                self.pipeline.stop()
            elif self.camera_source:
                self.camera_source.release()
            
            cv2.destroyAllWindows()
            if video_writer:
                video_writer.release()
    
    def switch_method(self, new_method):
        """Switch segmentation method on the fly"""
        print(f"Switching to: {new_method}")
        self.segmentation_method = new_method
        self.initialize_segmentation_model()

# Usage Examples
def demo_all_methods():
    """Demo all segmentation methods"""
    methods = ['yolo_seg', 'sam', 'watershed', 'grabcut', 'contour']
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"DEMONSTRATING: {method.upper()}")
        print('='*50)
        
        try:
            detector = RealSenseSegmentationDetector(
                segmentation_method=method,
                confidence_threshold=0.5
            )
            
            print(f"Press any key to start {method} demo...")
            input()
            
            detector.start_detection(
                save_video=True, 
                video_filename=f'{method}_segmentation.avi'
            )
            
        except Exception as e:
            print(f"Failed to demo {method}: {e}")
        
        print(f"Demo of {method} completed")

if __name__ == "__main__":
    print("RealSense Object Segmentation System")
    print("="*50)
    
    # Choose method
    print("Available segmentation methods:")
    print("1. YOLO Segmentation (yolo_seg) - Most accurate")
    print("2. SAM (sam) - Segment Anything Model")
    print("3. Watershed (watershed) - Traditional CV")
    print("4. GrabCut (grabcut) - Interactive segmentation")
    print("5. Contour (contour) - Edge-based segmentation")
    print("6. Demo all methods")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    method_map = {
        '1': 'yolo_seg',
        '2': 'sam', 
        '3': 'watershed',
        '4': 'grabcut',
        '5': 'contour'
    }
    
    if choice == '6':
        demo_all_methods()
    elif choice in method_map:
        method = method_map[choice]
        
        detector = RealSenseSegmentationDetector(
            segmentation_method=method,
            confidence_threshold=0.5
        )
        
        detector.start_detection(
            save_video=True,
            video_filename=f'{method}_segmentation_output.avi'
        )
    else:
        print("Invalid choice, using YOLO segmentation")
        detector = RealSenseSegmentationDetector(segmentation_method='yolo_seg')
        detector.start_detection(save_video=True)