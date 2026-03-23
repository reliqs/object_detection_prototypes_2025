import pyrealsense2 as rs
import cv2
import numpy as np
import mediapipe as mp
import time
import json
from datetime import datetime
from collections import deque
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

class RealSenseActivityDetector:
    def __init__(self, history_length=30, model_path=None):
        """
        Initialize the RealSense Activity Detection system
        
        Args:
            history_length: Number of frames to keep for activity analysis
            model_path: Path to pre-trained activity classification model
        """
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable color and depth streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Create alignment object
        self.align = rs.align(rs.stream.color)
        
        # Activity detection parameters
        self.history_length = history_length
        self.pose_history = deque(maxlen=history_length)
        self.activity_history = deque(maxlen=100)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Activity classification
        self.activity_classifier = None
        self.scaler = StandardScaler()
        if model_path:
            self.load_model(model_path)
        else:
            self.initialize_default_classifier()
        
        # Activity definitions
        self.activity_labels = {
            0: 'Standing',
            1: 'Walking',
            2: 'Sitting',
            3: 'Waving',
            4: 'Jumping',
            5: 'Unknown'
        }
        
        # Detection log
        self.activity_log = []
        
    def initialize_default_classifier(self):
        """Initialize a basic rule-based activity classifier"""
        self.activity_classifier = None  # Will use rule-based classification
        
    def load_model(self, model_path):
        """Load pre-trained activity classification model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.activity_classifier = model_data['classifier']
                self.scaler = model_data['scaler']
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.initialize_default_classifier()
    
    def extract_pose_features(self, landmarks):
        """
        Extract features from pose landmarks for activity classification
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Feature vector for classification
        """
        if not landmarks:
            return np.zeros(66)  # 33 landmarks * 2 (x, y) coordinates
        
        features = []
        
        # Extract x, y coordinates (normalized)
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y])
        
        return np.array(features)
    
    def calculate_movement_features(self, current_pose, previous_poses):
        """
        Calculate movement-based features for activity classification
        
        Args:
            current_pose: Current pose landmarks
            previous_poses: Historical pose data
            
        Returns:
            Movement feature vector
        """
        if len(previous_poses) < 2:
            return np.zeros(10)
        
        features = []
        
        # Calculate key point velocities
        key_points = [
            mp.solutions.pose.PoseLandmark.NOSE,
            mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        for point in key_points:
            if len(previous_poses) >= 2:
                curr_x = current_pose.landmark[point].x
                curr_y = current_pose.landmark[point].y
                prev_x = previous_poses[-1].landmark[point].x
                prev_y = previous_poses[-1].landmark[point].y
                
                velocity = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                features.append(velocity)
            else:
                features.append(0.0)
        
        # Calculate overall body movement
        if len(previous_poses) >= 5:
            recent_poses = previous_poses[-5:]
            movement_variance = self.calculate_pose_variance(recent_poses)
            features.extend(movement_variance)
        else:
            features.extend([0.0] * 5)
        
        return np.array(features)
    
    def calculate_pose_variance(self, poses):
        """Calculate variance in pose over time"""
        if len(poses) < 2:
            return [0.0] * 5
        
        positions = []
        for pose in poses:
            pose_features = self.extract_pose_features(pose)
            positions.append(pose_features)
        
        positions = np.array(positions)
        variances = np.var(positions, axis=0)
        
        # Return summary statistics
        return [
            np.mean(variances),
            np.std(variances),
            np.max(variances),
            np.min(variances),
            np.median(variances)
        ]
    
    def classify_activity_rule_based(self, current_pose, movement_features):
        """
        Rule-based activity classification
        
        Args:
            current_pose: Current pose landmarks
            movement_features: Movement-based features
            
        Returns:
            Activity classification
        """
        if not current_pose:
            return 5  # Unknown
        
        # Calculate key measurements
        nose = current_pose.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        left_wrist = current_pose.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        right_wrist = current_pose.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        left_ankle = current_pose.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = current_pose.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
        left_hip = current_pose.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        right_hip = current_pose.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate average movement
        avg_movement = np.mean(movement_features[:5])
        
        # Rule-based classification
        if avg_movement > 0.05:
            # High movement detected
            
            # Check for jumping (feet off ground relative to hips)
            hip_y = (left_hip.y + right_hip.y) / 2
            ankle_y = (left_ankle.y + right_ankle.y) / 2
            
            if hip_y > ankle_y - 0.1:  # Feet close to hips level
                return 4  # Jumping
            
            # Check for waving (wrist above shoulder level)
            if (left_wrist.y < nose.y - 0.1) or (right_wrist.y < nose.y - 0.1):
                return 3  # Waving
            
            # Default to walking for moderate movement
            return 1  # Walking
        
        else:
            # Low movement detected
            
            # Check if sitting (hips close to ankle level)
            hip_y = (left_hip.y + right_hip.y) / 2
            ankle_y = (left_ankle.y + right_ankle.y) / 2
            
            if abs(hip_y - ankle_y) < 0.2:
                return 2  # Sitting
            
            # Default to standing
            return 0  # Standing
    
    def start_detection(self, save_video=False, video_filename='activity_detection.avi'):
        """
        Start real-time activity detection
        
        Args:
            save_video: Whether to save detection results to video file
            video_filename: Output video filename
        """
        print("Starting RealSense Activity Detection...")
        print("Press 'q' to quit, 's' to save current frame, 'p' to pause/resume")
        print("Press 'r' to reset activity history, 'c' to calibrate")
        
        # Start pipeline
        self.pipeline.start(self.config)
        
        # Video writer setup
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))
        
        paused = False
        current_activity = "Unknown"
        activity_confidence = 0.0
        
        try:
            while True:
                if not paused:
                    # Get frames
                    frames = self.pipeline.wait_for_frames()
                    
                    # Align depth to color
                    aligned_frames = self.align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if not color_frame:
                        continue
                    
                    # Convert to numpy arrays
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Process pose detection
                    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(rgb_image)
                    
                    # Draw pose landmarks
                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            color_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                        )
                        
                        # Store pose history
                        self.pose_history.append(results.pose_landmarks)
                        
                        # Extract features and classify activity
                        pose_features = self.extract_pose_features(results.pose_landmarks)
                        movement_features = self.calculate_movement_features(
                            results.pose_landmarks, list(self.pose_history)[:-1]
                        )
                        
                        # Classify activity
                        if self.activity_classifier:
                            # Use ML classifier
                            combined_features = np.concatenate([pose_features, movement_features])
                            combined_features = self.scaler.transform([combined_features])
                            activity_pred = self.activity_classifier.predict(combined_features)[0]
                            activity_confidence = max(self.activity_classifier.predict_proba(combined_features)[0])
                        else:
                            # Use rule-based classifier
                            activity_pred = self.classify_activity_rule_based(
                                results.pose_landmarks, movement_features
                            )
                            activity_confidence = 0.8  # Default confidence for rule-based
                        
                        current_activity = self.activity_labels.get(activity_pred, "Unknown")
                        
                        # Store activity history
                        self.activity_history.append({
                            'timestamp': time.time(),
                            'activity': current_activity,
                            'confidence': activity_confidence
                        })
                        
                        # Log activity
                        self.log_activity(current_activity, activity_confidence, 
                                        pose_features, movement_features)
                    
                    # Add information overlay
                    self.add_info_overlay(color_image, current_activity, activity_confidence)
                    
                    # Show image
                    cv2.imshow('RealSense Activity Detection', color_image)
                    
                    # Save to video if enabled
                    if save_video and video_writer:
                        video_writer.write(color_image)
                    
                    # Update frame count
                    self.frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"activity_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, color_image)
                    print(f"Frame saved as {filename}")
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('r'):
                    self.pose_history.clear()
                    self.activity_history.clear()
                    print("Activity history reset")
                elif key == ord('c'):
                    print("Calibration mode - stand still for 3 seconds")
                    # Implement calibration logic here
                    
        except Exception as e:
            print(f"Error: {e}")
            
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            if video_writer:
                video_writer.release()
            
            # Save activity log
            self.save_activity_log()
            
    def add_info_overlay(self, image, activity, confidence):
        """Add information overlay to the image"""
        height, width = image.shape[:2]
        
        # Create semi-transparent overlay
        overlay = image.copy()
        
        # Activity information
        activity_text = f"Activity: {activity}"
        confidence_text = f"Confidence: {confidence:.2f}"
        
        # Performance metrics
        current_time = time.time()
        fps = self.frame_count / (current_time - self.start_time) if self.frame_count > 0 else 0
        fps_text = f"FPS: {fps:.1f}"
        
        # Activity history summary
        if self.activity_history:
            recent_activities = list(self.activity_history)[-10:]
            activity_counts = {}
            for act in recent_activities:
                activity_counts[act['activity']] = activity_counts.get(act['activity'], 0) + 1
            
            most_common = max(activity_counts, key=activity_counts.get)
            trend_text = f"Recent: {most_common}"
        else:
            trend_text = "Recent: None"
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        texts = [activity_text, confidence_text, fps_text, trend_text]
        y_offset = 30
        
        for i, text in enumerate(texts):                         
            y_pos = y_offset + i * 30
            
            # Text background
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            cv2.rectangle(overlay, (10, y_pos - 20), (10 + text_size[0], y_pos + 5), 
                         (0, 0, 0), -1)
            
            # Text
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(overlay, text, (10, y_pos), font, font_scale, color, thickness)
        
        # Blend overlay with original image
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Activity timeline (bottom of screen)
        self.draw_activity_timeline(image)
    
    def draw_activity_timeline(self, image):
        """Draw activity timeline at bottom of screen"""
        height, width = image.shape[:2]
        
        if len(self.activity_history) < 2:
            return
        
        # Timeline parameters
        timeline_height = 40
        timeline_y = height - timeline_height
        
        # Activity colors
        activity_colors = {
            'Standing': (0, 255, 0),     # Green
            'Walking': (255, 0, 0),      # Blue
            'Sitting': (0, 0, 255),      # Red
            'Waving': (255, 255, 0),     # Cyan
            'Jumping': (255, 0, 255),    # Magenta
            'Unknown': (128, 128, 128)   # Gray
        }
        
        # Draw timeline background
        cv2.rectangle(image, (0, timeline_y), (width, height), (0, 0, 0), -1)
        
        # Draw activity segments
        recent_activities = list(self.activity_history)[-50:]  # Last 50 activities
        if recent_activities:
            segment_width = width / len(recent_activities)
            
            for i, activity_data in enumerate(recent_activities):
                activity = activity_data['activity']
                color = activity_colors.get(activity, (128, 128, 128))
                
                x1 = int(i * segment_width)
                x2 = int((i + 1) * segment_width)
                
                cv2.rectangle(image, (x1, timeline_y), (x2, height), color, -1)
        
        # Add timeline labels
        cv2.putText(image, "Activity Timeline", (10, timeline_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def log_activity(self, activity, confidence, pose_features, movement_features):
        """Log activity detection data"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity': activity,
            'confidence': confidence,
            'pose_features': pose_features.tolist(),
            'movement_features': movement_features.tolist()
        }
        self.activity_log.append(log_entry)
    
    def save_activity_log(self, filename='activity_log.json'):
        """Save activity log to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.activity_log, f, indent=2)
        print(f"Activity log saved to {filename}")
    
    def get_activity_statistics(self):
        """Get activity detection statistics"""
        if not self.activity_log:
            return {}
        
        activities = [entry['activity'] for entry in self.activity_log]
        activity_counts = {}
        for activity in activities:
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        # Calculate time spent in each activity
        activity_durations = {}
        for activity in set(activities):
            activity_durations[activity] = activity_counts[activity] / max(1, len(activities))
        
        return {
            'total_detections': len(self.activity_log),
            'activity_distribution': activity_counts,
            'activity_durations': activity_durations,
            'average_confidence': np.mean([entry['confidence'] for entry in self.activity_log]),
            'detection_rate': len(self.activity_log) / max(1, self.frame_count)
        }
    
    def train_custom_classifier(self, training_data_path):
        """
        Train a custom activity classifier from labeled data
        
        Args:
            training_data_path: Path to training data CSV file
        """
        try:
            # Load training data
            df = pd.read_csv(training_data_path)
            
            # Prepare features and labels
            feature_columns = [col for col in df.columns if col.startswith('pose_') or col.startswith('movement_')]
            X = df[feature_columns].values
            y = df['activity_label'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train classifier
            self.activity_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            self.activity_classifier.fit(X_scaled, y)
            
            # Save trained model
            model_data = {
                'classifier': self.activity_classifier,
                'scaler': self.scaler,
                'feature_columns': feature_columns
            }
            
            with open('activity_classifier.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            print("Custom classifier trained and saved!")
            
        except Exception as e:
            print(f"Error training classifier: {e}")
    
    def export_training_data(self, filename='training_data.csv'):
        """Export logged data for training purposes"""
        if not self.activity_log:
            print("No data to export")
            return
        
        # Convert log to DataFrame
        rows = []
        for entry in self.activity_log:
            row = {
                'timestamp': entry['timestamp'],
                'activity_label': entry['activity'],
                'confidence': entry['confidence']
            }
            
            # Add pose features
            for i, feature in enumerate(entry['pose_features']):
                row[f'pose_{i}'] = feature
            
            # Add movement features
            for i, feature in enumerate(entry['movement_features']):
                row[f'movement_{i}'] = feature
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Training data exported to {filename}")

# Advanced Configuration and Analysis
class ActivityAnalyzer:
    def __init__(self, log_file='activity_log.json'):
        """Initialize activity analyzer with log file"""
        self.log_file = log_file
        self.load_log()
    
    def load_log(self):
        """Load activity log from file"""
        try:
            with open(self.log_file, 'r') as f:
                self.activity_log = json.load(f)
            print(f"Loaded {len(self.activity_log)} activity records")
        except FileNotFoundError:
            print(f"Log file {self.log_file} not found")
            self.activity_log = []
    
    def analyze_activity_patterns(self):
        """Analyze activity patterns over time"""
        if not self.activity_log:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.activity_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based analysis
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Activity distribution by hour
        hourly_activities = df.groupby(['hour', 'activity']).size().unstack(fill_value=0)
        
        # Activity transitions
        transitions = []
        for i in range(1, len(df)):
            if df.iloc[i-1]['activity'] != df.iloc[i]['activity']:
                transitions.append({
                    'from': df.iloc[i-1]['activity'],
                    'to': df.iloc[i]['activity'],
                    'timestamp': df.iloc[i]['timestamp']
                })
        
        print("\nActivity Analysis Results:")
        print("=" * 50)
        
        # Most common activities
        activity_counts = df['activity'].value_counts()
        print("\nMost Common Activities:")
        for activity, count in activity_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {activity}: {count} ({percentage:.1f}%)")
        
        # Activity transitions
        if transitions:
            transition_df = pd.DataFrame(transitions)
            transition_counts = transition_df.groupby(['from', 'to']).size().sort_values(ascending=False)
            print(f"\nMost Common Activity Transitions:")
            for (from_act, to_act), count in transition_counts.head(5).items():
                print(f"  {from_act} → {to_act}: {count}")
        
        # Peak activity hours
        peak_hours = df.groupby('hour').size().sort_values(ascending=False)
        print(f"\nPeak Activity Hours:")
        for hour, count in peak_hours.head(3).items():
            print(f"  {hour}:00 - {count} activities")
        
        return {
            'activity_counts': activity_counts.to_dict(),
            'hourly_distribution': hourly_activities.to_dict(),
            'transitions': transitions,
            'peak_hours': peak_hours.to_dict()
        }
    
    def generate_report(self, output_file='activity_report.html'):
        """Generate comprehensive activity report"""
        analysis = self.analyze_activity_patterns()
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Activity Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .activity-item {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RealSense Activity Detection Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Activities Detected: {len(self.activity_log)}</p>
            </div>
            
            <div class="section">
                <h2>Activity Distribution</h2>
                <table>
                    <tr><th>Activity</th><th>Count</th><th>Percentage</th></tr>
        """
        
        if analysis and 'activity_counts' in analysis:
            total = sum(analysis['activity_counts'].values())
            for activity, count in analysis['activity_counts'].items():
                percentage = (count / total) * 100
                html_report += f"<tr><td>{activity}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_report += """
                </table>
            </div>
            
            <div class="section">
                <h2>Usage Instructions</h2>
                <ul>
                    <li>Ensure RealSense D455 camera is connected</li>
                    <li>Run the detection script in a well-lit environment</li>
                    <li>Position yourself within 1-3 meters of the camera</li>
                    <li>Perform activities naturally for best detection accuracy</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_report)
        
        print(f"Report generated: {output_file}")

# Usage Example and Testing
if __name__ == "__main__":
    # Initialize activity detector
    detector = RealSenseActivityDetector(history_length=30)
    
    # Start activity detection
    detector.start_detection(save_video=True, video_filename='activity_detection_output.avi')
    
    # Print statistics after detection
    stats = detector.get_activity_statistics()
    print("\nActivity Detection Statistics:")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Export training data for future model improvement
    detector.export_training_data('activity_training_data.csv')
    
    # Analyze patterns
    analyzer = ActivityAnalyzer('activity_log.json')
    analyzer.analyze_activity_patterns()
    analyzer.generate_report('activity_report.html')