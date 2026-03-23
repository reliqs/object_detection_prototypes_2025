import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
import threading
import os
from collections import deque
from enum import Enum
import json

class ScanState(Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"

class RealSenseSlam3DScanner:
    def __init__(self):
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams with optimal settings for SLAM
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        self.depth_intrinsics = depth_profile.get_intrinsics()
        self.color_intrinsics = color_profile.get_intrinsics()
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
        # SLAM components
        self.rgbd_frames = []
        self.poses = []
        self.volume = None
        self.mesh = None
        
        # State management
        self.state = ScanState.IDLE
        self.frame_count = 0
        self.start_time = None
        
        # SLAM parameters
        self.voxel_size = 0.01  # 1cm voxels
        self.max_depth = 3.0    # 3 meter max depth
        self.depth_scale = 1000.0  # RealSense depth scale
        
        # Visual odometry setup
        self.prev_rgbd = None
        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]
        
        # Reconstruction volume
        self.volume_bounds = [-2, 2, -2, 2, -2, 2]  # 4x4x4 meter volume
        
        # Export formats
        self.export_formats = {
            'PLY': '.ply',
            'OBJ': '.obj', 
            'STL': '.stl',
            'GLTF': '.gltf',
            'FBX': '.fbx'
        }
        
        print("SLAM 3D Scanner initialized")
        print("Camera intrinsics loaded")
        
    def create_rgbd_image(self, color_image, depth_image):
        """Create Open3D RGBD image from camera frames"""
        # Convert color to RGB
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_image.astype(np.uint16))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=False
        )
        
        return rgbd
    
    def get_camera_intrinsics(self):
        """Get Open3D camera intrinsics"""
        return o3d.camera.PinholeCameraIntrinsic(
            width=848,
            height=480,
            fx=self.color_intrinsics.fx,
            fy=self.color_intrinsics.fy,
            cx=self.color_intrinsics.ppx,
            cy=self.color_intrinsics.ppy
        )
    
    def estimate_pose(self, current_rgbd, prev_rgbd, prev_pose):
        """Estimate camera pose using visual odometry"""
        if prev_rgbd is None:
            return np.eye(4)
        
        intrinsic = self.get_camera_intrinsics()
        
        # Use Open3D's RGBD odometry
        option = o3d.pipelines.odometry.OdometryOption()
        option.max_depth_diff = 0.07
        option.min_depth = 0.1
        option.max_depth = self.max_depth
        
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            current_rgbd, prev_rgbd, intrinsic,
            odo_init=np.eye(4),
            jacobian_method=o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option=option
        )
        
        if success:
            return prev_pose @ trans
        else:
            return prev_pose
    
    def start_recording(self):
        """Start SLAM recording"""
        if self.state != ScanState.IDLE:
            return False
            
        print("Starting SLAM recording...")
        self.state = ScanState.RECORDING
        self.rgbd_frames = []
        self.poses = []
        self.frame_count = 0
        self.start_time = time.time()
        self.prev_rgbd = None
        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]
        
        return True
    
    def stop_recording(self):
        """Stop recording and process the data"""
        if self.state != ScanState.RECORDING:
            return False
            
        print("Stopping recording...")
        print(f"Captured {len(self.rgbd_frames)} frames in {time.time() - self.start_time:.1f} seconds")
        
        self.state = ScanState.PROCESSING
        
        # Start reconstruction in background thread
        threading.Thread(target=self.reconstruct_scene, daemon=True).start()
        
        return True
    
    def reconstruct_scene(self):
        """Perform SLAM reconstruction"""
        try:
            print("Starting SLAM reconstruction...")
            
            if len(self.rgbd_frames) < 2:
                print("Not enough frames for reconstruction")
                self.state = ScanState.IDLE
                return
            
            intrinsic = self.get_camera_intrinsics()
            
            # Create volume for TSDF integration
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.voxel_size,
                sdf_trunc=0.04,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
            
            print(f"Integrating {len(self.rgbd_frames)} frames...")
            
            # Integrate all frames into the volume
            for i, (rgbd, pose) in enumerate(zip(self.rgbd_frames, self.poses)):
                volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
                
                if i % 10 == 0:
                    print(f"Integrated frame {i+1}/{len(self.rgbd_frames)}")
            
            print("Extracting mesh...")
            # Extract triangle mesh
            self.mesh = volume.extract_triangle_mesh()
            self.mesh.compute_vertex_normals()
            
            # Clean up the mesh
            print("Cleaning mesh...")
            self.mesh.remove_degenerate_triangles()
            self.mesh.remove_duplicated_triangles()
            self.mesh.remove_duplicated_vertices()
            self.mesh.remove_non_manifold_edges()
            
            # Smooth the mesh
            self.mesh = self.mesh.filter_smooth_simple(number_of_iterations=1)
            
            print(f"Reconstruction complete! Mesh has {len(self.mesh.vertices)} vertices and {len(self.mesh.triangles)} triangles")
            
            self.state = ScanState.COMPLETE
            
        except Exception as e:
            print(f"Reconstruction failed: {e}")
            self.state = ScanState.IDLE
    
    def process_frame(self):
        """Process a single frame during recording"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
            
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        if self.state == ScanState.RECORDING:
            # Create RGBD image
            rgbd = self.create_rgbd_image(color_image, depth_image)
            
            # Estimate pose using visual odometry
            self.current_pose = self.estimate_pose(rgbd, self.prev_rgbd, self.current_pose)
            
            # Store frame and pose every few frames to avoid too much data
            if self.frame_count % 3 == 0:  # Every 3rd frame
                self.rgbd_frames.append(rgbd)
                self.poses.append(self.current_pose.copy())
                self.trajectory.append(self.current_pose.copy())
            
            self.prev_rgbd = rgbd
            self.frame_count += 1
        
        return color_image, depth_image
    
    def export_model(self, filename, format_type):
        """Export the reconstructed model in specified format"""
        if self.state != ScanState.COMPLETE or self.mesh is None:
            print("No model available for export")
            return False
        
        try:
            base_name = os.path.splitext(filename)[0]
            
            if format_type.upper() == 'PLY':
                filepath = f"{base_name}.ply"
                o3d.io.write_triangle_mesh(filepath, self.mesh)
                
            elif format_type.upper() == 'OBJ':
                filepath = f"{base_name}.obj"
                o3d.io.write_triangle_mesh(filepath, self.mesh)
                
            elif format_type.upper() == 'STL':
                filepath = f"{base_name}.stl"
                o3d.io.write_triangle_mesh(filepath, self.mesh)
                
            elif format_type.upper() == 'GLTF':
                filepath = f"{base_name}.gltf"
                o3d.io.write_triangle_mesh(filepath, self.mesh)
                
            elif format_type.upper() == 'FBX':
                # For FBX, we'll use PLY as intermediate and note the limitation
                ply_path = f"{base_name}.ply"
                o3d.io.write_triangle_mesh(ply_path, self.mesh)
                print(f"Note: FBX export requires external conversion. PLY saved as {ply_path}")
                print("Use Blender or other 3D software to convert PLY to FBX")
                return True
                
            else:
                print(f"Unsupported format: {format_type}")
                return False
            
            print(f"Model exported successfully as {filepath}")
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def preview_3d(self):
        """Show 3D preview of the reconstructed model"""
        if self.state != ScanState.COMPLETE or self.mesh is None:
            print("No model available for preview")
            return
        
        print("Opening 3D preview...")
        vis = o3d.visualization.Visualizer()
        vis.create_window("3D Model Preview", width=1024, height=768)
        vis.add_geometry(self.mesh)
        
        # Set view to better show the model
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        
        vis.run()
        vis.destroy_window()
    
    def get_scan_info(self):
        """Get information about the current scan"""
        info = {
            'state': self.state.value,
            'frames_captured': len(self.rgbd_frames),
            'poses_estimated': len(self.poses),
            'recording_time': time.time() - self.start_time if self.start_time else 0,
            'mesh_vertices': len(self.mesh.vertices) if self.mesh else 0,
            'mesh_triangles': len(self.mesh.triangles) if self.mesh else 0
        }
        return info
    
    def run_interface(self):
        """Main interface loop"""
        print("\n=== RealSense D455 SLAM 3D Scanner ===")
        print("Controls:")
        print("  r - Start/Stop recording")
        print("  p - Preview 3D model") 
        print("  e - Export model")
        print("  i - Show scan info")
        print("  q - Quit")
        print("=====================================\n")
        
        try:
            while True:
                color_image, depth_image = self.process_frame()
                
                if color_image is not None:
                    # Create display image with UI
                    display_image = color_image.copy()
                    
                    # Status overlay
                    info = self.get_scan_info()
                    status_color = {
                        ScanState.IDLE: (100, 100, 100),
                        ScanState.RECORDING: (0, 0, 255),
                        ScanState.PROCESSING: (0, 165, 255),
                        ScanState.COMPLETE: (0, 255, 0)
                    }[self.state]
                    
                    cv2.putText(display_image, f"Status: {self.state.value}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                    
                    if self.state == ScanState.RECORDING:
                        cv2.putText(display_image, f"Frames: {info['frames_captured']}", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_image, f"Time: {info['recording_time']:.1f}s", (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    elif self.state == ScanState.COMPLETE:
                        cv2.putText(display_image, f"Vertices: {info['mesh_vertices']}", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_image, f"Triangles: {info['mesh_triangles']}", (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Controls help
                    cv2.putText(display_image, "r=record p=preview e=export i=info q=quit", 
                               (10, display_image.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    cv2.imshow("SLAM 3D Scanner", display_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                    
                elif key == ord('r'):
                    if self.state == ScanState.IDLE:
                        self.start_recording()
                    elif self.state == ScanState.RECORDING:
                        self.stop_recording()
                        
                elif key == ord('p'):
                    self.preview_3d()
                    
                elif key == ord('e'):
                    if self.state == ScanState.COMPLETE:
                        self.export_menu()
                    else:
                        print("No model available for export")
                        
                elif key == ord('i'):
                    info = self.get_scan_info()
                    print(f"\nScan Information:")
                    for k, v in info.items():
                        print(f"  {k}: {v}")
                    print()
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        finally:
            self.cleanup()
    
    def export_menu(self):
        """Interactive export menu"""
        print("\n=== Export Menu ===")
        print("Available formats:")
        for i, (name, ext) in enumerate(self.export_formats.items(), 1):
            print(f"  {i}. {name} ({ext})")
        
        try:
            choice = input("Select format (1-5): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(self.export_formats):
                format_name = list(self.export_formats.keys())[int(choice) - 1]
                
                filename = input("Enter filename (without extension): ").strip()
                if filename:
                    self.export_model(filename, format_name)
                else:
                    print("Export cancelled")
            else:
                print("Invalid choice")
        except KeyboardInterrupt:
            print("Export cancelled")
    
    def cleanup(self):
        """Clean up resources"""
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print("Scanner shut down successfully")

def main():
    """Main application entry point"""
    try:
        scanner = RealSenseSlam3DScanner()
        scanner.run_interface()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your RealSense D455 is connected and drivers are installed")

if __name__ == "__main__":
    main()
