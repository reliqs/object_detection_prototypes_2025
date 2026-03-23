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
    def __init__(self, simple_mode=False):
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Simple mode uses basic point cloud accumulation instead of full SLAM
        self.simple_mode = simple_mode
        
        # First, check if any RealSense devices are connected
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found")
        
        device = devices[0]
        print(f"Found device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial number: {device.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware version: {device.get_info(rs.camera_info.firmware_version)}")
        
        if simple_mode:
            print("Running in Simple Mode (basic point cloud accumulation)")
        else:
            print("Running in SLAM Mode (advanced reconstruction)")
        
        # Get available streams and find best configuration
        available_streams = self._get_available_streams(device)
        depth_config, color_config = self._select_best_streams(available_streams)
        
        if depth_config is None or color_config is None:
            raise RuntimeError("Could not find compatible depth and color streams")
        
        # Configure streams with found settings
        self.config.enable_stream(rs.stream.depth, 
                                 depth_config['width'], depth_config['height'], 
                                 depth_config['format'], depth_config['fps'])
        self.config.enable_stream(rs.stream.color, 
                                 color_config['width'], color_config['height'], 
                                 color_config['format'], color_config['fps'])
        
        print(f"Configured depth: {depth_config['width']}x{depth_config['height']} @ {depth_config['fps']}fps")
        print(f"Configured color: {color_config['width']}x{color_config['height']} @ {color_config['fps']}fps")
        
        # Start pipeline
        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            print(f"Failed to start pipeline: {e}")
            # Try with default configuration as fallback
            print("Trying with default configuration...")
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        self.depth_intrinsics = depth_profile.get_intrinsics()
        self.color_intrinsics = color_profile.get_intrinsics()
        
        # Store actual resolution for later use
        self.width = self.color_intrinsics.width
        self.height = self.color_intrinsics.height
        
        print(f"Camera intrinsics loaded: {self.width}x{self.height}")
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
        # SLAM components
        if simple_mode:
            # Simple mode: just accumulate point clouds
            self.accumulated_pointcloud = o3d.geometry.PointCloud()
        else:
            # SLAM mode: full reconstruction pipeline
            self.rgbd_frames = []
            self.poses = []
            self.volume = None
            self.mesh = None
            
            # Visual odometry setup
            self.prev_rgbd = None
            self.current_pose = np.eye(4)
            self.trajectory = [self.current_pose.copy()]
        
        # State management
        self.state = ScanState.IDLE
        self.frame_count = 0
        self.start_time = None
        
        # SLAM parameters
        self.voxel_size = 0.01  # 1cm voxels
        self.max_depth = 3.0    # 3 meter max depth
        self.depth_scale = 1000.0  # RealSense depth scale
        
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
        
        print("Scanner initialized successfully")
    
    def _get_available_streams(self, device):
        """Get all available stream configurations"""
        available_streams = {'depth': [], 'color': []}
        
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == rs.stream.depth:
                    vp = profile.as_video_stream_profile()
                    available_streams['depth'].append({
                        'width': vp.width(),
                        'height': vp.height(),
                        'format': vp.format(),
                        'fps': vp.fps()
                    })
                elif profile.stream_type() == rs.stream.color:
                    vp = profile.as_video_stream_profile()
                    available_streams['color'].append({
                        'width': vp.width(),
                        'height': vp.height(),
                        'format': vp.format(),
                        'fps': vp.fps()
                    })
        
        return available_streams
    
    def _select_best_streams(self, available_streams):
        """Select the best depth and color stream configuration"""
        # Preferred configurations in order of preference
        preferred_configs = [
            {'width': 848, 'height': 480, 'fps': 30},
            {'width': 640, 'height': 480, 'fps': 30},
            {'width': 424, 'height': 240, 'fps': 30},
            {'width': 848, 'height': 480, 'fps': 15},
            {'width': 640, 'height': 480, 'fps': 15},
        ]
        
        depth_config = None
        color_config = None
        
        # Try to find matching depth and color streams
        for pref in preferred_configs:
            # Find depth stream
            for d_stream in available_streams['depth']:
                if (d_stream['width'] == pref['width'] and 
                    d_stream['height'] == pref['height'] and 
                    d_stream['fps'] == pref['fps'] and
                    d_stream['format'] == rs.format.z16):
                    
                    # Find matching color stream
                    for c_stream in available_streams['color']:
                        if (c_stream['width'] == pref['width'] and 
                            c_stream['height'] == pref['height'] and 
                            c_stream['fps'] == pref['fps'] and
                            c_stream['format'] == rs.format.bgr8):
                            
                            depth_config = d_stream
                            color_config = c_stream
                            break
                    
                    if depth_config and color_config:
                        break
            
            if depth_config and color_config:
                break
        
        # If no perfect match, try to find any compatible streams
        if not depth_config or not color_config:
            print("No perfect stream match found, trying compatible streams...")
            
            # Find any depth stream with z16 format
            for d_stream in available_streams['depth']:
                if d_stream['format'] == rs.format.z16:
                    depth_config = d_stream
                    break
            
            # Find any color stream with bgr8 format
            for c_stream in available_streams['color']:
                if c_stream['format'] == rs.format.bgr8:
                    color_config = c_stream
                    break
        
        return depth_config, color_config
        
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
            width=self.width,
            height=self.height,
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
        
        try:
            # Try newer Open3D API first
            option = o3d.pipelines.odometry.OdometryOption()
            
            # Set parameters that are available in current Open3D version
            if hasattr(option, 'max_depth_diff'):
                option.max_depth_diff = 0.07
            if hasattr(option, 'min_depth'):
                option.min_depth = 0.1
            if hasattr(option, 'max_depth'):
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
                # If odometry fails, use a simple fallback
                return self._fallback_pose_estimation(current_rgbd, prev_rgbd, prev_pose)
                
        except Exception as e:
            print(f"Odometry estimation failed: {e}")
            # Use fallback pose estimation
            return self._fallback_pose_estimation(current_rgbd, prev_rgbd, prev_pose)
    
    def _fallback_pose_estimation(self, current_rgbd, prev_rgbd, prev_pose):
        """Fallback pose estimation using simple feature matching"""
        try:
            # Convert RGBD images to point clouds for ICP
            intrinsic = self.get_camera_intrinsics()
            
            current_pc = o3d.geometry.PointCloud.create_from_rgbd_image(
                current_rgbd, intrinsic
            )
            prev_pc = o3d.geometry.PointCloud.create_from_rgbd_image(
                prev_rgbd, intrinsic
            )
            
            # Downsample for faster processing
            current_pc = current_pc.voxel_down_sample(0.05)
            prev_pc = prev_pc.voxel_down_sample(0.05)
            
            if len(current_pc.points) < 100 or len(prev_pc.points) < 100:
                return prev_pose
            
            # Estimate normals
            current_pc.estimate_normals()
            prev_pc.estimate_normals()
            
            # ICP registration
            threshold = 0.1
            reg_p2p = o3d.pipelines.registration.registration_icp(
                current_pc, prev_pc, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            
            if reg_p2p.fitness > 0.1:  # Reasonable alignment
                return prev_pose @ reg_p2p.transformation
            else:
                return prev_pose
                
        except Exception as e:
            print(f"Fallback pose estimation failed: {e}")
            return prev_pose
    
    def start_recording(self):
        """Start recording"""
        if self.state != ScanState.IDLE:
            return False
            
        print(f"Starting {'simple' if self.simple_mode else 'SLAM'} recording...")
        self.state = ScanState.RECORDING
        self.frame_count = 0
        self.start_time = time.time()
        
        if self.simple_mode:
            self.accumulated_pointcloud = o3d.geometry.PointCloud()
        else:
            self.rgbd_frames = []
            self.poses = []
            self.prev_rgbd = None
            self.current_pose = np.eye(4)
            self.trajectory = [self.current_pose.copy()]
        
        return True
    
    def stop_recording(self):
        """Stop recording and process the data"""
        if self.state != ScanState.RECORDING:
            return False
            
        print("Stopping recording...")
        
        if self.simple_mode:
            frames_captured = self.frame_count // 5  # We capture every 5th frame
            points_count = len(self.accumulated_pointcloud.points) if hasattr(self, 'accumulated_pointcloud') else 0
            print(f"Captured {frames_captured} frames with {points_count} points in {time.time() - self.start_time:.1f} seconds")
        else:
            print(f"Captured {len(self.rgbd_frames)} frames in {time.time() - self.start_time:.1f} seconds")
        
        self.state = ScanState.PROCESSING
        
        # Start reconstruction in background thread
        threading.Thread(target=self.reconstruct_scene, daemon=True).start()
        
        return True
    
    def reconstruct_scene(self):
        """Perform reconstruction based on mode"""
        try:
            if self.simple_mode:
                self._reconstruct_simple()
            else:
                self._reconstruct_slam()
        except Exception as e:
            print(f"Reconstruction failed: {e}")
            self.state = ScanState.IDLE
    
    def _reconstruct_simple(self):
        """Simple reconstruction from accumulated point cloud"""
        print("Starting simple reconstruction...")
        
        if len(self.accumulated_pointcloud.points) < 1000:
            print("Not enough points for reconstruction")
            self.state = ScanState.IDLE
            return
        
        print(f"Processing {len(self.accumulated_pointcloud.points)} points...")
        
        # Clean up the point cloud
        self.accumulated_pointcloud, _ = self.accumulated_pointcloud.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        
        # Estimate normals for mesh generation
        self.accumulated_pointcloud.estimate_normals()
        
        # Create mesh using Poisson reconstruction
        print("Generating mesh using Poisson reconstruction...")
        self.mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.accumulated_pointcloud, depth=8
        )
        
        # Clean up the mesh
        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_duplicated_triangles()
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_non_manifold_edges()
        
        print(f"Simple reconstruction complete! Mesh has {len(self.mesh.vertices)} vertices and {len(self.mesh.triangles)} triangles")
        self.state = ScanState.COMPLETE
    
    def _reconstruct_slam(self):
        """SLAM reconstruction using TSDF integration"""
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
        
        print(f"SLAM reconstruction complete! Mesh has {len(self.mesh.vertices)} vertices and {len(self.mesh.triangles)} triangles")
        self.state = ScanState.COMPLETE
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
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
                
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            if self.state == ScanState.RECORDING:
                try:
                    if self.simple_mode:
                        # Simple mode: just accumulate point clouds
                        self._process_frame_simple(color_image, depth_image)
                    else:
                        # SLAM mode: full pose tracking
                        self._process_frame_slam(color_image, depth_image)
                    
                    self.frame_count += 1
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    # Continue with next frame
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"Camera frame error: {e}")
            return None, None
    
    def _process_frame_simple(self, color_image, depth_image):
        """Process frame in simple mode"""
        if self.frame_count % 5 == 0:  # Every 5th frame to avoid too much data
            # Create point cloud from current frame
            rgbd = self.create_rgbd_image(color_image, depth_image)
            intrinsic = self.get_camera_intrinsics()
            
            current_pc = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsic
            )
            
            if len(current_pc.points) > 0:
                # Clean up the point cloud
                current_pc, _ = current_pc.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0
                )
                current_pc = current_pc.voxel_down_sample(self.voxel_size)
                
                # Add to accumulated point cloud
                self.accumulated_pointcloud += current_pc
                
                # Downsample accumulated cloud to keep it manageable
                if len(self.accumulated_pointcloud.points) > 200000:
                    self.accumulated_pointcloud = self.accumulated_pointcloud.voxel_down_sample(self.voxel_size * 2)
    
    def _process_frame_slam(self, color_image, depth_image):
        """Process frame in SLAM mode"""
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
            'mode': 'Simple' if self.simple_mode else 'SLAM',
            'recording_time': time.time() - self.start_time if self.start_time else 0,
        }
        
        if self.simple_mode:
            info['points_accumulated'] = len(self.accumulated_pointcloud.points) if hasattr(self, 'accumulated_pointcloud') else 0
            info['frames_processed'] = self.frame_count // 5 if hasattr(self, 'frame_count') else 0
        else:
            info['frames_captured'] = len(self.rgbd_frames) if hasattr(self, 'rgbd_frames') else 0
            info['poses_estimated'] = len(self.poses) if hasattr(self, 'poses') else 0
        
        if hasattr(self, 'mesh') and self.mesh:
            info['mesh_vertices'] = len(self.mesh.vertices)
            info['mesh_triangles'] = len(self.mesh.triangles)
        else:
            info['mesh_vertices'] = 0
            info['mesh_triangles'] = 0
            
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
                        if self.simple_mode:
                            cv2.putText(display_image, f"Points: {info['points_accumulated']}", (10, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
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

def test_realsense_connection():
    """Test RealSense connection and print device info"""
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("No RealSense devices found!")
            print("Make sure your D455 is connected via USB 3.0")
            return False
        
        print(f"Found {len(devices)} RealSense device(s):")
        
        for i, device in enumerate(devices):
            print(f"  Device {i}:")
            print(f"    Name: {device.get_info(rs.camera_info.name)}")
            print(f"    Serial: {device.get_info(rs.camera_info.serial_number)}")
            print(f"    Firmware: {device.get_info(rs.camera_info.firmware_version)}")
            print(f"    USB Type: {device.get_info(rs.camera_info.usb_type_descriptor)}")
            
            # Check available sensors
            sensors = device.query_sensors()
            print(f"    Sensors: {len(sensors)}")
            
            for j, sensor in enumerate(sensors):
                print(f"      Sensor {j}: {sensor.get_info(rs.camera_info.name)}")
                profiles = sensor.get_stream_profiles()
                print(f"        Available streams: {len(profiles)}")
                
                # Show a few example streams
                depth_streams = [p for p in profiles if p.stream_type() == rs.stream.depth]
                color_streams = [p for p in profiles if p.stream_type() == rs.stream.color]
                
                if depth_streams:
                    print(f"        Depth streams: {len(depth_streams)}")
                    for p in depth_streams[:3]:  # Show first 3
                        vp = p.as_video_stream_profile()
                        print(f"          {vp.width()}x{vp.height()} @ {vp.fps()}fps ({vp.format()})")
                
                if color_streams:
                    print(f"        Color streams: {len(color_streams)}")
                    for p in color_streams[:3]:  # Show first 3
                        vp = p.as_video_stream_profile()
                        print(f"          {vp.width()}x{vp.height()} @ {vp.fps()}fps ({vp.format()})")
        
        return True
        
    except Exception as e:
        print(f"Error testing RealSense connection: {e}")
        return False

def main():
    """Main application entry point"""
    print("Testing RealSense connection...")
    
    if not test_realsense_connection():
        print("\nConnection test failed. Please check:")
        print("1. D455 is connected to USB 3.0 port")
        print("2. Intel RealSense SDK is installed")
        print("3. Device drivers are up to date")
        print("4. Try restarting the application")
        return
    
    print("\nConnection test passed!")
    
    # Ask user for mode selection
    print("\nSelect scanning mode:")
    print("1. SLAM Mode (advanced reconstruction with pose tracking)")
    print("2. Simple Mode (basic point cloud accumulation)")
    
    while True:
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                simple_mode = False
                break
            elif choice == "2":
                simple_mode = True
                break
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    print("Starting scanner...")
    
    try:
        scanner = RealSenseSlam3DScanner(simple_mode=simple_mode)
        scanner.run_interface()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Try unplugging and reconnecting the D455")
        print("2. Close Intel RealSense Viewer if it's running")
        print("3. Try a different USB 3.0 port")
        print("4. Check if another application is using the camera")
        print("5. Try running in Simple Mode if SLAM Mode fails")

if __name__ == "__main__":
    main()