# scripts/run_video_tracking_polygon.py
# Video tracking with polygon ROI selection

import cv2
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.centroid_tracker import CentroidTracker
from scripts.polygon_counter import PolygonCounter, PolygonSelector
from scripts.yolo_detector import YOLODetector, FrameProcessor


def process_video_with_polygon(video_path, output_path=None, conf=0.3, device="cpu", show=True):
    """
    Process video file with polygon ROI selection and pedestrian tracking.
    
    Args:
        video_path (str): Path to input video file (relative to data/videos/)
        output_path (str): Path to save output video (relative to data/output/)
        conf (float): Detection confidence threshold
        device (int or str): 0 for GPU, 'cpu' for CPU
        show (bool): Display video while processing
    """
    print("\n" + "="*60)
    print("POLYGON ROI TRACKING SYSTEM")
    print("="*60)
    
    # Resolve paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if video_path is absolute or needs data/videos/ prefix
    if not os.path.isabs(video_path) and not os.path.exists(video_path):
        video_path = os.path.join(project_root, 'data', 'videos', video_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {os.path.basename(video_path)}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print(f"Total frames: {total_frames}")
    
    # Read first frame for polygon selection
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        return
    
    # Interactive polygon selection
    print("\n" + "="*60)
    print("STEP 1: Select Polygon ROI")
    print("="*60)
    print("Instructions:")
    print("  1. Click 4 points to define the polygon region")
    print("  2. Press 'c' to confirm selection")
    print("  3. Press 'r' to reset and select again")
    print("  4. Press ESC to cancel")
    print()
    
    selector = PolygonSelector()
    polygon_points = selector.select_polygon(first_frame)
    
    if polygon_points is None:
        print("Polygon selection cancelled.")
        return
    
    print(f"\nPolygon ROI selected with points: {polygon_points}")
    
    # Initialize detector and tracker
    print("\n" + "="*60)
    print("STEP 2: Initializing Detection and Tracking")
    print("="*60)
    
    # YOLO model path
    yolo_model_path = os.path.join(project_root, 'yolov8m.pt')
    
    detector = YOLODetector(model_name=yolo_model_path, conf=conf, device=device)
    tracker = CentroidTracker(max_disappeared=30, max_distance=50)
    polygon_counter = PolygonCounter(polygon_points, buffer_distance=30)
    
    print("✓ YOLO detector initialized")
    print("✓ Centroid tracker initialized")
    print("✓ Polygon counter initialized")
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        # Resolve output path
        if not os.path.isabs(output_path):
            output_path = os.path.join(project_root, 'data', 'output', output_path)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"✓ Output will be saved to: {output_path}")
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    max_id_seen = 0
    
    print("\n" + "="*60)
    print("STEP 3: Processing Video")
    print("="*60)
    print("Press 'q' to stop processing early\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect pedestrians
        detections = detector.detect(frame, classes=[0])  # 0 = person
        
        # Extract bounding boxes
        rects = [[d[0], d[1], d[2], d[3]] for d in detections]
        
        # Update tracker
        objects = tracker.update(rects)
        
        # Track max ID
        if len(objects) > 0:
            max_id_seen = max(max(objects.keys()), max_id_seen)
        
        # Count crossings through polygon
        count_in, count_out, _ = polygon_counter.update_and_count(objects)
        
        # Draw on frame
        frame = FrameProcessor.draw_detections(frame, detections, color=(0, 255, 0))
        frame = FrameProcessor.draw_tracks(frame, objects, color=(255, 0, 0))
        
        # Draw polygon ROI
        frame = polygon_counter.draw_polygon(frame, color=(0, 255, 255), thickness=2)
        
        # Draw counts
        frame = polygon_counter.draw_counts(frame, position=(10, 30))
        
        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames} | Tracked: {len(objects)} | Unique IDs: {max_id_seen + 1}"
        cv2.putText(frame, info_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        if writer:
            writer.write(frame)
        
        # Display
        if show:
            cv2.imshow("Polygon ROI Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nProcessing stopped by user")
                break
        
        # Progress
        if frame_count % max(1, total_frames // 20) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"Detections: {len(detections)} | Tracked: {len(objects)} | "
                  f"IN: {count_in} | OUT: {count_out}")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total frames processed:       {frame_count}")
    print(f"Unique pedestrians detected:  {max_id_seen + 1}")
    print(f"Objects entering polygon:     {count_in}")
    print(f"Objects leaving polygon:      {count_out}")
    print(f"Total polygon crossings:      {count_in + count_out}")
    
    if output_path:
        print(f"\n✓ Output saved to: {output_path}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian tracking with polygon ROI")
    parser.add_argument("video", type=str, help="Path to input video file (in data/videos/ or absolute path)")
    parser.add_argument("--output", type=str, default=None, help="Output filename (will be saved in data/output/)")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence")
    parser.add_argument("--device", type=str, default="cpu", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--no-display", action="store_true", help="Don't display video")
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{base_name}_polygon_tracked.mp4"
    
    # Run processing
    process_video_with_polygon(
        args.video, 
        args.output, 
        conf=args.conf,
        device=args.device, 
        show=not args.no_display
    )
