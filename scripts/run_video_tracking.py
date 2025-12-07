# run_video_tracking_fixed.py
# FIXED video tracking script with proper counting logic

import cv2
import os
import argparse
from centroid_tracker import CentroidTracker, LineCounterFixed, SmartLineCounter
from yolo_detector import YOLODetector, FrameProcessor

def process_video_fixed(video_path, output_path=None, conf=0.5, device="cpu", show=True, counter_type='smart'):
    """
    Process video file with FIXED pedestrian tracking and counting.
    
    Args:
        video_path (str): Path to input video file
        output_path (str): Path to save output video (optional)
        conf (float): Detection confidence threshold
        device (int or str): 0 for GPU, 'cpu' for CPU
        show (bool): Display video while processing
        counter_type (str): 'fixed' or 'smart' line counter
    """
    
    print("\n" + "="*60)
    print("FIXED TRACKING SYSTEM - Preventing Double Counting")
    print("="*60)
    
    # Initialize detector and tracker
    detector = YOLODetector(model_name="yolov8m.pt", conf=conf, device=device)
    tracker = CentroidTracker(max_disappeared=30, max_distance=50)
    
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
    print(f"Counting mode: {counter_type}")
    
    # Initialize counter
    line_y = height // 2
    if counter_type == 'smart':
        line_counter = SmartLineCounter(line_y=line_y, neutral_zone_height=60)
        print(f"Using SMART counter (zones)")
    else:
        line_counter = LineCounterFixed(line_start=(0, line_y), line_end=(width, line_y))
        print(f"Using FIXED counter (state-based)")
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    max_id_seen = 0
    
    print("\nProcessing...")
    
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
        
        # Count crossings (FIXED version)
        count_up, count_down, _ = line_counter.update_and_count(objects)
        
        # Draw on frame
        frame = FrameProcessor.draw_detections(frame, detections, color=(0, 255, 0))
        frame = FrameProcessor.draw_tracks(frame, objects, color=(255, 0, 0))
        
        # Draw line with zones (if smart counter)
        if counter_type == 'smart':
            # Draw detection zones
            cv2.line(frame, (0, line_y - 60), (width, line_y - 60), (200, 200, 0), 2)  # Up zone
            cv2.line(frame, (0, line_y + 60), (width, line_y + 60), (200, 200, 0), 2)  # Down zone
            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)  # Main line
            
            # Zone labels
            cv2.putText(frame, "UP ZONE", (10, line_y - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "DOWN ZONE", (10, line_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Simple line
            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        
        frame = FrameProcessor.draw_counts(frame, count_up, count_down)
        
        # Add info
        info_text = f"Unique IDs: {max_id_seen + 1} | Current: {len(objects)}"
        cv2.putText(frame, info_text, (width - 400, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame
        if writer:
            writer.write(frame)
        
        # Display
        if show:
            cv2.imshow("FIXED Pedestrian Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Progress
        if frame_count % max(1, total_frames // 15) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"Detections: {len(detections)} | Tracked: {len(objects)} | "
                  f"Up: {count_up} | Down: {count_down}")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n" + "="*60)
    print("RESULTS (FIXED COUNTING)")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Unique pedestrians detected: {max_id_seen + 1}")
    print(f"People crossing UP: {count_up}")
    print(f"People crossing DOWN: {count_down}")
    print(f"Total crossings: {count_up + count_down}")
    print(f"\n✓ This should be MUCH more accurate than before!")
    print(f"✓ Max count should be close to unique pedestrians × 2 (up + down)")
    
    if output_path:
        print(f"✓ Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIXED real-time pedestrian tracking")
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence")
    parser.add_argument("--device", type=str, default="cpu", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--no-display", action="store_true", help="Don't display video")
    parser.add_argument("--counter", type=str, default="smart", choices=["fixed", "smart"],
                       help="Counter type: 'fixed' (state-based) or 'smart' (zone-based)")
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output is None:
        base_name = os.path.splitext(args.video)[0]
        args.output = f"{base_name}_tracked_FIXED.mp4"
    
    # Run processing
    process_video_fixed(args.video, args.output, conf=args.conf, 
                       device=args.device, show=not args.no_display,
                       counter_type=args.counter)
