# run_mot_dataset.py
# Process MOT17 dataset sequences with tracking and evaluation

import cv2
import os
import argparse
from pathlib import Path
from mot_dataset_processor import MOT17Processor
from centroid_tracker import CentroidTracker, LineCounter
from yolo_detector import YOLODetector, FrameProcessor

def process_mot_sequence(seq_path, output_dir=None, conf=0.5, device="cpu"):
    """
    Process a single MOT17 sequence.
    
    Args:
        seq_path (str): Path to sequence folder (e.g., MOT17/train/MOT17-02-DPM/)
        output_dir (str): Directory to save output video
        conf (float): Detection confidence threshold
        device (int or str): Device (0 for GPU, 'cpu' for CPU)
    """
    
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(seq_path)}")
    print(f"{'='*60}")
    
    # Load sequence data
    seq_data = MOT17Processor.load_sequence(seq_path)
    seqinfo = seq_data['seqinfo']
    
    print(f"Frames: {seqinfo.get('seqLength', 'Unknown')}")
    print(f"Resolution: {seqinfo.get('imWidth', 'Unknown')}x{seqinfo.get('imHeight', 'Unknown')}")
    print(f"FPS: {seqinfo.get('frameRate', 'Unknown')}")
    
    # Initialize detector and tracker
    detector = YOLODetector(model_name="yolov8m.pt", conf=conf, device=device)
    tracker = CentroidTracker(max_disappeared=30, max_distance=50)
    
    # Get frame dimensions
    frame, _, _ = MOT17Processor.get_frame(seq_data, 0)
    height, width = frame.shape[:2]
    fps = int(seqinfo.get('frameRate', 30))
    
    # Initialize line counter
    line_start = (0, height // 2)
    line_end = (width, height // 2)
    line_counter = LineCounter(line_start, line_end)
    
    # Setup video writer if output directory provided
    writer = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{os.path.basename(seq_path)}_tracked.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        print(f"Output: {output_file}")
    
    # Process frames
    frame_count = 0
    count_up = 0
    count_down = 0
    seq_length = int(seqinfo.get('seqLength', len(seq_data['frames'])))
    
    for frame_idx in range(seq_length):
        frame, gt_ann, det_ann = MOT17Processor.get_frame(seq_data, frame_idx)
        frame_count += 1
        
        # Detect pedestrians using YOLO
        detections = detector.detect(frame, classes=[0])
        
        # Extract bounding boxes
        rects = [[d[0], d[1], d[2], d[3]] for d in detections]
        
        # Update tracker
        objects = tracker.update(rects)
        
        # Count crossings
        count_up, count_down, _ = line_counter.update_and_count(objects)
        
        # Draw on frame
        frame = FrameProcessor.draw_detections(frame, detections, color=(0, 255, 0))
        frame = FrameProcessor.draw_tracks(frame, objects, color=(255, 0, 0))
        frame = FrameProcessor.draw_line(frame, line_start, line_end)
        frame = FrameProcessor.draw_counts(frame, count_up, count_down)
        
        # Add frame info
        info_text = f"Frame: {frame_count}/{seq_length}"
        cv2.putText(frame, info_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame
        if writer:
            writer.write(frame)
        
        # Progress
        if frame_count % max(1, seq_length // 10) == 0:
            progress = (frame_count / seq_length) * 100
            print(f"Progress: {frame_count}/{seq_length} ({progress:.1f}%) | "
                  f"Detections: {len(detections)} | Tracked: {len(objects)}")
    
    # Cleanup
    if writer:
        writer.release()
    
    print(f"\nSequence complete!")
    print(f"Total frames: {frame_count}")
    print(f"People crossing UP: {count_up}")
    print(f"People crossing DOWN: {count_down}")
    
    return {
        'sequence': os.path.basename(seq_path),
        'total_frames': frame_count,
        'count_up': count_up,
        'count_down': count_down,
        'total_count': count_up + count_down
    }


def process_mot_dataset(dataset_path, split='train', output_dir=None, conf=0.5, device="cpu"):
    """
    Process all sequences in MOT17 dataset.
    
    Args:
        dataset_path (str): Path to MOT17 dataset root
        split (str): 'train' or 'test'
        output_dir (str): Directory to save outputs
        conf (float): Detection confidence threshold
        device (int or str): Device (0 for GPU, 'cpu' for CPU)
    """
    
    split_path = os.path.join(dataset_path, split)
    
    if not os.path.exists(split_path):
        print(f"ERROR: Dataset split not found: {split_path}")
        return
    
    # Get all sequences
    sequences = sorted([d for d in os.listdir(split_path) 
                       if os.path.isdir(os.path.join(split_path, d))])
    
    print(f"\nFound {len(sequences)} sequences in {split} split")
    print(f"Sequences: {', '.join(sequences[:3])}...")
    
    results = []
    
    for seq_name in sequences:
        seq_path = os.path.join(split_path, seq_name)
        result = process_mot_sequence(seq_path, output_dir, conf, device)
        results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_all = 0
    for result in results:
        print(f"{result['sequence']}: UP={result['count_up']} DOWN={result['count_down']} "
              f"TOTAL={result['total_count']}")
        total_all += result['total_count']
    
    print(f"\nTotal people counted: {total_all}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MOT17 dataset")
    parser.add_argument("dataset", type=str, help="Path to MOT17 dataset root")
    parser.add_argument("--split", type=str, default="train", 
                       choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence")
    parser.add_argument("--device", type=str, default="cpu", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--sequence", type=str, default=None, 
                       help="Process single sequence (optional)")
    
    args = parser.parse_args()
    
    if args.sequence:
        # Process single sequence
        seq_path = os.path.join(args.dataset, args.split, args.sequence)
        process_mot_sequence(seq_path, args.output, args.conf, args.device)
    else:
        # Process entire dataset split
        process_mot_dataset(args.dataset, args.split, args.output, args.conf, args.device)
