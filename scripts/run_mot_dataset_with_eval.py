# scripts/run_mot_dataset_with_eval.py
# Process MOT17 dataset with evaluation against ground truth

import cv2
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mot_dataset_processor import MOT17Processor
from scripts.centroid_tracker import CentroidTracker, LineCounterFixed
from scripts.yolo_detector import YOLODetector, FrameProcessor
from scripts.mot_evaluator import MOTEvaluator


def process_mot_sequence_with_eval(seq_path, output_dir=None, conf=0.5, device="cpu", evaluate=True, yolo_model_path=None):
    """
    Process a single MOT17 sequence with evaluation.
    
    Args:
        seq_path (str): Path to sequence folder
        output_dir (str): Directory to save output video
        conf (float): Detection confidence threshold
        device (int or str): Device (0 for GPU, 'cpu' for CPU)
        evaluate (bool): Whether to evaluate against ground truth
        yolo_model_path (str): Path to YOLO model file
    
    Returns:
        dict: Results including metrics if evaluation enabled
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
    detector = YOLODetector(model_name=yolo_model_path, conf=conf, device=device)
    tracker = CentroidTracker(max_disappeared=30, max_distance=50)
    
    # Initialize evaluator if evaluation enabled
    evaluator = MOTEvaluator() if evaluate else None
    
    # Get frame dimensions
    frame, _, _ = MOT17Processor.get_frame(seq_data, 0)
    height, width = frame.shape[:2]
    fps = int(seqinfo.get('frameRate', 30))
    
    # Initialize line counter (for compatibility)
    line_start = (0, height // 2)
    line_end = (width, height // 2)
    line_counter = LineCounterFixed(line_start, line_end)
    
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
    
    print(f"\nProcessing {seq_length} frames...")
    if evaluate:
        print("Evaluation: ENABLED - comparing against ground truth")
    
    for frame_idx in range(seq_length):
        frame, gt_ann, det_ann = MOT17Processor.get_frame(seq_data, frame_idx)
        frame_count += 1
        frame_id = frame_idx + 1  # MOT format is 1-based
        
        # Detect pedestrians using YOLO
        detections = detector.detect(frame, classes=[0])
        
        # Extract bounding boxes
        rects = [[d[0], d[1], d[2], d[3]] for d in detections]
        
        # Update tracker
        objects = tracker.update(rects)
        
        # Count crossings
        count_up, count_down, _ = line_counter.update_and_count(objects)
        
        # Add to evaluator if evaluation enabled
        if evaluator and len(gt_ann) > 0:
            # Add ground truth
            evaluator.add_ground_truth(frame_id, gt_ann)
            
            # Add predictions (convert centroids to bbox format)
            predictions = []
            for obj_id, (cx, cy) in objects.items():
                # Find corresponding detection bbox
                matched = False
                for det in detections:
                    det_cx = (det[0] + det[2]) / 2
                    det_cy = (det[1] + det[3]) / 2
                    dist = ((cx - det_cx)**2 + (cy - det_cy)**2)**0.5
                    
                    if dist < 20:  # Close enough
                        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                        w, h = x2 - x1, y2 - y1
                        predictions.append((obj_id, x1, y1, w, h))
                        matched = True
                        break
                
                # If no match, use estimated bbox
                if not matched:
                    w, h = 50, 100  # Default size
                    predictions.append((obj_id, cx - w/2, cy - h/2, w, h))
            
            evaluator.add_predictions(frame_id, predictions)
        
        # Draw on frame (optional - only if saving video)
        if writer:
            frame = FrameProcessor.draw_detections(frame, detections, color=(0, 255, 0))
            frame = FrameProcessor.draw_tracks(frame, objects, color=(255, 0, 0))
            frame = FrameProcessor.draw_line(frame, line_start, line_end)
            frame = FrameProcessor.draw_counts(frame, count_up, count_down)
            
            # Add frame info
            info_text = f"Frame: {frame_count}/{seq_length}"
            cv2.putText(frame, info_text, (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
    
    # Evaluate metrics if enabled
    metrics = None
    if evaluator:
        print(f"\nComputing evaluation metrics...")
        metrics = evaluator.evaluate_sequence(iou_threshold=0.5)
        evaluator.print_metrics(metrics, sequence_name=os.path.basename(seq_path))
    
    return {
        'sequence': os.path.basename(seq_path),
        'total_frames': frame_count,
        'count_up': count_up,
        'count_down': count_down,
        'total_count': count_up + count_down,
        'metrics': metrics
    }


def process_mot_dataset_with_eval(dataset_path, split='train', output_dir=None, 
                                   conf=0.5, device="cpu", evaluate=True, yolo_model_path=None):
    """
    Process all sequences in MOT17 dataset with evaluation.
    
    Args:
        dataset_path (str): Path to MOT17 dataset root (data/MOT17)
        split (str): 'train' or 'test'
        output_dir (str): Directory to save outputs (data/output/)
        conf (float): Detection confidence threshold
        device (int or str): Device (0 for GPU, 'cpu' for CPU)
        evaluate (bool): Whether to evaluate against ground truth
        yolo_model_path (str): Path to YOLO model file
    """
    split_path = os.path.join(dataset_path, split)
    
    if not os.path.exists(split_path):
        print(f"ERROR: Dataset split not found: {split_path}")
        return
    
    # Check if ground truth is available
    if evaluate and split == 'test':
        print("WARNING: Test split typically doesn't have ground truth.")
        print("         Evaluation will be skipped.")
        evaluate = False
    
    # Get all sequences
    sequences = sorted([d for d in os.listdir(split_path)
                       if os.path.isdir(os.path.join(split_path, d))])
    
    print(f"\n{'='*70}")
    print(f"MOT17 DATASET EVALUATION")
    print(f"{'='*70}")
    print(f"Found {len(sequences)} sequences in {split} split")
    print(f"Sequences: {', '.join(sequences[:3])}...")
    print(f"Evaluation: {'ENABLED' if evaluate else 'DISABLED'}")
    print(f"{'='*70}")
    
    results = []
    
    for seq_name in sequences:
        seq_path = os.path.join(split_path, seq_name)
        result = process_mot_sequence_with_eval(seq_path, output_dir, conf, device, evaluate, yolo_model_path)
        results.append(result)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY - Counting Results")
    print(f"{'='*70}")
    
    total_all = 0
    for result in results:
        print(f"{result['sequence']:30s}: UP={result['count_up']:4d} DOWN={result['count_down']:4d} "
              f"TOTAL={result['total_count']:4d}")
        total_all += result['total_count']
    
    print(f"\nTotal people counted across all sequences: {total_all}")
    
    # Print evaluation summary if enabled
    if evaluate:
        print(f"\n{'='*70}")
        print("SUMMARY - Evaluation Metrics (Average)")
        print(f"{'='*70}")
        
        # Calculate average metrics
        valid_results = [r for r in results if r['metrics'] is not None]
        
        if len(valid_results) > 0:
            avg_mota = sum(r['metrics']['MOTA'] for r in valid_results) / len(valid_results)
            avg_precision = sum(r['metrics']['Precision'] for r in valid_results) / len(valid_results)
            avg_recall = sum(r['metrics']['Recall'] for r in valid_results) / len(valid_results)
            total_fp = sum(r['metrics']['FP'] for r in valid_results)
            total_fn = sum(r['metrics']['FN'] for r in valid_results)
            total_switches = sum(r['metrics']['ID_Sw'] for r in valid_results)
            total_gt_ids = sum(r['metrics']['Num_GT_IDs'] for r in valid_results)
            total_pred_ids = sum(r['metrics']['Num_Pred_IDs'] for r in valid_results)
            
            print(f"\nAverage Performance:")
            print(f"  MOTA:                    {avg_mota:.2f}%")
            print(f"  Precision:               {avg_precision:.2f}%")
            print(f"  Recall:                  {avg_recall:.2f}%")
            
            print(f"\nTotal Statistics:")
            print(f"  False Positives:         {total_fp}")
            print(f"  False Negatives:         {total_fn}")
            print(f"  ID Switches:             {total_switches}")
            print(f"  GT Trajectories:         {total_gt_ids}")
            print(f"  Predicted Trajectories:  {total_pred_ids}")
            
            print(f"\nPer-Sequence Metrics:")
            print(f"{'Sequence':<30s} {'MOTA':>8s} {'Prec':>8s} {'Rcll':>8s} {'FP':>6s} {'FN':>6s} {'ID_Sw':>6s}")
            print("-" * 70)
            for result in valid_results:
                m = result['metrics']
                print(f"{result['sequence']:<30s} "
                      f"{m['MOTA']:>7.2f}% "
                      f"{m['Precision']:>7.2f}% "
                      f"{m['Recall']:>7.2f}% "
                      f"{m['FP']:>6d} "
                      f"{m['FN']:>6d} "
                      f"{m['ID_Sw']:>6d}")
        else:
            print("No valid evaluation results (no ground truth found)")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MOT17 dataset with evaluation")
    parser.add_argument("--dataset", type=str, default=None, 
                       help="Path to MOT17 dataset root (default: data/MOT17)")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output directory (default: data/output)")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence")
    parser.add_argument("--device", type=str, default="cpu", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--sequence", type=str, default=None,
                       help="Process single sequence (e.g., MOT17-02-DPM)")
    parser.add_argument("--no-eval", action="store_true", 
                       help="Disable evaluation against ground truth")
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dataset path
    if args.dataset is None:
        dataset_path = os.path.join(project_root, 'data', 'MOT17')
    else:
        dataset_path = args.dataset
    
    # Output path
    if args.output is None:
        output_dir = os.path.join(project_root, 'data', 'output')
    else:
        output_dir = args.output
    
    # YOLO model path
    yolo_model_path = os.path.join(project_root, 'yolov8m.pt')
    
    evaluate = not args.no_eval
    
    if args.sequence:
        # Process single sequence
        seq_path = os.path.join(dataset_path, args.split, args.sequence)
        process_mot_sequence_with_eval(seq_path, output_dir, args.conf, args.device, evaluate, yolo_model_path)
    else:
        # Process entire dataset split
        process_mot_dataset_with_eval(dataset_path, args.split, output_dir, 
                                      args.conf, args.device, evaluate, yolo_model_path)
