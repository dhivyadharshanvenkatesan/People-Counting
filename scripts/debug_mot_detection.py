# scripts/debug_mot_detection.py
# Debug script to diagnose detection and dataset issues

import cv2
import os
import sys
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mot_dataset_processor import MOT17Processor
from scripts.yolo_detector import YOLODetector

def debug_gt_loading(seq_path, num_frames=5):
    """
    Debug ground truth loading to ensure it's correct.
    """
    print("\n" + "="*80)
    print("DEBUG 1: Ground Truth Loading")
    print("="*80)
    
    seq_data = MOT17Processor.load_sequence(seq_path)
    seqinfo = seq_data['seqinfo']
    
    print(f"\nSequence: {os.path.basename(seq_path)}")
    print(f"Seqinfo keys: {list(seqinfo.keys())}")
    print(f"seqLength: {seqinfo.get('seqLength')}")
    print(f"imWidth: {seqinfo.get('imWidth')}")
    print(f"imHeight: {seqinfo.get('imHeight')}")
    
    print(f"\nFrame-by-Frame GT Analysis (first {num_frames} frames):")
    print(f"{'Frame':<8} {'Image Path':<40} {'GT Dets':<10} {'Valid':<10}")
    print("-" * 70)
    
    total_gt = 0
    for frame_idx in range(num_frames):
        frame, gt_ann, det_ann = MOT17Processor.get_frame(seq_data, frame_idx)
        
        # Count GT with conf=1 (active)
        gt_active = sum(1 for g in gt_ann if len(g) > 5 and g[5] == 1)
        
        print(f"{frame_idx+1:<8} {f'Frame {frame_idx+1}':<40} {len(gt_ann):<10} {gt_active:<10}")
        total_gt += gt_active
        
        if len(gt_ann) > 0:
            print(f"  Sample GT: {gt_ann[0]}")
        if frame is not None:
            print(f"  Frame shape: {frame.shape}")
    
    print(f"\nTotal GT detections (conf=1): {total_gt}")
    print(f"Average per frame: {total_gt / num_frames:.1f}")


def test_yolo_raw(seq_path, num_frames=5):
    """
    Test YOLO detection directly on raw frames.
    """
    print("\n" + "="*80)
    print("DEBUG 2: YOLO Raw Detection")
    print("="*80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yolo_model_path = os.path.join(project_root, 'yolov8m.pt')
    
    print(f"\nModel path: {yolo_model_path}")
    print(f"Model exists: {os.path.exists(yolo_model_path)}")
    
    # Test with different confidence levels
    confs = [0.25, 0.5]
    
    seq_data = MOT17Processor.load_sequence(seq_path)
    
    for conf in confs:
        print(f"\n\nTesting with confidence = {conf}")
        print("-" * 70)
        
        detector = YOLODetector(model_name=yolo_model_path, conf=conf, device="cpu")
        
        print(f"Detector initialized")
        print(f"  Model: {detector.model.names}")
        print(f"  Conf: {detector.conf}")
        print(f"  Device: {detector.device}")
        
        total_detections = 0
        total_gt = 0
        
        print(f"\n{'Frame':<8} {'Frame Size':<20} {'GT':<8} {'YOLO':<8} {'Ratio %':<12}")
        print("-" * 70)
        
        for frame_idx in range(num_frames):
            frame, gt_ann, _ = MOT17Processor.get_frame(seq_data, frame_idx)
            
            # Count GT
            gt_count = sum(1 for g in gt_ann if len(g) > 5 and g[5] == 1)
            
            # Get YOLO detections
            detections = detector.detect(frame, classes=[0])
            yolo_count = len(detections)
            
            ratio = (yolo_count / gt_count * 100) if gt_count > 0 else 0
            
            print(f"{frame_idx+1:<8} {str(frame.shape):<20} {gt_count:<8} {yolo_count:<8} {ratio:<12.1f}%")
            
            total_detections += yolo_count
            total_gt += gt_count
            
            # Show sample detections
            if frame_idx == 0 and len(detections) > 0:
                print(f"\n  Sample YOLO detection: {detections[0]}")
            if frame_idx == 0 and len(gt_ann) > 0:
                print(f"  Sample GT: {gt_ann[0]}")
        
        print(f"\nSummary for conf={conf}:")
        print(f"  Total GT: {total_gt}")
        print(f"  Total YOLO: {total_detections}")
        print(f"  Coverage: {total_detections / total_gt * 100:.1f}%" if total_gt > 0 else "  N/A")


def visualize_detections(seq_path, frame_idx=0, conf=0.5):
    """
    Save a visualization of GT vs YOLO detections.
    """
    print("\n" + "="*80)
    print("DEBUG 3: Visual Comparison (Frame 1)")
    print("="*80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yolo_model_path = os.path.join(project_root, 'yolov8m.pt')
    
    seq_data = MOT17Processor.load_sequence(seq_path)
    frame, gt_ann, _ = MOT17Processor.get_frame(seq_data, frame_idx)
    
    # Draw GT
    frame_gt = frame.copy()
    for gt in gt_ann:
        if len(gt) > 5 and gt[5] == 1:  # conf=1
            x, y, w, h = int(gt[1]), int(gt[2]), int(gt[3]), int(gt[4])
            cv2.rectangle(frame_gt, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green = GT
    
    # Draw YOLO detections
    frame_yolo = frame.copy()
    detector = YOLODetector(model_name=yolo_model_path, conf=conf, device="cpu")
    detections = detector.detect(frame, classes=[0])
    
    for det in detections:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red = YOLO
    
    # Draw both
    frame_both = frame.copy()
    
    # Green for GT
    for gt in gt_ann:
        if len(gt) > 5 and gt[5] == 1:
            x, y, w, h = int(gt[1]), int(gt[2]), int(gt[3]), int(gt[4])
            cv2.rectangle(frame_both, (x, y), (x+w, y+h), (0, 255, 0), 1)
    
    # Red for YOLO
    for det in detections:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        cv2.rectangle(frame_both, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # Save visualizations
    output_dir = os.path.join(project_root, 'data', 'output', 'debug')
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, 'frame_gt.jpg'), frame_gt)
    cv2.imwrite(os.path.join(output_dir, 'frame_yolo.jpg'), frame_yolo)
    cv2.imwrite(os.path.join(output_dir, 'frame_both.jpg'), frame_both)
    
    print(f"\nVisualization saved to: {output_dir}")
    print(f"  frame_gt.jpg - Ground truth boxes (green)")
    print(f"  frame_yolo.jpg - YOLO detections (red)")
    print(f"  frame_both.jpg - Both overlaid")
    print(f"\nGT Detections: {len([g for g in gt_ann if len(g) > 5 and g[5] == 1])}")
    print(f"YOLO Detections: {len(detections)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug MOT detection issues")
    parser.add_argument("--sequence", type=str, default="MOT17-02-DPM",
                       help="Sequence name")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="YOLO confidence")
    
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seq_path = os.path.join(project_root, 'data', 'MOT17', 'train', args.sequence)
    
    if not os.path.exists(seq_path):
        print(f"ERROR: Sequence not found: {seq_path}")
        return
    
    print("\n" + "="*80)
    print("MOT DETECTION DEBUGGING TOOL")
    print("="*80)
    print("\nThis will help identify why YOLO detection is so low.\n")
    
    # Run all tests
    debug_gt_loading(seq_path, num_frames=5)
    test_yolo_raw(seq_path, num_frames=10)
    visualize_detections(seq_path, frame_idx=0, conf=args.conf)
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)
    print("\nCheck the output images in data/output/debug/")
    print("This will show you if GT boxes and YOLO boxes are in the same locations.\n")


if __name__ == "__main__":
    main()
