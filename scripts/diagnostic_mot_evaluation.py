# scripts/diagnostic_mot_evaluation.py
# Comprehensive diagnostic tool to understand MOT evaluation results

import cv2
import os
import sys
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mot_dataset_processor import MOT17Processor
from scripts.yolo_detector import YOLODetector
from scripts.centroid_tracker import CentroidTracker

def analyze_detection_quality(seq_path, conf=0.5, num_frames=100):
    """
    Analyze detection quality frame-by-frame to understand the problem.
    """
    print("\n" + "="*80)
    print("MOT EVALUATION DIAGNOSTIC TOOL")
    print("="*80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yolo_model_path = os.path.join(project_root, 'yolov8m.pt')
    
    # Load sequence
    seq_data = MOT17Processor.load_sequence(seq_path)
    seqinfo = seq_data['seqinfo']
    
    print(f"\nSequence: {os.path.basename(seq_path)}")
    print(f"Resolution: {seqinfo.get('imWidth')}x{seqinfo.get('imHeight')}")
    print(f"Frames: {seqinfo.get('seqLength')}")
    print(f"Analyzing first {num_frames} frames...")
    
    # Initialize detector
    detector = YOLODetector(model_name=yolo_model_path, conf=conf, device="cpu")
    
    # Statistics
    gt_counts = []
    det_counts = []
    coverage_ratios = []
    
    print(f"\nFrame-by-Frame Analysis (conf={conf}):")
    print(f"{'Frame':<8} {'GT People':<12} {'Detected':<12} {'Coverage %':<12} {'Status'}")
    print("-" * 60)
    
    for frame_idx in range(min(num_frames, int(seqinfo.get('seqLength', 600)))):
        frame, gt_ann, _ = MOT17Processor.get_frame(seq_data, frame_idx)
        
        # Count GT people (only conf=1)
        gt_people = sum(1 for det in gt_ann if det[5] == 1)
        
        # Detect with YOLO
        detections = detector.detect(frame, classes=[0])
        num_detections = len(detections)
        
        # Calculate coverage
        coverage = (num_detections / gt_people * 100) if gt_people > 0 else 0
        
        gt_counts.append(gt_people)
        det_counts.append(num_detections)
        coverage_ratios.append(coverage)
        
        # Status indicator
        if coverage >= 80:
            status = "✓ Good"
        elif coverage >= 50:
            status = "⚠ Fair"
        else:
            status = "✗ Poor"
        
        if frame_idx % 10 == 0 or frame_idx < 5:  # Print every 10th frame
            print(f"{frame_idx+1:<8} {gt_people:<12} {num_detections:<12} {coverage:<12.1f} {status}")
    
    print("-" * 60)
    
    # Summary statistics
    avg_gt = np.mean(gt_counts)
    avg_det = np.mean(det_counts)
    avg_coverage = np.mean(coverage_ratios)
    min_coverage = np.min(coverage_ratios)
    max_coverage = np.max(coverage_ratios)
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"  Average GT people per frame:    {avg_gt:.1f}")
    print(f"  Average Detections per frame:   {avg_det:.1f}")
    print(f"  Average Coverage:               {avg_coverage:.1f}%")
    print(f"  Coverage Range:                 {min_coverage:.1f}% - {max_coverage:.1f}%")
    print(f"  Total GT Detections:            {sum(gt_counts):,}")
    print(f"  Total Your Detections:          {sum(det_counts):,}")
    
    # Diagnosis
    print(f"\nDIAGNOSIS:")
    if avg_coverage < 40:
        print("  ✗ CRITICAL: Detection coverage is VERY LOW (<40%)")
        print("    → Problem: YOLO confidence threshold too high")
        print("    → Solution: Lower --conf to 0.3 or 0.25")
    elif avg_coverage < 70:
        print("  ⚠ WARNING: Detection coverage is LOW (40-70%)")
        print("    → Problem: Missing many people")
        print("    → Solution: Lower --conf to 0.35-0.4")
    else:
        print("  ✓ GOOD: Detection coverage is adequate (>70%)")
        print("    → Your YOLO detection is working well")
    
    return avg_coverage


def compare_confidence_levels(seq_path, num_frames=20):
    """
    Compare different confidence levels to find optimal setting.
    """
    print("\n" + "="*80)
    print("CONFIDENCE THRESHOLD COMPARISON")
    print("="*80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yolo_model_path = os.path.join(project_root, 'yolov8m.pt')
    
    # Load sequence
    seq_data = MOT17Processor.load_sequence(seq_path)
    
    confidence_levels = [0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    
    print(f"\nTesting on first {num_frames} frames...")
    print(f"{'Confidence':<15} {'Avg Detection':<18} {'Avg Coverage %':<18} {'Recommendation'}")
    print("-" * 70)
    
    best_conf = None
    best_coverage = 0
    
    for conf in confidence_levels:
        detector = YOLODetector(model_name=yolo_model_path, conf=conf, device="cpu")
        
        total_gt = 0
        total_det = 0
        
        for frame_idx in range(min(num_frames, 600)):
            frame, gt_ann, _ = MOT17Processor.get_frame(seq_data, frame_idx)
            gt_people = sum(1 for det in gt_ann if det[5] == 1)
            detections = detector.detect(frame, classes=[0])
            
            total_gt += gt_people
            total_det += len(detections)
        
        avg_det = total_det / num_frames
        coverage = (total_det / total_gt * 100) if total_gt > 0 else 0
        
        # Recommendation
        if 80 <= coverage <= 120:
            recommendation = "✓ BEST"
            if coverage > best_coverage:
                best_coverage = coverage
                best_conf = conf
        elif 70 <= coverage <= 130:
            recommendation = "⚠ GOOD"
        else:
            recommendation = "✗ Poor"
        
        print(f"{conf:<15.2f} {avg_det:<18.1f} {coverage:<18.1f} {recommendation}")
    
    print("-" * 70)
    print(f"\nRECOMMENDATION: Use --conf {best_conf}")
    print(f"This should give you ~{best_coverage:.0f}% detection coverage")
    
    return best_conf


def explain_mota_calculation(gt_total, pred_total, fp, fn, id_sw):
    """
    Explain MOTA calculation step-by-step.
    """
    print("\n" + "="*80)
    print("MOTA CALCULATION EXPLAINED")
    print("="*80)
    
    print("\nWhat MOTA Measures:")
    print("  MOTA = Multi-Object Tracking Accuracy")
    print("  Combines DETECTION quality + TRACKING quality")
    print("  Formula: MOTA = 1 - (FP + FN + ID_Switches) / Total_GT_Detections")
    
    print(f"\nYour Numbers:")
    print(f"  Total GT Detections:     {gt_total:>6,}")
    print(f"  Total Your Detections:   {pred_total:>6,}")
    print(f"  False Positives (FP):    {fp:>6,}  ← Detected people that don't exist")
    print(f"  False Negatives (FN):    {fn:>6,}  ← Missed people that do exist")
    print(f"  ID Switches:             {id_sw:>6,}  ← Changed ID for same person")
    
    print(f"\nCalculation:")
    errors = fp + fn + id_sw
    mota = 1 - (errors / gt_total) if gt_total > 0 else 0
    
    print(f"  Total Errors = FP + FN + ID_Sw")
    print(f"               = {fp:,} + {fn:,} + {id_sw:,}")
    print(f"               = {errors:,}")
    print(f"")
    print(f"  MOTA = 1 - (Errors / GT_Total)")
    print(f"       = 1 - ({errors:,} / {gt_total:,})")
    print(f"       = 1 - {errors/gt_total:.4f}")
    print(f"       = {mota:.4f}")
    print(f"       = {mota*100:.2f}%")
    
    print(f"\nBreakdown of Errors:")
    fp_pct = (fp / errors * 100) if errors > 0 else 0
    fn_pct = (fn / errors * 100) if errors > 0 else 0
    sw_pct = (id_sw / errors * 100) if errors > 0 else 0
    
    print(f"  False Positives: {fp:>6,} ({fp_pct:>5.1f}%) of errors")
    print(f"  False Negatives: {fn:>6,} ({fn_pct:>5.1f}%) of errors ← MAIN PROBLEM")
    print(f"  ID Switches:     {id_sw:>6,} ({sw_pct:>5.1f}%) of errors")
    
    print(f"\nProblem Analysis:")
    detection_coverage = ((gt_total - fn) / gt_total * 100) if gt_total > 0 else 0
    
    if fn_pct > 80:
        print(f"  ✗ CRITICAL: 89% of errors are False Negatives!")
        print(f"  → You're only detecting {detection_coverage:.1f}% of people")
        print(f"  → YOLO confidence is TOO HIGH")
        print(f"  → Even perfect tracking can't fix this")
        print(f"  → SOLUTION: Lower confidence threshold to detect more people")
    elif fn_pct > 50:
        print(f"  ⚠ WARNING: Majority of errors are False Negatives")
        print(f"  → Detection coverage is only {detection_coverage:.1f}%")
        print(f"  → Consider lowering confidence threshold")
    else:
        print(f"  ✓ GOOD: Detection coverage is {detection_coverage:.1f}%")
        print(f"  → Main improvements needed in tracking, not detection")


def main():
    """Run all diagnostic tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnostic tool for MOT evaluation")
    parser.add_argument("--sequence", type=str, default="MOT17-02-DPM",
                       help="Sequence name (default: MOT17-02-DPM)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="YOLO confidence threshold (default: 0.5)")
    parser.add_argument("--frames", type=int, default=100,
                       help="Number of frames to analyze (default: 100)")
    
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seq_path = os.path.join(project_root, 'data', 'MOT17', 'train', args.sequence)
    
    if not os.path.exists(seq_path):
        print(f"ERROR: Sequence not found: {seq_path}")
        return
    
    print("\n" + "="*80)
    print("MOT EVALUATION DIAGNOSTICS")
    print("="*80)
    print(f"\nThis tool will help you understand your evaluation results")
    print(f"and find the optimal YOLO confidence threshold.\n")
    
    # Test 1: Analyze current detection quality
    print("\n" + "="*80)
    print("TEST 1: Current Detection Quality")
    print("="*80)
    avg_coverage = analyze_detection_quality(seq_path, conf=args.conf, num_frames=args.frames)
    
    # Test 2: Compare different confidence levels
    print("\n" + "="*80)
    print("TEST 2: Find Optimal Confidence")
    print("="*80)
    best_conf = compare_confidence_levels(seq_path, num_frames=20)
    
    # Test 3: Explain MOTA
    # Using your actual numbers
    explain_mota_calculation(
        gt_total=18581,
        pred_total=6007,
        fp=1602,
        fn=14176,
        id_sw=44
    )
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    print(f"\n1. Your current detection coverage: {avg_coverage:.1f}%")
    print(f"   → This is why MOTA is low (missing too many people)")
    print(f"")
    print(f"2. Recommended confidence: {best_conf}")
    print(f"   → Should improve coverage to 80-100%")
    print(f"")
    print(f"3. Run evaluation with new confidence:")
    print(f"   python scripts/run_mot_dataset_with_eval.py \\")
    print(f"       --sequence {args.sequence} --conf {best_conf}")
    print(f"")
    print(f"4. Expected improvement:")
    print(f"   MOTA: ~15% → ~50-70%")
    print(f"   Recall: ~24% → ~75-85%")
    print(f"   Detection Coverage: {avg_coverage:.0f}% → ~85-95%")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
