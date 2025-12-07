# scripts/mot_evaluator.py
# MOT17 evaluation metrics calculator - FIXED VERSION

import numpy as np
from collections import defaultdict
import os


class MOTEvaluator:
    """
    Calculate MOT17 tracking accuracy metrics by comparing predictions with ground truth.
    
    Key Metrics:
    - MOTA: Multi-Object Tracking Accuracy (detection + association)
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - ID Switches: How often same person gets different ID
    - MT/ML: Mostly Tracked/Lost trajectories
    """
    
    def __init__(self):
        """Initialize MOT evaluator."""
        self.reset()
    
    def reset(self):
        """Reset all tracking statistics."""
        self.ground_truth_tracks = defaultdict(list)  # {frame_id: [(id, x, y, w, h), ...]}
        self.predicted_tracks = defaultdict(list)      # {frame_id: [(id, x, y, w, h), ...]}
        
        # Per-frame metrics (will accumulate)
        self.num_frames = 0
        self.num_matches = 0
        self.num_false_positives = 0
        self.num_misses = 0
        self.num_switches = 0
        
        # Track statistics
        self.gt_trajectories = {}  # {id: [frame_ids]}
        self.pred_trajectories = {}  # {id: [frame_ids]}
        
        # FIXED: Keep match history across ALL frames (not reset per frame!)
        self.match_history = {}  # {frame_id: {gt_id: pred_id}}
        
    def add_ground_truth(self, frame_id, detections):
        """
        Add ground truth detections for a frame.
        
        Args:
            frame_id (int): Frame number (1-based)
            detections (list): List of (id, x, y, w, h, conf, class, vis) tuples
        """
        for det in detections:
            obj_id, x, y, w, h = det[0], det[1], det[2], det[3], det[4]
            conf = det[5] if len(det) > 5 else 1.0
            
            # Only consider active ground truth (conf != 0)
            # conf=0 means "ignore this detection in evaluation"
            if conf != 0:
                self.ground_truth_tracks[frame_id].append((obj_id, x, y, w, h))
                
                # Track trajectory (when does this person appear)
                if obj_id not in self.gt_trajectories:
                    self.gt_trajectories[obj_id] = []
                self.gt_trajectories[obj_id].append(frame_id)
    
    def add_predictions(self, frame_id, detections):
        """
        Add predicted detections for a frame.
        
        Args:
            frame_id (int): Frame number (1-based)
            detections (list): List of (id, x, y, w, h) tuples or (id, cx, cy) for centroids
        """
        for det in detections:
            if len(det) == 3:  # Centroid format (id, cx, cy)
                obj_id, cx, cy = det
                # Estimate bounding box (assuming average pedestrian size)
                w, h = 50, 100  # Default pedestrian size
                x, y = cx - w/2, cy - h/2
            else:  # Bounding box format
                obj_id, x, y, w, h = det[0], det[1], det[2], det[3], det[4]
            
            self.predicted_tracks[frame_id].append((obj_id, x, y, w, h))
            
            # Track trajectory
            if obj_id not in self.pred_trajectories:
                self.pred_trajectories[obj_id] = []
            self.pred_trajectories[obj_id].append(frame_id)
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1 (tuple): (x, y, w, h) - top-left corner and dimensions
            box2 (tuple): (x, y, w, h) - top-left corner and dimensions
            
        Returns:
            float: IoU value [0, 1]
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to (x1, y1, x2, y2) format
        box1_x2 = x1 + w1
        box1_y2 = y1 + h1
        box2_x2 = x2 + w2
        box2_y2 = y2 + h2
        
        # Calculate intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_detections(self, gt_boxes, pred_boxes, iou_threshold=0.5):
        """
        Match ground truth and predicted boxes using Hungarian algorithm (greedy IoU matching).
        
        Args:
            gt_boxes (list): List of (id, x, y, w, h) ground truth boxes
            pred_boxes (list): List of (id, x, y, w, h) predicted boxes
            iou_threshold (float): Minimum IoU for match
            
        Returns:
            tuple: (matches, unmatched_gt, unmatched_pred)
                   matches: list of (gt_id, pred_id) pairs
                   unmatched_gt: list of gt_ids (False Negatives)
                   unmatched_pred: list of pred_ids (False Positives)
        """
        if len(gt_boxes) == 0:
            return [], [], [p[0] for p in pred_boxes]
        
        if len(pred_boxes) == 0:
            return [], [g[0] for g in gt_boxes], []
        
        # Build IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = self.calculate_iou(
                    gt_box[1:5], pred_box[1:5]
                )
        
        # Greedy matching (highest IoU first)
        matches = []
        matched_gt = set()
        matched_pred = set()
        
        # Sort by IoU (descending)
        indices = np.unravel_index(np.argsort(iou_matrix, axis=None)[::-1], iou_matrix.shape)
        
        for i, j in zip(indices[0], indices[1]):
            if i in matched_gt or j in matched_pred:
                continue
            
            if iou_matrix[i, j] >= iou_threshold:
                gt_id = gt_boxes[i][0]
                pred_id = pred_boxes[j][0]
                matches.append((gt_id, pred_id))
                matched_gt.add(i)
                matched_pred.add(j)
        
        unmatched_gt = [gt_boxes[i][0] for i in range(len(gt_boxes)) if i not in matched_gt]
        unmatched_pred = [pred_boxes[j][0] for j in range(len(pred_boxes)) if j not in matched_pred]
        
        return matches, unmatched_gt, unmatched_pred
    
    def evaluate_sequence(self, iou_threshold=0.5):
        """
        Evaluate entire sequence.
        
        This is the MAIN evaluation function that computes all metrics.
        
        Args:
            iou_threshold (float): Minimum IoU for match (default: 0.5)
            
        Returns:
            dict: Evaluation metrics
        """
        # Reset per-sequence metrics
        self.num_matches = 0
        self.num_false_positives = 0
        self.num_misses = 0
        self.num_switches = 0
        self.match_history = {}
        
        # Get all frame IDs
        all_frames = sorted(set(list(self.ground_truth_tracks.keys()) + 
                               list(self.predicted_tracks.keys())))
        self.num_frames = len(all_frames)
        
        if len(all_frames) == 0:
            return self._empty_metrics()
        
        # FIXED: Track GT ID to Pred ID mapping across ALL frames
        gt_to_pred_mapping = {}  # {gt_id: last_matched_pred_id}
        
        # Evaluate each frame sequentially
        for frame_id in all_frames:
            gt_boxes = self.ground_truth_tracks.get(frame_id, [])
            pred_boxes = self.predicted_tracks.get(frame_id, [])
            
            matches, unmatched_gt, unmatched_pred = self.match_detections(
                gt_boxes, pred_boxes, iou_threshold
            )
            
            # Update metrics
            self.num_matches += len(matches)
            self.num_misses += len(unmatched_gt)  # False Negatives
            self.num_false_positives += len(unmatched_pred)  # False Positives
            
            # Store matches for this frame
            frame_matches = {}
            for gt_id, pred_id in matches:
                frame_matches[gt_id] = pred_id
                
                # Check for ID switch
                if gt_id in gt_to_pred_mapping:
                    # This GT ID was matched before
                    previous_pred_id = gt_to_pred_mapping[gt_id]
                    if previous_pred_id != pred_id:
                        # Same person (GT ID) now has different predicted ID!
                        self.num_switches += 1
                
                # Update mapping
                gt_to_pred_mapping[gt_id] = pred_id
            
            self.match_history[frame_id] = frame_matches
        
        # Calculate trajectory statistics (MT/ML)
        num_gt_trajectories = len(self.gt_trajectories)
        num_pred_trajectories = len(self.pred_trajectories)
        
        mt_count = 0  # Mostly Tracked
        ml_count = 0  # Mostly Lost
        pt_count = 0  # Partially Tracked
        
        for gt_id, frames in self.gt_trajectories.items():
            track_length = len(frames)
            
            # Count how many frames this GT was successfully matched
            detected_frames = 0
            for frame_id in frames:
                if frame_id in self.match_history and gt_id in self.match_history[frame_id]:
                    detected_frames += 1
            
            coverage = detected_frames / track_length if track_length > 0 else 0
            
            if coverage >= 0.8:
                mt_count += 1
            elif coverage <= 0.2:
                ml_count += 1
            else:
                pt_count += 1
        
        # Calculate metrics
        num_gt_detections = sum(len(boxes) for boxes in self.ground_truth_tracks.values())
        
        # MOTA (Multi-Object Tracking Accuracy)
        # Penalizes: False Positives, False Negatives (Misses), ID Switches
        if num_gt_detections > 0:
            mota = 1 - (self.num_false_positives + self.num_misses + self.num_switches) / num_gt_detections
            mota = max(mota, -1.0)  # MOTA can be negative if errors > GT
        else:
            mota = 0.0
        
        # Precision and Recall
        total_predicted = self.num_matches + self.num_false_positives
        total_gt = self.num_matches + self.num_misses
        
        precision = self.num_matches / total_predicted if total_predicted > 0 else 0
        recall = self.num_matches / total_gt if total_gt > 0 else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'MOTA': mota * 100,  # As percentage
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1': f1 * 100,
            'MT': mt_count,
            'MT_ratio': (mt_count / num_gt_trajectories * 100) if num_gt_trajectories > 0 else 0,
            'PT': pt_count,
            'PT_ratio': (pt_count / num_gt_trajectories * 100) if num_gt_trajectories > 0 else 0,
            'ML': ml_count,
            'ML_ratio': (ml_count / num_gt_trajectories * 100) if num_gt_trajectories > 0 else 0,
            'FP': self.num_false_positives,
            'FN': self.num_misses,
            'ID_Sw': self.num_switches,
            'Num_GT_IDs': num_gt_trajectories,
            'Num_Pred_IDs': num_pred_trajectories,
            'Num_Frames': self.num_frames,
            'Num_GT_Dets': num_gt_detections,
            'Num_Pred_Dets': sum(len(boxes) for boxes in self.predicted_tracks.values())
        }
        
        return metrics
    
    def _empty_metrics(self):
        """Return empty metrics when no data available."""
        return {
            'MOTA': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'MT': 0,
            'MT_ratio': 0.0,
            'PT': 0,
            'PT_ratio': 0.0,
            'ML': 0,
            'ML_ratio': 0.0,
            'FP': 0,
            'FN': 0,
            'ID_Sw': 0,
            'Num_GT_IDs': 0,
            'Num_Pred_IDs': 0,
            'Num_Frames': 0,
            'Num_GT_Dets': 0,
            'Num_Pred_Dets': 0
        }
    
    def print_metrics(self, metrics, sequence_name=""):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics (dict): Metrics dictionary
            sequence_name (str): Name of sequence (optional)
        """
        print(f"\n{'='*70}")
        if sequence_name:
            print(f"Evaluation Metrics for {sequence_name}")
        else:
            print("Evaluation Metrics")
        print(f"{'='*70}")
        
        print(f"\nOverall Performance:")
        print(f"  MOTA (Multi-Object Tracking Accuracy): {metrics['MOTA']:>7.2f}%")
        print(f"  Precision:                             {metrics['Precision']:>7.2f}%")
        print(f"  Recall:                                {metrics['Recall']:>7.2f}%")
        print(f"  F1 Score:                              {metrics['F1']:>7.2f}%")
        
        print(f"\nTrajectory Statistics:")
        print(f"  Mostly Tracked (MT):                   {metrics['MT']:>3d} ({metrics['MT_ratio']:>5.1f}%)")
        print(f"  Partially Tracked (PT):                {metrics['PT']:>3d} ({metrics['PT_ratio']:>5.1f}%)")
        print(f"  Mostly Lost (ML):                      {metrics['ML']:>3d} ({metrics['ML_ratio']:>5.1f}%)")
        
        print(f"\nError Statistics:")
        print(f"  False Positives (FP):                  {metrics['FP']:>6d}")
        print(f"  False Negatives (FN/Misses):           {metrics['FN']:>6d}")
        print(f"  ID Switches:                           {metrics['ID_Sw']:>6d}")
        
        print(f"\nDataset Statistics:")
        print(f"  Ground Truth Trajectories:             {metrics['Num_GT_IDs']:>6d}")
        print(f"  Predicted Trajectories:                {metrics['Num_Pred_IDs']:>6d}")
        print(f"  Total Frames:                          {metrics['Num_Frames']:>6d}")
        print(f"  Total GT Detections:                   {metrics['Num_GT_Dets']:>6d}")
        print(f"  Total Predicted Detections:            {metrics['Num_Pred_Dets']:>6d}")
        
        print(f"{'='*70}\n")
