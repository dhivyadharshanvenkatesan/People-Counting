# mot_dataset_processor.py
# MOT17 dataset processor for handling MOT format data

import os
import cv2
import numpy as np
from pathlib import Path

class MOT17Processor:
    """
    Process MOT17 dataset format and extract information.
    """
    
    @staticmethod
    def read_seqinfo(seqinfo_path):
        """
        Read seqinfo.ini file.
        
        Args:
            seqinfo_path (str): Path to seqinfo.ini
            
        Returns:
            dict: Sequence information
        """
        seqinfo = {}
        with open(seqinfo_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=')
                    seqinfo[key.strip()] = value.strip()
        
        return seqinfo
    
    @staticmethod
    def read_gt_txt(gt_path):
        """
        Read ground truth file (gt.txt).
        
        Format: <frame_id>,<id>,<x>,<y>,<width>,<height>,<conf>,<class>,<visibility>
        
        Args:
            gt_path (str): Path to gt.txt
            
        Returns:
            dict: {frame_id: [(id, x, y, w, h, conf, class, vis), ...]}
        """
        gt_data = {}
        
        if not os.path.exists(gt_path):
            return gt_data
        
        with open(gt_path, 'r') as f:
            for line in f:
                values = line.strip().split(',')
                if len(values) < 8:
                    continue
                
                frame_id = int(values[0])
                obj_id = int(values[1])
                x = float(values[2])
                y = float(values[3])
                w = float(values[4])
                h = float(values[5])
                conf = float(values[6])
                cls = int(values[7])
                vis = float(values[8]) if len(values) > 8 else 1.0
                
                if frame_id not in gt_data:
                    gt_data[frame_id] = []
                
                gt_data[frame_id].append((obj_id, x, y, w, h, conf, cls, vis))
        
        return gt_data
    
    @staticmethod
    def read_det_txt(det_path):
        """
        Read detection file (det.txt).
        
        Format: <frame_id>,<id>,<x>,<y>,<width>,<height>,<conf>,...
        
        Args:
            det_path (str): Path to det.txt
            
        Returns:
            dict: {frame_id: [(id, x, y, w, h, conf), ...]}
        """
        det_data = {}
        
        if not os.path.exists(det_path):
            return det_data
        
        with open(det_path, 'r') as f:
            for line in f:
                values = line.strip().split(',')
                if len(values) < 7:
                    continue
                
                frame_id = int(values[0])
                obj_id = int(values[1])
                x = float(values[2])
                y = float(values[3])
                w = float(values[4])
                h = float(values[5])
                conf = float(values[6])
                
                if frame_id not in det_data:
                    det_data[frame_id] = []
                
                det_data[frame_id].append((obj_id, x, y, w, h, conf))
        
        return det_data
    
    @staticmethod
    def load_sequence(seq_path):
        """
        Load complete MOT sequence.
        
        Args:
            seq_path (str): Path to sequence folder (e.g., MOT17/train/MOT17-02-DPM/)
            
        Returns:
            dict: Sequence data including frames, ground truth, detections, info
        """
        seq_path = Path(seq_path)
        
        # Read seqinfo
        seqinfo_file = seq_path / "seqinfo.ini"
        seqinfo = MOT17Processor.read_seqinfo(str(seqinfo_file))
        
        # Read ground truth
        gt_file = seq_path / "gt" / "gt.txt"
        gt_data = MOT17Processor.read_gt_txt(str(gt_file))
        
        # Read detections
        det_file = seq_path / "det" / "det.txt"
        det_data = MOT17Processor.read_det_txt(str(det_file))
        
        # Load frame paths
        img_dir = seq_path / seqinfo.get('imDir', 'img1')
        frame_files = sorted([f for f in os.listdir(img_dir) 
                             if f.endswith(('.jpg', '.png'))])
        
        return {
            'seqinfo': seqinfo,
            'gt': gt_data,
            'det': det_data,
            'img_dir': str(img_dir),
            'frames': frame_files
        }
    
    @staticmethod
    def get_frame(seq_data, frame_idx):
        """
        Load a specific frame from sequence.
        
        Args:
            seq_data (dict): Sequence data from load_sequence()
            frame_idx (int): Frame index (1-based or 0-based depending on dataset)
            
        Returns:
            tuple: (frame, gt_annotations, det_annotations)
        """
        frame_file = os.path.join(seq_data['img_dir'], seq_data['frames'][frame_idx])
        frame = cv2.imread(frame_file)
        
        # Frame ID is 1-based in MOT format
        frame_id = frame_idx + 1
        gt_ann = seq_data['gt'].get(frame_id, [])
        det_ann = seq_data['det'].get(frame_id, [])
        
        return frame, gt_ann, det_ann
    
    @staticmethod
    def convert_to_xyxy(x, y, w, h):
        """
        Convert bbox from (x, y, w, h) to (x1, y1, x2, y2).
        
        Args:
            x, y: Top-left corner
            w, h: Width and height
            
        Returns:
            tuple: (x1, y1, x2, y2)
        """
        return int(x), int(y), int(x + w), int(y + h)
    
    @staticmethod
    def convert_to_xywh(x1, y1, x2, y2):
        """
        Convert bbox from (x1, y1, x2, y2) to (x, y, w, h).
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            tuple: (x, y, w, h)
        """
        return x1, y1, x2 - x1, y2 - y1
