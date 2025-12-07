# yolo_detector.py
# YOLOv8 detection wrapper for pedestrian detection

from ultralytics import YOLO
import cv2
import numpy as np

class YOLODetector:
    """
    YOLOv8 detector wrapper for pedestrian detection.
    """
    
    def __init__(self, model_name="yolov8m.pt", conf=0.5, device=0):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_name (str): Model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            conf (float): Confidence threshold (0-1)
            device (int or str): Device (0 for GPU, 'cpu' for CPU)
        """
        self.model = YOLO(model_name)
        self.conf = conf
        self.device = device
    
    def detect(self, frame, classes=[0]):
        """
        Detect pedestrians in frame.
        
        Args:
            frame (np.array): Input frame
            classes (list): Class IDs to detect (0 = person in COCO)
            
        Returns:
            list: List of bounding boxes [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            for detection in result.boxes:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                conf = float(detection.conf[0])
                cls_id = int(detection.cls[0])
                
                # Filter by class (0 = person)
                if cls_id in classes:
                    detections.append([x1, y1, x2, y2, conf, cls_id])
        
        return detections


class FrameProcessor:
    """
    Process frames for visualization and counting.
    """
    
    @staticmethod
    def draw_detections(frame, detections, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes on frame.
        
        Args:
            frame (np.array): Input frame
            detections (list): List of detections [x1, y1, x2, y2, conf, cls_id]
            color (tuple): BGR color for boxes
            thickness (int): Line thickness
            
        Returns:
            np.array: Frame with drawn boxes
        """
        for x1, y1, x2, y2, conf, cls_id in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"Person {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    @staticmethod
    def draw_tracks(frame, objects, color=(0, 255, 255), thickness=2):
        """
        Draw tracked objects with IDs.
        
        Args:
            frame (np.array): Input frame
            objects (dict): {object_id: (cx, cy)}
            color (tuple): BGR color
            thickness (int): Line thickness
            
        Returns:
            np.array: Frame with drawn tracks
        """
        for obj_id, (cx, cy) in objects.items():
            cx = int(cx)
            cy = int(cy)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(frame, f"ID{obj_id}", (cx - 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    @staticmethod
    def draw_line(frame, line_start, line_end, color=(0, 0, 255), thickness=2):
        """
        Draw counting line on frame.
        
        Args:
            frame (np.array): Input frame
            line_start (tuple): (x, y) start point
            line_end (tuple): (x, y) end point
            color (tuple): BGR color
            thickness (int): Line thickness
            
        Returns:
            np.array: Frame with drawn line
        """
        cv2.line(frame, line_start, line_end, color, thickness)
        return frame
    
    @staticmethod
    def draw_counts(frame, count_up, count_down, position=(10, 30)):
        """
        Draw count information on frame.
        
        Args:
            frame (np.array): Input frame
            count_up (int): Number crossing up
            count_down (int): Number crossing down
            position (tuple): Position for text
            
        Returns:
            np.array: Frame with count text
        """
        cv2.putText(frame, f"Down: {count_down}", position,
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Up: {count_up}", (position[0], position[1] + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
