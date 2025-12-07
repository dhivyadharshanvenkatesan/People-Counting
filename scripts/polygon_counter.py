# polygon_counter.py
# Polygon-based counting system for flexible ROI selection

import cv2
import numpy as np
from collections import OrderedDict


class PolygonCounter:
    """
    Count objects crossing through a user-defined polygon region.
    Uses point-in-polygon tests and directional tracking.
    """
    
    def __init__(self, polygon_points, buffer_distance=30):
        """
        Initialize polygon counter.
        
        Args:
            polygon_points (list): List of (x, y) tuples defining polygon vertices
            buffer_distance (int): Distance threshold for state reset
        """
        self.polygon_points = np.array(polygon_points, dtype=np.int32)
        self.buffer_distance = buffer_distance
        
        # Compute polygon centroid for direction determination
        self.centroid = np.mean(self.polygon_points, axis=0)
        
        # Tracking state for each object
        self.object_states = {}  # {id: {'position': 'inside'/'outside', 'last_pos': (x,y), 'crossed': False}}
        
        # Counters
        self.count_in = 0   # Objects entering the polygon
        self.count_out = 0  # Objects leaving the polygon
        
    def point_in_polygon(self, point):
        """
        Check if a point is inside the polygon using ray casting algorithm.
        
        Args:
            point (tuple): (x, y) coordinates
            
        Returns:
            bool: True if point is inside polygon
        """
        result = cv2.pointPolygonTest(self.polygon_points, point, False)
        return result >= 0  # >= 0 means inside or on boundary
    
    def distance_to_polygon(self, point):
        """
        Calculate minimum distance from point to polygon boundary.
        
        Args:
            point (tuple): (x, y) coordinates
            
        Returns:
            float: Distance to nearest polygon edge
        """
        return abs(cv2.pointPolygonTest(self.polygon_points, point, True))
    
    def update_and_count(self, objects):
        """
        Update tracked objects and count polygon crossings.
        
        Args:
            objects (OrderedDict): {object_id: (cx, cy)}
            
        Returns:
            tuple: (count_in, count_out, objects_dict)
        """
        # Clean up disappeared objects
        disappeared_ids = [oid for oid in self.object_states if oid not in objects]
        for oid in disappeared_ids:
            del self.object_states[oid]
        
        # Update states and check for crossings
        for object_id, (cx, cy) in objects.items():
            point = (int(cx), int(cy))
            is_inside = self.point_in_polygon(point)
            
            if object_id not in self.object_states:
                # New object - determine initial position
                position = 'inside' if is_inside else 'outside'
                self.object_states[object_id] = {
                    'position': position,
                    'last_pos': point,
                    'crossed': False,
                    'entry_side': None
                }
            else:
                state = self.object_states[object_id]
                prev_position = state['position']
                prev_point = state['last_pos']
                
                # Determine current position
                curr_position = 'inside' if is_inside else 'outside'
                
                # Check if crossed polygon boundary
                if prev_position != curr_position and not state['crossed']:
                    # Object crossed the polygon boundary!
                    if curr_position == 'inside':
                        self.count_in += 1
                    else:
                        self.count_out += 1
                    
                    # Mark as crossed to prevent double counting
                    state['crossed'] = True
                
                # Reset crossed flag once object moves far enough from boundary
                distance_to_boundary = self.distance_to_polygon(point)
                if distance_to_boundary > self.buffer_distance:
                    state['crossed'] = False
                
                # Update state
                state['position'] = curr_position
                state['last_pos'] = point
        
        return self.count_in, self.count_out, objects
    
    def draw_polygon(self, frame, color=(0, 255, 255), thickness=2):
        """
        Draw the polygon ROI on the frame.
        
        Args:
            frame (np.array): Input frame
            color (tuple): BGR color
            thickness (int): Line thickness
            
        Returns:
            np.array: Frame with drawn polygon
        """
        cv2.polylines(frame, [self.polygon_points], isClosed=True, 
                     color=color, thickness=thickness)
        
        # Draw polygon vertices
        for i, point in enumerate(self.polygon_points):
            cv2.circle(frame, tuple(point), 5, color, -1)
            cv2.putText(frame, str(i+1), (point[0]+10, point[1]+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_counts(self, frame, position=(10, 30)):
        """
        Draw count information on frame.
        
        Args:
            frame (np.array): Input frame
            position (tuple): Position for text
            
        Returns:
            np.array: Frame with count text
        """
        cv2.putText(frame, f"IN: {self.count_in}", position,
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {self.count_out}", 
                   (position[0], position[1] + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"TOTAL: {self.count_in + self.count_out}", 
                   (position[0], position[1] + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame


class PolygonSelector:
    """
    Interactive polygon selection tool using mouse callbacks.
    """
    
    def __init__(self, window_name="Select ROI Polygon"):
        """
        Initialize polygon selector.
        
        Args:
            window_name (str): Name of the OpenCV window
        """
        self.window_name = window_name
        self.points = []
        self.max_points = 4
        self.image = None
        self.image_copy = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for point selection.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < self.max_points:
                self.points.append((x, y))
                
                # Draw the point
                cv2.circle(self.image_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.image_copy, str(len(self.points)), 
                           (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                # Draw line to previous point
                if len(self.points) > 1:
                    cv2.line(self.image_copy, self.points[-2], self.points[-1],
                            (0, 255, 255), 2)
                
                # Complete the polygon when 4 points are selected
                if len(self.points) == self.max_points:
                    cv2.line(self.image_copy, self.points[-1], self.points[0],
                            (0, 255, 255), 2)
                    cv2.putText(self.image_copy, "Press 'c' to confirm or 'r' to reset", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                
                cv2.imshow(self.window_name, self.image_copy)
    
    def select_polygon(self, frame):
        """
        Allow user to select 4 points defining a polygon ROI.
        
        Args:
            frame (np.array): First frame of video
            
        Returns:
            list: List of (x, y) tuples or None if cancelled
        """
        self.image = frame.copy()
        self.image_copy = frame.copy()
        self.points = []
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Instructions
        instruction_img = self.image_copy.copy()
        cv2.putText(instruction_img, "Click 4 points to define ROI polygon", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(self.window_name, instruction_img)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and len(self.points) == self.max_points:
                # Confirm selection
                cv2.destroyWindow(self.window_name)
                return self.points
            
            elif key == ord('r'):
                # Reset selection
                self.points = []
                self.image_copy = self.image.copy()
                cv2.putText(self.image_copy, "Click 4 points to define ROI polygon", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                cv2.imshow(self.window_name, self.image_copy)
            
            elif key == 27:  # ESC to cancel
                cv2.destroyWindow(self.window_name)
                return None
        
        cv2.destroyWindow(self.window_name)
        return self.points
