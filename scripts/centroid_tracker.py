# centroid_tracker_fixed.py
# FIXED Centroid-based tracking algorithm with proper crossing detection

import numpy as np
from scipy.spatial import distance
from collections import OrderedDict

class CentroidTracker:
    """
    FIXED centroid-based object tracking algorithm.
    Improved to prevent multiple counting of same object.
    """
    
    def __init__(self, max_disappeared=50, max_distance=50):
        """
        Initialize centroid tracker.
        
        Args:
            max_disappeared (int): Max frames object can disappear before deregister
            max_distance (float): Max Euclidean distance for matching centroids
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # {object_id: centroid}
        self.disappeared = OrderedDict()  # {object_id: frames_disappeared}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid):
        """Register a new object."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, rects):
        """
        Update tracker with new detections.
        
        Args:
            rects (list): List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            OrderedDict: Tracked objects with their IDs and centroids
        """
        # Compute centroids for current detections
        input_centroids = np.zeros((len(rects), 2))
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = (x1 + x2) // 2
            cY = (y1 + y2) // 2
            input_centroids[i] = (cX, cY)
        
        # If no current objects, register all detections
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Match existing objects with detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distance between centroids
            D = distance.cdist(np.array(object_centroids), input_centroids)
            
            # Find minimum distance associations
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Update existing objects
            for row, col in zip(rows, cols):
                if D[row, col] > self.max_distance:
                    continue
                
                if row in used_rows or col in used_cols:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Deregister disappeared objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        return self.objects


class LineCounterFixed:
    """
    FIXED: Count objects crossing a line with proper state tracking.
    Prevents multiple counting of the same object.
    """
    
    def __init__(self, line_start, line_end, buffer_pixels=30):
        """
        Initialize line counter with proper crossing detection.
        
        Args:
            line_start (tuple): (x, y) coordinates of line start
            line_end (tuple): (x, y) coordinates of line end
            buffer_pixels (int): Pixels away from line to reset crossing state
        """
        self.line_start = line_start
        self.line_end = line_end
        self.line_y = line_start[1]  # Assuming horizontal line
        self.buffer_pixels = buffer_pixels
        
        self.count_up = 0
        self.count_down = 0
        
        # Track state for each object: {id: {'side': 'above'/'below', 'crossed': False}}
        self.object_states = {}
    
    def update_and_count(self, objects):
        """
        Update tracked objects and count crossings.
        
        Key improvement: Only count ONCE per crossing using state machine
        
        Args:
            objects (OrderedDict): {object_id: (cx, cy)}
            
        Returns:
            tuple: (count_up, count_down, objects_dict)
        """
        
        # Clean up disappeared objects
        disappeared_ids = [oid for oid in self.object_states if oid not in objects]
        for oid in disappeared_ids:
            del self.object_states[oid]
        
        # Update states and check for crossings
        for object_id, (cx, cy) in objects.items():
            if object_id not in self.object_states:
                # New object - determine initial side
                side = 'above' if cy < self.line_y else 'below'
                self.object_states[object_id] = {
                    'side': side,
                    'crossed': False,
                    'last_y': cy
                }
            else:
                state = self.object_states[object_id]
                prev_side = state['side']
                prev_y = state['last_y']
                
                # Determine current side
                curr_side = 'above' if cy < self.line_y else 'below'
                
                # Check if crossed line
                if prev_side != curr_side and not state['crossed']:
                    # Object crossed the line!
                    if curr_side == 'below':
                        self.count_down += 1
                    else:
                        self.count_up += 1
                    
                    # Mark as crossed to prevent double counting
                    state['crossed'] = True
                
                # Reset crossed flag once object moves far enough from line
                distance_from_line = abs(cy - self.line_y)
                if distance_from_line > self.buffer_pixels:
                    state['crossed'] = False
                
                # Update state
                state['side'] = curr_side
                state['last_y'] = cy
        
        return self.count_up, self.count_down, objects


class SmartLineCounter:
    """
    ALTERNATIVE: More robust line counting with directional zones.
    
    Pattern:
        ↑ UP DETECTION ZONE
    ═══════════════════════════════════════
        NEUTRAL ZONE (buffer)
    ═══════════════════════════════════════
        ↓ DOWN DETECTION ZONE
    
    Only counts when entering detection zones from neutral zone.
    """
    
    def __init__(self, line_y, neutral_zone_height=60):
        """
        Initialize smart line counter with zones.
        
        Args:
            line_y (int): Y coordinate of counting line
            neutral_zone_height (int): Height of neutral zone around line
        """
        self.line_y = line_y
        self.neutral_zone_height = neutral_zone_height
        
        self.up_zone_start = line_y - neutral_zone_height
        self.up_zone_end = line_y
        
        self.down_zone_start = line_y
        self.down_zone_end = line_y + neutral_zone_height
        
        self.count_up = 0
        self.count_down = 0
        
        # Track which zone each object is in
        self.object_zones = {}  # {id: 'up'/'neutral'/'down'}
    
    def get_zone(self, cy):
        """Determine which zone a centroid is in."""
        if cy < self.up_zone_start:
            return 'up'
        elif cy < self.up_zone_end:
            return 'neutral_up'
        elif cy < self.down_zone_end:
            return 'neutral_down'
        else:
            return 'down'
    
    def update_and_count(self, objects):
        """
        Update with zone-based counting.
        
        Args:
            objects (OrderedDict): {object_id: (cx, cy)}
            
        Returns:
            tuple: (count_up, count_down, objects_dict)
        """
        
        # Clean up disappeared objects
        disappeared_ids = [oid for oid in self.object_zones if oid not in objects]
        for oid in disappeared_ids:
            del self.object_zones[oid]
        
        # Update zones and count transitions
        for object_id, (cx, cy) in objects.items():
            curr_zone = self.get_zone(cy)
            
            if object_id not in self.object_zones:
                # New object
                self.object_zones[object_id] = curr_zone
            else:
                prev_zone = self.object_zones[object_id]
                
                # Count only when entering detection zones from neutral zone
                # Coming from UP → entering DOWN (moving down)
                if prev_zone in ['up', 'neutral_up'] and curr_zone in ['neutral_down', 'down']:
                    self.count_down += 1
                
                # Coming from DOWN → entering UP (moving up)
                elif prev_zone in ['down', 'neutral_down'] and curr_zone in ['neutral_up', 'up']:
                    self.count_up += 1
                
                # Update zone
                self.object_zones[object_id] = curr_zone
        
        return self.count_up, self.count_down, objects
