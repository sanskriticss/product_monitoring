import cv2
from yolo_segmentation import YOLOSEG
import cvzone
from tracker import*
import numpy as np

# Initialize YOLO with correct input size expectations
ys = YOLOSEG("apples.pt")
ys.conf_threshold = 0.2
tracker = Tracker()

cap = cv2.VideoCapture('apples.mp4')

# Get original frame dimensions
ret, test_frame = cap.read()
orig_height, orig_width = test_frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Define counting area - adjusted to match the conveyor belt position
area = [
    (400, 200),    # Top left
    (400, 500),    # Bottom left
    (450, 500),    # Bottom right
    (450, 200)     # Top right
]

healthy_counter = []
defective_counter = []
tracked_objects = {}

def scale_to_original(box, orig_width, orig_height):
    x1, y1, x2, y2 = map(int, box)
    x1 = int(x1 * orig_width / 416)
    x2 = int(x2 * orig_width / 416)
    y1 = int(y1 * orig_height / 256)
    y2 = int(y2 * orig_height / 256)
    return (x1, y1, x2, y2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize for detection while keeping original for display
    input_frame = cv2.resize(frame, (416, 256))
    
    # Get detections
    bboxes, classes, segmentations, scores = ys.detect(input_frame)
    
    # Scale boxes back to original size
    scaled_boxes = []
    for box in bboxes:
        scaled_boxes.append(scale_to_original(box, orig_width, orig_height))
    
    # Update tracker with scaled boxes
    bbox_idx = tracker.update(scaled_boxes)
    
    current_tracked = set()
    
    # Process each tracked object
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        
        # Calculate center point
        cx = int((x3 + x4) // 2)
        cy = int((y3 + y4) // 2)
        
        # Check if apple crosses counting line
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        
        if result >= 0:
            if id not in tracked_objects:
                tracked_objects[id] = 1
            else:
                tracked_objects[id] += 1
            
            current_tracked.add(id)
            
            if tracked_objects[id] >= 2:
                roi = frame[y3:y4, x3:x4]
                if roi.size > 0:
                    avg_color = np.mean(roi, axis=(0, 1))
                    
                    # Enhance color detection threshold
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    red_mask = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([10, 255, 255]))
                    red_ratio = np.sum(red_mask) / (roi.shape[0] * roi.shape[1])
                    
                    if red_ratio > 0.3:  # Adjusted threshold for healthy apples
                        color = (0, 255, 0)  # Green for healthy
                        if id not in healthy_counter:
                            healthy_counter.append(id)
                    else:
                        color = (0, 0, 255)  # Red for defective
                        if id not in defective_counter:
                            defective_counter.append(id)
                    
                    # Draw detection box and ID
                    cv2.rectangle(frame, (x3, y3), (x4, y4), color, 3)
                    cv2.putText(frame, f'{id}', (x3, y3-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
    
    # Remove old tracks
    tracked_ids = list(tracked_objects.keys())
    for id in tracked_ids:
        if id not in current_tracked:
            del tracked_objects[id]
    
    # Draw counting area
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 255, 255), 2)
    
    # Display counts
    cvzone.putTextRect(frame, f'Healthy: {len(healthy_counter)}', (50, 60), 2, 2, (0, 255, 0))
    cvzone.putTextRect(frame, f'Defective: {len(defective_counter)}', (50, 120), 2, 2, (0, 0, 255))
    cvzone.putTextRect(frame, f'Current Detections: {len(bboxes)}', (50, 180), 2, 2)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()