import cv2
from yolo_segmentation import YOLOSEG
import cvzone
from tracker import*
import numpy as np

ys = YOLOSEG("milkcarton.pt")
ys.conf_threshold = 0.3  

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

cap = cv2.VideoCapture('cy.mp4')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
tracker = Tracker()

area = [
    (400, 0),    
    (400, 720),  
    (420, 720),  
    (420, 0)     
]

counter1 = []
tracked_objects = {} 
min_frames = 4  
last_positions = {} 
min_distance = 100  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1280, 720))
    
    bboxes, classes, segmentations, scores = ys.detect(frame)
    bbox_idx = tracker.update(bboxes)
    
    current_tracked = set()
    
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        if result >= 0:
            if id not in tracked_objects:
                tracked_objects[id] = 1
            else:
                tracked_objects[id] += 1
            
            current_tracked.add(id)
            
            if tracked_objects[id] >= min_frames:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 3)
                # Removed the ID display line that was here
                
                if counter1.count(id) == 0:
                    counter1.append(id)
    
    tracked_ids = list(tracked_objects.keys())
    for id in tracked_ids:
        if id not in current_tracked:
            del tracked_objects[id]
    
    detection_area = np.array(area, np.int32)
    
    ca1 = len(counter1)
    cvzone.putTextRect(frame, f'Count: {ca1}', (50, 60), 3, 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()