from ultralytics import YOLO
import numpy as np

class YOLOSEG:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        height, width, channels = img.shape
        segmentation_contours_idx = []
        
        # Run inference
        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
        
        # Initialize empty arrays in case no detections
        bboxes = np.array([])
        class_ids = np.array([])
        scores = np.array([])
        
        if len(result) > 0:  # If there are any detections
            # Handle segmentation masks if they exist
            if hasattr(result, 'masks') and result.masks is not None:
                for seg in result.masks.xyn:
                    # Scale segments to image dimensions
                    seg[:, 0] *= width
                    seg[:, 1] *= height
                    segment = np.array(seg, dtype=np.int32)
                    segmentation_contours_idx.append(segment)
            
            # Get bounding boxes
            if result.boxes is not None and len(result.boxes.xyxy) > 0:
                bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
                class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
                scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

        return bboxes, class_ids, segmentation_contours_idx, scores