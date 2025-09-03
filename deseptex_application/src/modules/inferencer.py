import numpy as np
import cv2 as cv

from ultralytics import YOLO



class inferencer_t:
    def __init__(self, classifier_model_path, classifier_image_shape, pants_avant_model_path, pants_arriere_model_path, chemises_model_path, segmentation_image_shape):
        self.classifier_model = YOLO(classifier_model_path)
        self.classifier_image_shape = classifier_image_shape
        self.pants_avant_model = YOLO(pants_avant_model_path)
        self.pants_arriere_model = YOLO(pants_arriere_model_path)
        self.chemises_model = YOLO(chemises_model_path)
        self.segmentation_image_shape = segmentation_image_shape
        
        self.chemises_indices = [1, 3] # boutons, extremites manches
        self.pants_avant_indices = [1, 2, 5] # boutons, rivets, tirette
        self.pants_arriere_indices = [1, 2] # boutons, rivets
    
    
    def run_inference(self, image):
        # First, classify the image to determine which segmentation model to use
        classification_result = self.classifier_model.predict(image, conf=0.25, iou=0.6, imgsz=self.classifier_image_shape, verbose=False)[0]
        
        # Get probabilities and class names
        if classification_result.probs is None:
            return None, None

        if classification_result.probs.data.is_cuda:
            probs = classification_result.probs.data.cpu().numpy()
        else:
            probs = classification_result.probs.data.numpy()
        class_names = classification_result.names
        
        # get highest probability class
        max_index = probs.argmax()
        selected_class = class_names[max_index]
        
        # Based on the selected class, choose the appropriate segmentation model
        if selected_class == "pants_avant":
            segmentation_model = self.pants_avant_model
            relevant_indices = self.pants_avant_indices
        elif selected_class == "pants_arriere":
            segmentation_model = self.pants_arriere_model
            relevant_indices = self.pants_arriere_indices
        elif selected_class == "chemise":
            segmentation_model = self.chemises_model
            relevant_indices = self.chemises_indices
        else:
            return None, None
        
        # Run segmentation
        segmentation_result = segmentation_model.predict([image], conf=0.25, iou=0.6, imgsz=self.segmentation_image_shape, verbose=False)[0]
        return segmentation_result, relevant_indices
        
    
    def run_inference_without_postproc(self, image):
        result, _ = self.run_inference(image)
        contours = []
        
        if result is None:
            return contours
        
        if result.masks is None:
            return contours
        
        for index, mask in enumerate(result.masks): #type: ignore
            contour = mask.xy.pop()
            contour = contour.astype(np.int32)
            contours.append(contour)
            
        return contours
    
    
    def run_inference_with_postproc(self, image):
        result, relevant_indices = self.run_inference(image)
        
        if result is None or relevant_indices is None:
            return []
        
        if result.masks is None or result.boxes is None:
            return []
        
        
        # only get the masks for the classes in the relevant indices
        masks = result.masks
        masks_xy = masks.xy
        mask_image = np.zeros_like(image)
        
        boxes = result.boxes
        for i in range(len(boxes)):
            predicted_class = int(boxes.cls[i].item())
            if predicted_class in relevant_indices:
                cv.fillPoly(mask_image, np.array([masks_xy[i]], dtype=np.int32), (0, 255, 0)) #type: ignore
        
        # dilation
        kernel = np.ones((3,3), np.uint8)
        dilated_mask_image = cv.dilate(mask_image, kernel, iterations=12)
        
        # find contours in dilated mask image
        gray = cv.cvtColor(dilated_mask_image, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [contour.reshape(-1, 2) for contour in contours]
        return contours