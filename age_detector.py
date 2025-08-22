

import sys
import subprocess

def install_and_import(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing missing package: {pip_name or package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or package])
        __import__(package)


install_and_import("cv2", "opencv-python")
install_and_import("numpy")
install_and_import("pathlib")
install_and_import("imutils")
install_and_import("requests")

import cv2
import numpy as np
import os
from pathlib import Path

class SimpleAgeDetector:
    
    def __init__(self, models_dir="models", confidence_threshold=0.7):
        self.models_dir = Path(models_dir)
        self.confidence_threshold = confidence_threshold
        
        # Age buckets
        self.AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        # Model preprocessing values
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load face detection and age classification models"""
        try:
            # Model file paths
            face_proto = self.models_dir / "deploy.prototxt"
            face_model = self.models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
            age_proto = self.models_dir / "age_deploy.prototxt"
            age_model = self.models_dir / "age_net.caffemodel"
            
            # Check if files exist
            for model_file in [face_proto, face_model, age_proto, age_model]:
                if not model_file.exists():
                    print(f" Model file not found: {model_file}")
                    exit(1)
            
            # Load networks
            print("Loading models...")
            self.face_net = cv2.dnn.readNet(str(face_model), str(face_proto))
            self.age_net = cv2.dnn.readNet(str(age_model), str(age_proto))
            print("Models loaded")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            exit(1)
    
    def detect_faces(self, image):
        """Detect faces in image"""
        (h, w) = image.shape[:2]
        
        # Create blob and detect faces
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Keep coordinates within bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                faces.append((startX, startY, endX, endY, confidence))
        
        return faces
    
    def predict_age(self, face_roi):
        """Predict age from face region"""
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), 
                                   self.MODEL_MEAN_VALUES, swapRB=False)
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        
        age_idx = np.argmax(age_preds)
        return self.AGE_BUCKETS[age_idx], age_preds[0][age_idx]
    
    def process_image(self, image_path):
        """Process single image"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        print(f" Processing: {image_path.name}")
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if not faces:
            print("No faces detected!")
            return
        
        # Process each face
        result_image = image.copy()
        
        for i, (startX, startY, endX, endY, face_conf) in enumerate(faces):
            # Extract face
            face_roi = image[startY:endY, startX:endX]
            
            if face_roi.size > 0:
                # Predict age
                predicted_age, age_conf = self.predict_age(face_roi)
                
                # Print result
                print(f"👤 Face {i+1}: Age {predicted_age} (confidence: {age_conf:.3f})")
                
                # Draw on image
                cv2.rectangle(result_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                
                # Add age label
                label = f"Age: {predicted_age}"
                label_y = startY - 10 if startY > 20 else startY + 25
                
                # Label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result_image, (startX, label_y - h - 5), 
                            (startX + w, label_y + 5), (0, 255, 0), -1)
                
                # Label text
                cv2.putText(result_image, label, (startX, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save processed image
        output_path = Path("results") / f"processed_{image_path.name}"
        output_path.parent.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_path), result_image)
        print(f"Saved: {output_path}")
        print()

def main():
    """Main function"""
    print("Simple Age Detection System")
    print("=" * 35)
    
    # Initialize detector
    detector = SimpleAgeDetector()
    
    # Process images from test_images directory
    test_dir = Path("test_images")
    
    if not test_dir.exists():
        print(f" Directory not found: {test_dir}")
        return
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in test_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f" No images found in {test_dir}")
        return
    
    print(f"Found {len(image_files)} image(s)")
    print()
    
    # Process each image
    for image_file in image_files:
        detector.process_image(image_file)
    
    print("done")

if __name__ == "__main__":
    main()
    print("Please check the Processed image in results/   , Thank you ")
