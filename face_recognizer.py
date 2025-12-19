import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import warnings
from typing import List, Dict, Any, Optional, Tuple
import logging
from config import settings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/face_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceRecognizer:
    """Core face recognition engine"""
    
    def __init__(self):
        logger.info("Initializing Face Recognition Engine...")
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=settings.DETECTION_CONFIDENCE
        )
        
        # Load face recognition model
        self.face_model = self._load_face_model()
        
        # Settings
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        
        logger.info("Face Recognition Engine initialized")
    
    def _load_face_model(self) -> nn.Module:
        """Load pre-trained ResNet for face recognition"""
        try:
            import torchvision.models as models
            model = models.resnet18(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            model.eval()
            logger.info("ResNet18 model loaded")
            return model
        except Exception as e:
            logger.error(f"Failed to load ResNet model: {e}")
            raise
    
    def extract_face_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract deep embedding from face image"""
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Convert to PIL
            pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            
            # Apply transforms
            image_tensor = transform(pil_image).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                embedding = self.face_model(image_tensor)
            
            # Convert to numpy and normalize
            embedding = embedding.squeeze().numpy()
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            # Fallback: simple features
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
    
    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        try:
            # Ensure same dimension
            min_len = min(len(emb1), len(emb2))
            emb1_trunc = emb1[:min_len]
            emb2_trunc = emb2[:min_len]
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_trunc, emb2_trunc) / (
                np.linalg.norm(emb1_trunc) * np.linalg.norm(emb2_trunc)
            )
            
            return float(similarity)
        except:
            return 0.0
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image using MediaPipe"""
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.face_detection.process(rgb_image)
        
        faces = []
        
        if results.detections:
            for detection in results.detections:
                confidence = detection.score[0]
                
                if confidence < settings.DETECTION_CONFIDENCE:
                    continue
                
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                box_width = int(bboxC.width * w)
                box_height = int(bboxC.height * h)
                
                # Add padding
                padding = int(min(box_width, box_height) * 0.2)
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x1 + box_width + 2*padding)
                y2 = min(h, y1 + box_height + 2*padding)
                
                # Extract face
                face_roi = image[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'face': face_roi,
                        'confidence': float(confidence)
                    })
        
        return faces
    
    def set_target_person(self, image_path: str, person_name: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Set target person from image"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False, {"error": "Cannot read image"}
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if len(faces) == 0:
                return False, {"error": "No face detected"}
            
            # Use the face with highest confidence
            best_face = max(faces, key=lambda x: x['confidence'])
            
            # Extract embedding
            target_embedding = self.extract_face_embedding(best_face['face'])
            
            # Save target face
            target_filename = f"target_{person_name or os.path.basename(image_path).split('.')[0]}.jpg"
            target_path = os.path.join(settings.RESULTS_DIR, "snapshots", target_filename)
            cv2.imwrite(target_path, best_face['face'])
            
            # Save embedding
            embedding_path = target_path.replace('.jpg', '.npy')
            np.save(embedding_path, target_embedding)
            
            result = {
                "success": True,
                "target_name": person_name,
                "face_image_path": target_path,
                "embedding_path": embedding_path,
                "face_size": best_face['face'].shape,
                "confidence": best_face['confidence']
            }
            
            logger.info(f"Target person set: {person_name}")
            return True, result
            
        except Exception as e:
            logger.error(f"Error setting target person: {e}")
            return False, {"error": str(e)}
    
    def search_in_video(
        self, 
        video_path: str, 
        target_embedding: np.ndarray,
        target_name: str = "Target",
        camera_name: str = "CCTV",
        frame_skip: int = None,
        similarity_threshold: float = None
    ) -> Dict[str, Any]:
        """Search for target person in video"""
        try:
            # Use provided settings or defaults
            frame_skip = frame_skip or settings.FRAME_SKIP
            similarity_threshold = similarity_threshold or self.similarity_threshold
            
            logger.info(f"Starting search in video: {video_path}")
            logger.info(f"Settings: threshold={similarity_threshold}, frame_skip={frame_skip}")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Cannot open video"}
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0:
                fps = 30
            
            logger.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Prepare output video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{camera_name}_{timestamp}.mp4"
            output_path = os.path.join(settings.RESULTS_DIR, "evidence", output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps/frame_skip, (width, height))
            
            # Tracking
            matches = []
            frame_count = 0
            match_count = 0
            start_time = time.time()
            
            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for speed
                if frame_count % frame_skip != 0:
                    continue
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Check each face
                for face in faces:
                    # Extract embedding
                    face_embedding = self.extract_face_embedding(face['face'])
                    
                    # Calculate similarity
                    similarity = self.calculate_similarity(target_embedding, face_embedding)
                    
                    # Check if it's a match
                    if similarity >= similarity_threshold:
                        match_count += 1
                        match_info = {
                            'bbox': face['bbox'],
                            'similarity': similarity,
                            'frame': frame_count,
                            'time': frame_count / fps
                        }
                        matches.append(match_info)
                        
                        # Save first match snapshot
                        if match_count == 1:
                            snap_filename = f"{camera_name}_match_{timestamp}.jpg"
                            snap_path = os.path.join(settings.RESULTS_DIR, "snapshots", snap_filename)
                            cv2.imwrite(snap_path, face['face'])
                        
                        # Save alert
                        alert_filename = f"{camera_name}_alert_{match_count}_{timestamp}.jpg"
                        alert_path = os.path.join(settings.RESULTS_DIR, "alerts", alert_filename)
                        cv2.imwrite(alert_path, frame)
                
                # Draw results and write to output
                result_frame = self._draw_detections(frame, faces, matches[-1] if matches else None, 
                                                     camera_name, frame_count, total_frames, fps)
                out.write(result_frame)
            
            # Cleanup
            cap.release()
            out.release()
            
            processing_time = time.time() - start_time
            
            # Generate report
            report_data = self._generate_report_data(
                camera_name, video_path, output_path, matches, 
                processing_time, similarity_threshold, target_name
            )
            
            # Save report
            report_filename = f"{camera_name}_report_{timestamp}.json"
            report_path = os.path.join(settings.RESULTS_DIR, "reports", report_filename)
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            result = {
                "success": True,
                "matches_found": match_count,
                "processing_time": processing_time,
                "output_video_path": output_path,
                "report_path": report_path,
                "match_details": matches,
                "video_info": {
                    "fps": fps,
                    "resolution": f"{width}x{height}",
                    "total_frames": total_frames,
                    "duration": total_frames / fps
                }
            }
            
            logger.info(f"Search completed: {match_count} matches found in {processing_time:.1f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error searching in video: {e}")
            return {"error": str(e)}
    
    def _draw_detections(self, frame: np.ndarray, faces: List[Dict], match_info: Optional[Dict], 
                        camera_name: str, current_frame: int, total_frames: int, fps: int) -> np.ndarray:
        """Draw detections on frame"""
        result = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw all detected faces
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            confidence = face['confidence']
            
            # Check if this face is a match
            is_match = False
            if match_info and match_info['bbox'] == (x1, y1, x2, y2):
                is_match = True
            
            if is_match:
                # MATCH - Green box
                color = (0, 255, 0)
                thickness = 3
                label = f"Match: {match_info['similarity']:.1%}"
            else:
                # NOT A MATCH - Gray box
                color = (100, 100, 100)
                thickness = 1
                label = f"Face: {confidence:.0%}"
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result,
                         (x1, y2 - label_size[1] - 5),
                         (x1 + label_size[0], y2),
                         color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(result, label, (x1, y2 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add status bar
        status_bar = np.zeros((50, width, 3), dtype=np.uint8)
        status_bar[:] = (30, 30, 30)
        result[:50, :] = status_bar
        
        # Camera name
        cv2.putText(result, f"Camera: {camera_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Frame counter
        frame_text = f"Frame: {current_frame}/{total_frames}"
        cv2.putText(result, frame_text, (width // 3, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Time
        time_text = f"Time: {current_frame/max(fps, 1):.1f}s"
        cv2.putText(result, time_text, (width // 2, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result, timestamp, (width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def _generate_report_data(self, camera_name: str, video_path: str, output_path: str,
                             matches: List[Dict], processing_time: float, 
                             similarity_threshold: float, target_name: str) -> Dict[str, Any]:
        """Generate report data"""
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "target_person": target_name,
            "camera_name": camera_name,
            "video_source": video_path,
            "output_video": output_path,
            "similarity_threshold": similarity_threshold,
            "matches_found": len(matches),
            "processing_time": processing_time,
            "match_details": matches
        }
        
        if matches:
            best_match = max(matches, key=lambda x: x['similarity'])
            report["best_match"] = {
                "similarity": best_match['similarity'],
                "frame": best_match['frame'],
                "time": best_match['time']
            }
        
        return report

# Global instance
face_recognizer = FaceRecognizer()