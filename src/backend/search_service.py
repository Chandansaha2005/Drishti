"""
Search Service - Face detection and matching in video files.
Core ML functionality for DRISTI system.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys
from pathlib import Path as PathlibPath
import warnings

# Suppress deprecation warnings from torch
warnings.filterwarnings('ignore', category=UserWarning)

# Add src to path for imports
sys.path.insert(0, str(PathlibPath(__file__).parent.parent.parent))

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class SearchService:
    """Service for searching lost persons in video footage using face recognition."""
    
    def __init__(self):
        """Initialize face detection and recognition models."""
        logger.info("Initializing SearchService...")
        
        # Initialize OpenCV Face Detection (Haar Cascade - more reliable than MediaPipe on Render)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load face recognition model
        self.face_model = self._load_face_model()
        
        # Settings
        self.similarity_threshold = 0.6  # Lower threshold for crowd detection
        self.frame_skip = 5  # Process every 5th frame for speed
        self.device = torch.device('cpu')  # Use CPU for Render compatibility
        
        logger.info("SearchService initialized successfully")
    
    def _load_face_model(self):
        """Load pre-trained ResNet18 for face recognition."""
        try:
            import torchvision.models as models
            from torchvision.models import ResNet18_Weights
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-1])
            model = model.to(self.device)
            model.eval()
            logger.info("ResNet18 face model loaded")
            return model
        except Exception as e:
            logger.warning(f"Could not load ResNet: {e}. Using histogram fallback.")
            return None
    
    def extract_face_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding (feature vector) from image.
        
        Uses ResNet18 if available, falls back to histogram-based features.
        """
        try:
            if self.face_model is None:
                return self._histogram_features(face_image)
            
            # Preprocess for ResNet18
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            image_tensor = transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.face_model(image_tensor)
            
            embedding = embedding.squeeze().cpu().numpy()
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Error in embedding extraction: {e}. Using fallback.")
            return self._histogram_features(face_image)
    
    @staticmethod
    def _histogram_features(face_image: np.ndarray) -> np.ndarray:
        """Fallback: Extract histogram-based features."""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return cv2.normalize(hist, hist).flatten()
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect all faces in a frame using OpenCV Haar Cascade.
        
        Returns:
            List of detected faces with bounding boxes and confidence scores.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detected_faces = []
        h, w, _ = frame.shape
        
        for (x, y, fw, fh) in faces:
            # Add padding around face
            x1 = max(0, x - 15)
            y1 = max(0, y - 15)
            x2 = min(w, x + fw + 15)
            y2 = min(h, y + fh + 15)
            
            # Extract face ROI
            face_roi = frame[y1:y2, x1:x2].copy()
            
            if face_roi.size > 0:
                detected_faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'face': face_roi,
                    'confidence': 0.9  # Haar cascade doesn't return confidence
                })
        
        return detected_faces
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Returns:
            Similarity score between 0 and 1.
        """
        # Normalize embeddings
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity
        similarity = float(np.dot(e1, e2))
        
        # Convert from [-1, 1] to [0, 1] range
        similarity = (similarity + 1) / 2
        
        return similarity
    
    def search_in_videos(
        self,
        lost_person_path: str,
        video_paths: List[Path],
        search_id: str
    ) -> Dict[str, Any]:
        """
        Search for lost person in video files.
        Returns results in memory - disk writes are optional and non-blocking.
        
        Args:
            lost_person_path: Path to the lost person's photo
            video_paths: List of video file paths to search
            search_id: Unique search identifier
            
        Returns:
            Dictionary containing search results and matches (never depends on disk reads)
        """
        logger.info(f"Starting search {search_id} for {len(video_paths)} videos")
        
        try:
            # Extract lost person's face embedding
            lost_person_image = cv2.imread(lost_person_path)
            if lost_person_image is None:
                logger.error(f"Could not read lost person image: {lost_person_path}")
                return {
                    "search_id": search_id,
                    "status": "error",
                    "error": "Could not read uploaded image",
                    "matches": [],
                    "stats": {"matches_found": 0}
                }
            
            # Detect face in lost person photo
            lost_person_faces = self.detect_faces_in_frame(lost_person_image)
            
            if not lost_person_faces:
                logger.warning("No face detected in lost person photo")
                return {
                    "search_id": search_id,
                    "status": "error",
                    "error": "No face detected in the uploaded photo",
                    "matches": [],
                    "stats": {"matches_found": 0}
                }
            
            # Use the first (largest) face
            lost_person_embedding = self.extract_face_embedding(lost_person_faces[0]['face'])
            logger.info("Lost person face embedding extracted")
            
            # Search results - stored in memory
            matches = []
            search_stats = {
                "total_videos": len(video_paths),
                "videos_processed": 0,
                "total_frames_processed": 0,
                "matches_found": 0
            }
            
            # Process each video
            for video_path in video_paths:
                video_path = Path(video_path)
                if not video_path.exists():
                    logger.warning(f"Video file not found: {video_path}")
                    continue
                
                logger.info(f"Processing video: {video_path.name}")
                
                # Open video
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    logger.error(f"Cannot open video: {video_path}")
                    continue
                
                frame_count = 0
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idx = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames for faster processing
                    if frame_idx % self.frame_skip != 0:
                        frame_idx += 1
                        continue
                    
                    frame_idx += 1
                    frame_count += 1
                    
                    # Detect faces in frame
                    detected_faces = self.detect_faces_in_frame(frame)
                    
                    for face_data in detected_faces:
                        face_embedding = self.extract_face_embedding(face_data['face'])
                        similarity = self.calculate_similarity(lost_person_embedding, face_embedding)
                        
                        # If match found (exceeds threshold)
                        if similarity >= self.similarity_threshold:
                            timestamp = (frame_idx / fps) if fps > 0 else 0
                            
                            # Create match record (in memory)
                            snapshot_name = f"{search_id}_{video_path.stem}_{frame_count}.jpg"
                            
                            match = {
                                "camera": video_path.stem,
                                "camera_name": video_path.stem.replace("_", " ").title(),
                                "confidence": round(similarity * 100, 2),
                                "timestamp": round(timestamp, 2),
                                "frame_number": frame_count,
                                "snapshot": snapshot_name,
                                "time_formatted": self._format_time(timestamp)
                            }
                            
                            matches.append(match)
                            search_stats["matches_found"] += 1
                            
                            logger.info(
                                f"Match found: {match['camera']} at {match['time_formatted']} "
                                f"(confidence: {match['confidence']}%)"
                            )
                            
                            # Try to save snapshot (non-blocking, optional)
                            self._save_snapshot_async(frame, face_data, snapshot_name)
                
                cap.release()
                search_stats["videos_processed"] += 1
                search_stats["total_frames_processed"] += frame_count
                logger.info(f"Completed video: {video_path.name} ({frame_count} frames)")
            
            # Sort matches by confidence (descending)
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Prepare final results - stored in memory
            final_results = {
                "search_id": search_id,
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "matches": matches,
                "stats": search_stats,
                "summary": {
                    "total_matches": len(matches),
                    "best_match_confidence": matches[0]['confidence'] if matches else 0,
                    "cameras_with_matches": len(set(m['camera'] for m in matches)) if matches else 0
                }
            }
            
            # Try to save results to file (non-blocking, optional)
            self._save_results_async(final_results)
            
            logger.info(f"Search {search_id} completed. Found {len(matches)} matches.")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}", exc_info=True)
            return {
                "search_id": search_id,
                "status": "error",
                "error": str(e),
                "matches": [],
                "stats": {"matches_found": 0}
            }
            
            return error_result
    
    def _save_snapshot_async(self, frame: np.ndarray, face_data: Dict[str, Any], snapshot_name: str) -> None:
        """
        Try to save snapshot asynchronously (non-blocking).
        Failures are logged but do not affect search results.
        """
        try:
            snapshot_path = Settings.RESULTS_DIR / snapshot_name
            
            # Draw box on frame
            frame_copy = frame.copy()
            x1, y1, x2, y2 = face_data['bbox']
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Save image
            success = cv2.imwrite(str(snapshot_path), frame_copy)
            if not success:
                logger.warning(f"Failed to save snapshot: {snapshot_path}")
        except Exception as e:
            logger.warning(f"Error saving snapshot {snapshot_name}: {e}")
    
    def _save_results_async(self, results: Dict[str, Any]) -> None:
        """
        Try to save results to JSON file asynchronously (non-blocking).
        Failures are logged but do not affect the API response.
        """
        try:
            results_file = Settings.RESULTS_DIR / f"{results.get('search_id', 'unknown')}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results to {results_file}")
        except Exception as e:
            logger.warning(f"Failed to save results to disk: {e}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def test_rtsp_connection(self, rtsp_url: str) -> Dict[str, Any]:
        """Test if RTSP stream is accessible."""
        try:
            logger.info(f"Testing RTSP connection: {rtsp_url[:50]}...")
            
            cap = cv2.VideoCapture(rtsp_url)
            
            if not cap.isOpened():
                logger.warning("Cannot connect to RTSP stream")
                return {
                    "connected": False,
                    "error": "Cannot open RTSP stream"
                }
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                logger.info("RTSP connection successful")
                return {
                    "connected": True,
                    "message": "Stream is accessible"
                }
            else:
                return {
                    "connected": False,
                    "error": "Cannot read frame from stream"
                }
                
        except Exception as e:
            logger.error(f"RTSP test error: {e}")
            return {
                "connected": False,
                "error": str(e)
            }
    
    def capture_rtsp_footage(
        self,
        rtsp_url: str,
        duration_seconds: int = 15,
        camera_name: str = "CCTV",
        output_dir: str = "CCTVS"
    ) -> str:
        """
        Capture footage from live RTSP stream (max 15 seconds).
        
        Args:
            rtsp_url: RTSP stream URL
            duration_seconds: How long to record (default/max 15 seconds)
            camera_name: Name of camera for logging
            output_dir: Directory to save captured video
            
        Returns:
            Path to saved video file or None if failed
        """
        try:
            # Enforce 15-second maximum
            duration_seconds = min(duration_seconds, 15)
            
            logger.info(f"Starting RTSP capture from {camera_name} for {duration_seconds}s...")
            
            # Open RTSP stream
            cap = cv2.VideoCapture(rtsp_url)
            
            if not cap.isOpened():
                logger.error(f"Failed to connect to RTSP stream: {camera_name}")
                return None
            
            # Get stream properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if width == 0 or height == 0:
                logger.error(f"Invalid stream resolution: {width}x{height}")
                cap.release()
                return None
            
            logger.info(f"Stream {camera_name}: {width}x{height} @ {fps} FPS")
            
            # Create output directory
            output_path_obj = Path(output_dir)
            output_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Create output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in camera_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_file = output_path_obj / f"rtsp_capture_{safe_name}_{timestamp}.mp4"
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_file),
                fourcc,
                fps,
                (width, height)
            )
            
            if not out.isOpened():
                logger.error(f"Failed to open video writer for {output_file}")
                cap.release()
                return None
            
            # Capture frames
            frame_count = 0
            max_frames = fps * duration_seconds
            start_time = datetime.now()
            
            logger.info(f"Capturing {max_frames} frames ({duration_seconds}s at {fps} FPS)")
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Lost connection to {camera_name} after {frame_count} frames")
                    break
                
                out.write(frame)
                frame_count += 1
                
                # Log progress every 5 seconds
                if frame_count % max(1, fps * 5) == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.info(f"{camera_name}: {frame_count}/{max_frames} frames ({elapsed:.1f}s)")
            
            # Release resources
            cap.release()
            out.release()
            
            if frame_count == 0:
                logger.error(f"No frames captured from {camera_name}")
                output_file.unlink(missing_ok=True)
                return None
            
            logger.info(f"Capture complete: {output_file} ({frame_count} frames)")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error capturing from RTSP {camera_name}: {e}", exc_info=True)
            return None
