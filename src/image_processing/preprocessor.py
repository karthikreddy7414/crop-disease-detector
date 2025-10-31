"""
Image Preprocessing for crop disease identification.
Handles image enhancement, noise reduction, and preprocessing for field conditions.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Preprocesses crop images for disease identification.
    Handles various field conditions and image quality issues.
    """
    
    def __init__(self):
        self.target_size = (224, 224)
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
    
    def preprocess_image(self, image_path: str, enhance: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for crop images.
        
        Args:
            image_path: Path to input image
            enhance: Whether to apply enhancement
            
        Returns:
            Preprocessed image array
        """
        # Load image
        image = self._load_image(image_path)
        
        # Apply preprocessing steps
        image = self._resize_image(image)
        try:
            image = self._remove_background(image)
        except Exception:
            # Skip background removal if color conversion fails
            pass
        try:
            image = self._enhance_contrast(image)
        except Exception:
            # Skip contrast enhancement if conversion fails
            pass
        
        if enhance:
            image = self._enhance_image(image)
        
        # Normalize for model input
        image = self._normalize_image(image)
        
        return image
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array
        """
        try:
            # Prefer PIL for robust decoding on Windows
            pil_img = Image.open(image_path).convert('RGB')
            image = np.array(pil_img)
            if image is None or image.size == 0:
                raise ValueError(f"Could not load image from {image_path}")
            return image
        except Exception:
            # Fallback to OpenCV
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image from {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                raise
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio.
        
        Args:
            image: Input image array
            
        Returns:
            Resized image array
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def _remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background using color-based segmentation.
        
        Args:
            image: Input image array
            
        Returns:
            Image with background removed
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define range for green (plant) colors
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green regions
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to image
        result = image.copy()
        result[mask == 0] = [0, 0, 0]  # Set background to black
        
        return result
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: Input image array
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply additional image enhancements.
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image
        """
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(image)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Enhance color
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Convert back to numpy array
        enhanced = np.array(pil_image)
        
        return enhanced
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image for model input.
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image array
        """
        # Convert to float and normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array(self.normalize_mean)
        std = np.array(self.normalize_std)
        
        normalized = (normalized - mean) / std
        
        return normalized
    
    def detect_disease_symptoms(self, image: np.ndarray) -> Dict:
        """
        Detect potential disease symptoms in image.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of detected symptoms
        """
        # Ensure image is uint8 RGB for OpenCV ops (preprocess may output float32 normalized)
        if image.dtype != np.uint8:
            img_min, img_max = image.min(), image.max()
            # If image seems normalized around [-3, +3] due to standardization, denormalize best-effort
            # Otherwise, assume [0,1] and scale up
            if img_max <= 1.5 and img_min >= -0.5:
                safe = np.clip(image, 0.0, 1.0)
                work_image = (safe * 255.0).astype(np.uint8)
            else:
                # Fallback: rescale to 0-255 by min-max
                rng = max(1e-8, float(img_max - img_min))
                work_image = ((image - img_min) * (255.0 / rng)).astype(np.uint8)
        else:
            work_image = image

        symptoms = {
            'spots': self._detect_spots(work_image),
            'discoloration': self._detect_discoloration(work_image),
            'lesions': self._detect_lesions(work_image),
            'wilting': self._detect_wilting(work_image)
        }
        
        return symptoms
    
    def _detect_spots(self, image: np.ndarray) -> List[Dict]:
        """
        Detect spots on leaves (potential disease symptoms).
        
        Args:
            image: Input image array
            
        Returns:
            List of detected spots
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect circles (spots)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        spots = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                spots.append({
                    'center': (x, y),
                    'radius': r,
                    'area': np.pi * r * r
                })
        
        return spots
    
    def _detect_discoloration(self, image: np.ndarray) -> Dict:
        """
        Detect color changes in leaves.
        
        Args:
            image: Input image array
            
        Returns:
            Discoloration analysis
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for different discolorations
        yellow_range = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        brown_range = cv2.inRange(hsv, np.array([10, 100, 20]), np.array([20, 255, 200]))
        
        # Calculate discoloration percentages
        total_pixels = image.shape[0] * image.shape[1]
        yellow_pixels = np.sum(yellow_range > 0)
        brown_pixels = np.sum(brown_range > 0)
        
        return {
            'yellow_percentage': (yellow_pixels / total_pixels) * 100,
            'brown_percentage': (brown_pixels / total_pixels) * 100,
            'total_discoloration': ((yellow_pixels + brown_pixels) / total_pixels) * 100
        }
    
    def _detect_lesions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect lesions on plant tissue.
        
        Args:
            image: Input image array
            
        Returns:
            List of detected lesions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to find dark regions
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lesions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small regions
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                lesions.append({
                    'area': area,
                    'bounding_box': (x, y, w, h),
                    'perimeter': cv2.arcLength(contour, True)
                })
        
        return lesions
    
    def _detect_wilting(self, image: np.ndarray) -> Dict:
        """
        Detect wilting symptoms.
        
        Args:
            image: Input image array
            
        Returns:
            Wilting analysis
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines (potential wilting)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        wilting_score = 0
        if lines is not None:
            # Analyze line patterns for wilting
            vertical_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) > 80:  # Vertical lines indicate wilting
                    vertical_lines += 1
            
            wilting_score = min(vertical_lines / 10, 1.0)  # Normalize to 0-1
        
        return {
            'wilting_score': wilting_score,
            'line_count': len(lines) if lines is not None else 0
        }
    
    def augment_image(self, image: np.ndarray, augmentations: List[str]) -> List[np.ndarray]:
        """
        Apply data augmentation to image.
        
        Args:
            image: Input image array
            augmentations: List of augmentation types to apply
            
        Returns:
            List of augmented images
        """
        augmented_images = [image]
        
        for aug_type in augmentations:
            if aug_type == 'rotation':
                rotated = self._rotate_image(image, 15)
                augmented_images.append(rotated)
            elif aug_type == 'flip':
                flipped = self._flip_image(image)
                augmented_images.append(flipped)
            elif aug_type == 'brightness':
                bright = self._adjust_brightness(image, 1.2)
                augmented_images.append(bright)
            elif aug_type == 'noise':
                noisy = self._add_noise(image)
                augmented_images.append(noisy)
        
        return augmented_images
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated
    
    def _flip_image(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally."""
        return cv2.flip(image, 1)
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return noisy
