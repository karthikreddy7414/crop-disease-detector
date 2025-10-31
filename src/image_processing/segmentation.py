"""
Crop Segmentation for disease identification.
Segments plant regions from background for focused disease analysis.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from skimage.segmentation import slic, watershed
try:
    from skimage.feature import peak_local_maxima
except ImportError:
    from scipy.ndimage import maximum_filter
    def peak_local_maxima(image, min_distance=1, threshold_abs=0):
        """Fallback implementation of peak_local_maxima."""
        # Simple implementation using maximum filter
        local_maxima = maximum_filter(image, size=min_distance*2+1) == image
        local_maxima = local_maxima & (image > threshold_abs)
        return np.where(local_maxima)
from skimage.morphology import disk
from scipy import ndimage

logger = logging.getLogger(__name__)

class CropSegmentation:
    """
    Segments crop regions from images for disease analysis.
    """
    
    def __init__(self):
        self.min_leaf_area = 1000  # Minimum area for leaf detection
        self.max_leaf_area = 50000  # Maximum area for leaf detection
        
    def segment_plant_regions(self, image: np.ndarray, method: str = 'color') -> Dict:
        """
        Segment plant regions from image.
        
        Args:
            image: Input image array
            method: Segmentation method ('color', 'watershed', 'slic', 'combined')
            
        Returns:
            Dictionary with segmentation results
        """
        if method == 'color':
            return self._color_based_segmentation(image)
        elif method == 'watershed':
            return self._watershed_segmentation(image)
        elif method == 'slic':
            return self._slic_segmentation(image)
        elif method == 'combined':
            return self._combined_segmentation(image)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def _color_based_segmentation(self, image: np.ndarray) -> Dict:
        """
        Color-based plant segmentation using HSV color space.
        
        Args:
            image: Input image array
            
        Returns:
            Segmentation results dictionary
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define green color range for plants
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        plant_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_leaf_area < area < self.max_leaf_area:
                x, y, w, h = cv2.boundingRect(contour)
                plant_regions.append({
                    'contour': contour,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'mask': np.zeros_like(mask)
                })
        
        # Create individual masks for each region
        for i, region in enumerate(plant_regions):
            cv2.fillPoly(region['mask'], [region['contour']], 255)
        
        return {
            'method': 'color_based',
            'total_regions': len(plant_regions),
            'regions': plant_regions,
            'global_mask': mask
        }
    
    def _watershed_segmentation(self, image: np.ndarray) -> Dict:
        """
        Watershed-based segmentation for plant regions.
        
        Args:
            image: Input image array
            
        Returns:
            Segmentation results dictionary
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Distance transform
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        
        # Find local maxima
        try:
            local_maxima = peak_local_maxima(dist_transform, min_distance=20, threshold_abs=0.3)
        except:
            # Fallback if peak_local_maxima is not available
            local_maxima = np.array([])
        
        # Create markers
        markers = np.zeros_like(gray, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Apply watershed
        labels = watershed(-dist_transform, markers, mask=thresh)
        
        # Extract regions
        plant_regions = []
        for label in np.unique(labels):
            if label == 0:  # Background
                continue
                
            # Create mask for this label
            mask = (labels == label).astype(np.uint8) * 255
            
            # Find contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = contours[0]
                area = cv2.contourArea(contour)
                
                if self.min_leaf_area < area < self.max_leaf_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    plant_regions.append({
                        'contour': contour,
                        'area': area,
                        'bbox': (x, y, w, h),
                        'mask': mask,
                        'label': label
                    })
        
        return {
            'method': 'watershed',
            'total_regions': len(plant_regions),
            'regions': plant_regions,
            'labels': labels
        }
    
    def _slic_segmentation(self, image: np.ndarray) -> Dict:
        """
        SLIC (Simple Linear Iterative Clustering) segmentation.
        
        Args:
            image: Input image array
            
        Returns:
            Segmentation results dictionary
        """
        # Apply SLIC segmentation
        segments = slic(image, n_segments=300, compactness=10, sigma=1)
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Analyze each segment
        plant_regions = []
        for segment_id in np.unique(segments):
            # Create mask for this segment
            mask = (segments == segment_id).astype(np.uint8) * 255
            
            # Calculate segment properties
            area = np.sum(mask > 0)
            mean_intensity = np.mean(gray[mask > 0])
            
            # Filter based on area and intensity (plants are typically darker)
            if self.min_leaf_area < area < self.max_leaf_area and mean_intensity < 150:
                # Find contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    contour = contours[0]
                    x, y, w, h = cv2.boundingRect(contour)
                    plant_regions.append({
                        'contour': contour,
                        'area': area,
                        'bbox': (x, y, w, h),
                        'mask': mask,
                        'segment_id': segment_id,
                        'mean_intensity': mean_intensity
                    })
        
        return {
            'method': 'slic',
            'total_regions': len(plant_regions),
            'regions': plant_regions,
            'segments': segments
        }
    
    def _combined_segmentation(self, image: np.ndarray) -> Dict:
        """
        Combined segmentation using multiple methods.
        
        Args:
            image: Input image array
            
        Returns:
            Segmentation results dictionary
        """
        # Get results from different methods
        color_results = self._color_based_segmentation(image)
        watershed_results = self._watershed_segmentation(image)
        
        # Combine regions from different methods
        all_regions = []
        all_regions.extend(color_results['regions'])
        all_regions.extend(watershed_results['regions'])
        
        # Remove duplicate regions (overlapping)
        filtered_regions = self._filter_overlapping_regions(all_regions)
        
        return {
            'method': 'combined',
            'total_regions': len(filtered_regions),
            'regions': filtered_regions,
            'color_regions': len(color_results['regions']),
            'watershed_regions': len(watershed_results['regions'])
        }
    
    def _filter_overlapping_regions(self, regions: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
        """
        Filter overlapping regions.
        
        Args:
            regions: List of region dictionaries
            overlap_threshold: Threshold for considering regions as overlapping
            
        Returns:
            Filtered list of regions
        """
        if not regions:
            return []
        
        # Sort regions by area (largest first)
        sorted_regions = sorted(regions, key=lambda x: x['area'], reverse=True)
        
        filtered_regions = []
        
        for region in sorted_regions:
            is_overlapping = False
            
            for existing_region in filtered_regions:
                overlap = self._calculate_overlap(region, existing_region)
                if overlap > overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_regions.append(region)
        
        return filtered_regions
    
    def _calculate_overlap(self, region1: Dict, region2: Dict) -> float:
        """
        Calculate overlap between two regions.
        
        Args:
            region1: First region
            region2: Second region
            
        Returns:
            Overlap ratio
        """
        x1, y1, w1, h1 = region1['bbox']
        x2, y2, w2, h2 = region2['bbox']
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = region1['area'] + region2['area'] - intersection_area
        
        return intersection_area / (union_area + 1e-8)
    
    def extract_leaf_regions(self, image: np.ndarray, regions: List[Dict]) -> List[np.ndarray]:
        """
        Extract individual leaf regions from image.
        
        Args:
            image: Input image
            regions: List of region dictionaries
            
        Returns:
            List of extracted leaf images
        """
        leaf_images = []
        
        for region in regions:
            x, y, w, h = region['bbox']
            
            # Extract region
            leaf_image = image[y:y+h, x:x+w]
            
            # Apply mask if available
            if 'mask' in region:
                mask = region['mask'][y:y+h, x:x+w]
                leaf_image = cv2.bitwise_and(leaf_image, leaf_image, mask=mask)
            
            leaf_images.append(leaf_image)
        
        return leaf_images
    
    def analyze_leaf_health(self, leaf_image: np.ndarray) -> Dict:
        """
        Analyze health of individual leaf.
        
        Args:
            leaf_image: Individual leaf image
            
        Returns:
            Health analysis dictionary
        """
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(leaf_image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for different conditions
        healthy_green = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255]))
        yellow_discoloration = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        brown_spots = cv2.inRange(hsv, np.array([10, 100, 20]), np.array([20, 255, 200]))
        
        # Calculate percentages
        total_pixels = leaf_image.shape[0] * leaf_image.shape[1]
        healthy_percentage = np.sum(healthy_green > 0) / total_pixels * 100
        yellow_percentage = np.sum(yellow_discoloration > 0) / total_pixels * 100
        brown_percentage = np.sum(brown_spots > 0) / total_pixels * 100
        
        # Calculate health score
        health_score = healthy_percentage - (yellow_percentage + brown_percentage) * 0.5
        health_score = max(0, min(100, health_score))
        
        return {
            'health_score': health_score,
            'healthy_percentage': healthy_percentage,
            'yellow_percentage': yellow_percentage,
            'brown_percentage': brown_percentage,
            'total_pixels': total_pixels
        }
    
    def segment_disease_regions(self, image: np.ndarray, plant_regions: List[Dict]) -> List[Dict]:
        """
        Segment disease-affected regions within plant areas.
        
        Args:
            image: Input image
            plant_regions: List of plant region dictionaries
            
        Returns:
            List of disease region dictionaries
        """
        disease_regions = []
        
        for plant_region in plant_regions:
            x, y, w, h = plant_region['bbox']
            plant_roi = image[y:y+h, x:x+w]
            
            # Convert to HSV for disease detection
            hsv = cv2.cvtColor(plant_roi, cv2.COLOR_RGB2HSV)
            
            # Detect disease symptoms
            yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
            brown_mask = cv2.inRange(hsv, np.array([10, 100, 20]), np.array([20, 255, 200]))
            dark_spots = cv2.threshold(cv2.cvtColor(plant_roi, cv2.COLOR_RGB2GRAY), 50, 255, cv2.THRESH_BINARY_INV)[1]
            
            # Combine disease masks
            disease_mask = cv2.bitwise_or(yellow_mask, brown_mask)
            disease_mask = cv2.bitwise_or(disease_mask, dark_spots)
            
            # Find disease contours
            contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Minimum disease spot area
                    # Get contour in original image coordinates
                    contour_global = contour + np.array([x, y])
                    
                    disease_regions.append({
                        'contour': contour_global,
                        'area': area,
                        'bbox': cv2.boundingRect(contour_global),
                        'plant_region_id': plant_region.get('id', 0),
                        'disease_type': self._classify_disease_type(plant_roi, contour),
                        'severity': self._calculate_severity(area, plant_region['area'])
                    })
        
        return disease_regions
    
    def _classify_disease_type(self, plant_roi: np.ndarray, contour: np.ndarray) -> str:
        """
        Classify disease type based on visual characteristics.
        
        Args:
            plant_roi: Plant region of interest
            contour: Disease contour
            
        Returns:
            Disease type classification
        """
        # Extract region around contour
        x, y, w, h = cv2.boundingRect(contour)
        disease_roi = plant_roi[y:y+h, x:x+w]
        
        if disease_roi.size == 0:
            return 'unknown'
        
        # Convert to HSV
        hsv = cv2.cvtColor(disease_roi, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Analyze color characteristics
        mean_h = np.mean(h)
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        
        # Classify based on color
        if mean_h < 20 and mean_s > 100:
            return 'brown_spot'
        elif 20 <= mean_h <= 30 and mean_s > 100:
            return 'yellow_spot'
        elif mean_v < 100:
            return 'dark_spot'
        else:
            return 'discoloration'
    
    def _calculate_severity(self, disease_area: float, plant_area: float) -> str:
        """
        Calculate disease severity.
        
        Args:
            disease_area: Area of disease region
            plant_area: Total plant area
            
        Returns:
            Severity level
        """
        severity_ratio = disease_area / plant_area
        
        if severity_ratio < 0.05:
            return 'low'
        elif severity_ratio < 0.15:
            return 'moderate'
        else:
            return 'high'
    
    def visualize_segmentation(self, image: np.ndarray, regions: List[Dict], 
                              disease_regions: List[Dict] = None) -> np.ndarray:
        """
        Visualize segmentation results.
        
        Args:
            image: Original image
            regions: Plant regions
            disease_regions: Disease regions (optional)
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Draw plant regions in green
        for region in regions:
            cv2.drawContours(vis_image, [region['contour']], -1, (0, 255, 0), 2)
            x, y, w, h = region['bbox']
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        # Draw disease regions in red
        if disease_regions:
            for disease_region in disease_regions:
                cv2.drawContours(vis_image, [disease_region['contour']], -1, (255, 0, 0), 2)
                x, y, w, h = disease_region['bbox']
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
        
        return vis_image
