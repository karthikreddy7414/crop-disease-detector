"""
Image Feature Extraction for crop disease identification.
Extracts visual features from crop images for machine learning models.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from skimage.feature import local_binary_pattern, hog
from skimage.measure import regionprops
from skimage.segmentation import slic
from skimage.color import rgb2hsv, rgb2lab
import logging

logger = logging.getLogger(__name__)

class ImageFeatureExtractor:
    """
    Extracts features from crop images for disease identification.
    """
    
    def __init__(self):
        self.feature_names = []
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2)
        }
    
    def extract_all_features(self, image: np.ndarray) -> Dict:
        """
        Extract all features from crop image.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Color features
        features.update(self._extract_color_features(image))
        
        # Texture features
        features.update(self._extract_texture_features(image))
        
        # Shape features
        features.update(self._extract_shape_features(image))
        
        # HOG features
        features.update(self._extract_hog_features(image))
        
        # LBP features
        features.update(self._extract_lbp_features(image))
        
        # Disease-specific features
        features.update(self._extract_disease_features(image))
        
        return features
    
    def _extract_color_features(self, image: np.ndarray) -> Dict:
        """
        Extract color-based features.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of color features
        """
        features = {}
        
        # RGB color statistics
        r, g, b = cv2.split(image)
        features['r_mean'] = np.mean(r)
        features['g_mean'] = np.mean(g)
        features['b_mean'] = np.mean(b)
        features['r_std'] = np.std(r)
        features['g_std'] = np.std(g)
        features['b_std'] = np.std(b)
        
        # HSV color features
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        features['h_mean'] = np.mean(h)
        features['s_mean'] = np.mean(s)
        features['v_mean'] = np.mean(v)
        features['h_std'] = np.std(h)
        features['s_std'] = np.std(s)
        features['v_std'] = np.std(v)
        
        # LAB color features
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b_lab = cv2.split(lab)
        features['l_mean'] = np.mean(l)
        features['a_mean'] = np.mean(a)
        features['b_lab_mean'] = np.mean(b_lab)
        
        # Color ratios
        features['r_g_ratio'] = np.mean(r) / (np.mean(g) + 1e-8)
        features['g_b_ratio'] = np.mean(g) / (np.mean(b) + 1e-8)
        features['r_b_ratio'] = np.mean(r) / (np.mean(b) + 1e-8)
        
        # Dominant colors
        features.update(self._extract_dominant_colors(image))
        
        return features
    
    def _extract_dominant_colors(self, image: np.ndarray) -> Dict:
        """
        Extract dominant color features.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of dominant color features
        """
        features = {}
        
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3)
        
        # Use K-means to find dominant colors
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get dominant colors and their frequencies
        dominant_colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        for i, color in enumerate(dominant_colors):
            features[f'dominant_color_{i}_r'] = color[0]
            features[f'dominant_color_{i}_g'] = color[1]
            features[f'dominant_color_{i}_b'] = color[2]
            features[f'dominant_color_{i}_freq'] = np.sum(labels == i) / len(labels)
        
        return features
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict:
        """
        Extract texture-based features.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of texture features
        """
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # GLCM features
        features.update(self._extract_glcm_features(gray))
        
        # Gabor filter features
        features.update(self._extract_gabor_features(gray))
        
        # Wavelet features
        features.update(self._extract_wavelet_features(gray))
        
        return features
    
    def _extract_glcm_features(self, gray_image: np.ndarray) -> Dict:
        """
        Extract GLCM (Gray-Level Co-occurrence Matrix) features.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary of GLCM features
        """
        features = {}
        
        # Calculate GLCM
        from skimage.feature import graycomatrix, graycoprops
        
        # Quantize image to reduce number of gray levels
        quantized = (gray_image // 32) * 32
        
        # Calculate GLCM
        glcm = graycomatrix(quantized, distances=[1], angles=[0, 45, 90, 135], levels=8, symmetric=True, normed=True)
        
        # Extract GLCM properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        for prop in properties:
            values = graycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = np.mean(values)
            features[f'glcm_{prop}_std'] = np.std(values)
        
        return features
    
    def _extract_gabor_features(self, gray_image: np.ndarray) -> Dict:
        """
        Extract Gabor filter features.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary of Gabor features
        """
        features = {}
        
        # Gabor filter parameters
        frequencies = [0.1, 0.2, 0.3]
        orientations = [0, 45, 90, 135]
        
        for freq in frequencies:
            for angle in orientations:
                # Create Gabor filter
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
                
                # Extract statistics
                features[f'gabor_freq_{freq}_angle_{angle}_mean'] = np.mean(filtered)
                features[f'gabor_freq_{freq}_angle_{angle}_std'] = np.std(filtered)
                features[f'gabor_freq_{freq}_angle_{angle}_energy'] = np.sum(filtered**2)
        
        return features
    
    def _extract_wavelet_features(self, gray_image: np.ndarray) -> Dict:
        """
        Extract wavelet features.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary of wavelet features
        """
        features = {}
        
        try:
            import pywt
            
            # Apply wavelet transform
            coeffs = pywt.dwt2(gray_image, 'db4')
            cA, (cH, cV, cD) = coeffs
            
            # Extract features from each subband
            for name, coeff in [('LL', cA), ('LH', cH), ('HL', cV), ('HH', cD)]:
                features[f'wavelet_{name}_mean'] = np.mean(coeff)
                features[f'wavelet_{name}_std'] = np.std(coeff)
                features[f'wavelet_{name}_energy'] = np.sum(coeff**2)
                features[f'wavelet_{name}_entropy'] = -np.sum(coeff * np.log(np.abs(coeff) + 1e-8))
        
        except ImportError:
            logger.warning("PyWavelets not available, skipping wavelet features")
        
        return features
    
    def _extract_shape_features(self, image: np.ndarray) -> Dict:
        """
        Extract shape-based features.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of shape features
        """
        features = {}
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Basic shape features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features['area'] = area
            features['perimeter'] = perimeter
            features['perimeter_area_ratio'] = perimeter / (area + 1e-8)
            
            # Bounding rectangle features
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['bounding_rect_area'] = w * h
            features['aspect_ratio'] = w / (h + 1e-8)
            features['extent'] = area / (w * h + 1e-8)
            
            # Convex hull features
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = area / (hull_area + 1e-8)
            
            # Circularity
            features['circularity'] = 4 * np.pi * area / (perimeter**2 + 1e-8)
            
            # Moments
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                features['centroid_x'] = cx
                features['centroid_y'] = cy
        
        return features
    
    def _extract_hog_features(self, image: np.ndarray) -> Dict:
        """
        Extract HOG (Histogram of Oriented Gradients) features.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of HOG features
        """
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate HOG features
        hog_features = hog(
            gray,
            orientations=self.hog_params['orientations'],
            pixels_per_cell=self.hog_params['pixels_per_cell'],
            cells_per_block=self.hog_params['cells_per_block'],
            block_norm='L2-Hys'
        )
        
        # Extract statistics from HOG features
        features['hog_mean'] = np.mean(hog_features)
        features['hog_std'] = np.std(hog_features)
        features['hog_max'] = np.max(hog_features)
        features['hog_min'] = np.min(hog_features)
        
        return features
    
    def _extract_lbp_features(self, image: np.ndarray) -> Dict:
        """
        Extract LBP (Local Binary Pattern) features.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of LBP features
        """
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate LBP
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Extract LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-8)
        
        # Extract statistics
        features['lbp_mean'] = np.mean(lbp)
        features['lbp_std'] = np.std(lbp)
        features['lbp_entropy'] = -np.sum(hist * np.log(hist + 1e-8))
        
        return features
    
    def _extract_disease_features(self, image: np.ndarray) -> Dict:
        """
        Extract disease-specific features.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of disease features
        """
        features = {}
        
        # Convert to HSV for better disease detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Yellow/brown discoloration (common in diseases)
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        brown_mask = cv2.inRange(hsv, np.array([10, 100, 20]), np.array([20, 255, 200]))
        
        total_pixels = image.shape[0] * image.shape[1]
        features['yellow_percentage'] = np.sum(yellow_mask > 0) / total_pixels * 100
        features['brown_percentage'] = np.sum(brown_mask > 0) / total_pixels * 100
        
        # Dark spots detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dark_spots = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
        features['dark_spots_percentage'] = np.sum(dark_spots > 0) / total_pixels * 100
        
        # Edge density (disease often causes irregular edges)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / total_pixels * 100
        
        return features
    
    def extract_region_features(self, image: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """
        Extract features from specific regions of interest.
        
        Args:
            image: Input image array
            regions: List of region dictionaries with 'bbox' key
            
        Returns:
            List of feature dictionaries for each region
        """
        region_features = []
        
        for region in regions:
            x, y, w, h = region['bbox']
            roi = image[y:y+h, x:x+w]
            
            if roi.size > 0:
                features = self.extract_all_features(roi)
                features['region_id'] = region.get('id', 0)
                features['region_area'] = w * h
                region_features.append(features)
        
        return region_features
