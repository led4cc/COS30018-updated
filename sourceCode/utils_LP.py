
"""Utility module for cropping and rotating license plate images in a detection pipeline."""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('license_plate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess(image: np.ndarray, use_adaptive: bool = False, blur: bool = True,
               threshold_type: str = 'otsu', custom_thresh: Optional[int] = None,
               block_size: int = 11, adaptive_constant: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Convert image to grayscale and apply thresholding for license plate preprocessing.

    Args:
        image: Input image (RGB or grayscale).
        use_adaptive: Use adaptive thresholding instead of Otsu's.
        blur: Apply Gaussian blur before thresholding.
        threshold_type: 'otsu' or 'fixed' for thresholding method.
        custom_thresh: Threshold value for 'fixed' mode.
        block_size: Block size for adaptive thresholding.
        adaptive_constant: Constant subtracted from mean in adaptive thresholding.

    Returns:
        Tuple of grayscale and binary thresholded images.
    """
    if image is None or len(image.shape) < 2:
        raise ValueError("Invalid image for preprocessing")

    # Convert to grayscale if not already
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement before thresholding
    gray = cv2.convertScaleAbs(gray, alpha=1., beta=0)
    
    if blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduced kernel size
    
    if use_adaptive:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, adaptive_constant)
        logger.info(f"Using adaptive thresholding (block_size={block_size}, C={adaptive_constant})")
    elif threshold_type == 'otsu':
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info("Using Otsu's thresholding")
    elif threshold_type == 'fixed' and custom_thresh is not None:
        _, thresh = cv2.threshold(gray, custom_thresh, 255, cv2.THRESH_BINARY)
        logger.info(f"Using fixed thresholding with threshold {custom_thresh}")
    else:
        raise ValueError(f"Invalid threshold type: {threshold_type}")
    
    # Morphological operations to separate characters and preserve small features
    thresh = cv2.erode(thresh, np.ones((2, 2), np.uint8), iterations=1)  # Shrink to separate
    thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)  # Restore shape
    
    logger.info(f"Preprocessed image shape: {thresh.shape}")
    return gray, thresh

def Hough_transform(image: np.ndarray, nol: int = 6, threshold: int = 100, 
                   minLineLength: int = 50, maxLineGap: int = 10) -> Optional[np.ndarray]:
    """Detect lines using Probabilistic Hough Transform."""
    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=threshold, 
                           minLineLength=minLineLength, maxLineGap=maxLineGap)
    return lines[:nol] if lines is not None and len(lines) > 0 else None

def rotation_angle(lines: Optional[np.ndarray]) -> float:
    """Calculate average angle of detected lines."""
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
        logger.debug(f"Line: p1=({x1}, {y1}), p2=({x2}, {y2}), angle={angle:.2f}, "
                    f"length={np.sqrt((x2-x1)**2 + (y2-y1)**2):.2f}")
    angle = np.mean(angles) if angles else 0.0
    logger.info(f"Final rotation angle: {angle:.2f}")
    return angle

def rotate_LP(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by specified angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def crop_n_rotate_LP(
    source_img: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    min_ratio: float = 1.0,
    max_ratio: float = 4.73,
    canny_low: int = 250,
    canny_high: int = 255,
    kernel_size: int = 3,
    hough_nol: int = 6,
    hough_threshold: int = 100,
    hough_min_line: int = 50,
    hough_max_gap: int = 10,
    use_adaptive_thresh: bool = False
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Crop and rotate a license plate region from an image.

    Args:
        source_img: Input RGB image (NumPy array).
        x1, y1, x2, y2: Bounding box coordinates of the license plate.
        min_ratio, max_ratio: Valid aspect ratio range for license plates.
        canny_low, canny_high: Canny edge detection thresholds.
        kernel_size: Size of dilation kernel.
        hough_nol: Maximum number of lines to detect.
        hough_threshold: Hough Transform threshold.
        hough_min_line: Minimum line length for Hough Transform.
        hough_max_gap: Maximum gap between line segments.
        use_adaptive_thresh: Use adaptive thresholding in preprocess.

    Returns:
        Tuple containing:
        - Rotation angle (degrees) or None if invalid.
        - Rotated thresholded image or None.
        - Rotated cropped license plate image or None.

    Raises:
        ValueError: If input image or coordinates are invalid.
    """
    try:
        # Validate inputs
        if not isinstance(source_img, np.ndarray) or len(source_img.shape) != 3 or source_img.shape[2] != 3:
            raise ValueError("Source image must be a valid RGB NumPy array")
        
        img_h, img_w = source_img.shape[:2]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if not (0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h):
            raise ValueError(f"Invalid bounding box coordinates: ({x1}, {y1}, {x2}, {y2})")

        # Calculate dimensions and aspect ratio
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            raise ValueError("Invalid bounding box dimensions")
        ratio = w / h
        logger.info(f"Detected license plate with aspect ratio {ratio:.2f}")

        if not (min_ratio <= ratio <= max_ratio):
            logger.warning(f"Invalid aspect ratio: {ratio:.2f} (expected {min_ratio} to {max_ratio})")
            return None, None, None

        # Crop the license plate
        cropped_LP = source_img[y1:y2, x1:x2]
        logger.info(f"Cropped license plate shape: {cropped_LP.shape}")

        # Preprocess
        imgGrayscaleplate, imgThreshplate = preprocess(cropped_LP, use_adaptive=use_adaptive_thresh)
        logger.info(f"Preprocessed license plate shape: {imgThreshplate.shape}")

        # Edge detection and dilation
        canny_image = cv2.Canny(imgThreshplate, canny_low, canny_high)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.dilate(canny_image, kernel, iterations=2)
        logger.info(f"Dilated image shape: {dilated_image.shape}")

        # Line detection
        linesP = Hough_transform(dilated_image, nol=hough_nol, threshold=hough_threshold,
                                minLineLength=hough_min_line, maxLineGap=hough_max_gap)
        if linesP is None:
            logger.warning("No lines detected by Hough Transform")
            return None, None, None

        # Calculate rotation angle
        angle = rotation_angle(linesP)

        # Rotate images
        rotate_thresh = rotate_LP(imgThreshplate, angle)
        LP_rotated = rotate_LP(cropped_LP, angle)
        logger.info(f"Rotated license plate shape: {LP_rotated.shape}")

        return angle, rotate_thresh, LP_rotated

    except Exception as e:
        logger.error(f"Error in crop_n_rotate_LP: {str(e)}")
        return None, None, None