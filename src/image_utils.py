"""
Image processing utilities for HYGO project.

Handles image loading, grid splitting, and encoding operations.
All operations optimized for minimal memory usage and processing time.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

from PIL import Image


# Constants for grid configuration
GRID_ROWS = 3
GRID_COLS = 3

# Supported image formats - lowercase for fast comparison
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}


def collect_images(directory: Path) -> List[Path]:
    """
    Collect all supported image files from a directory.
    
    Uses set-based extension lookup for O(1) filtering per file.
    Time complexity: O(n) where n is number of files in directory
    Space complexity: O(m) where m is number of matching images
    
    Args:
        directory: Path object pointing to directory to scan
        
    Returns:
        Sorted list of Path objects for supported images
    """
    # List comprehension with generator for memory efficiency
    images = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    ]
    
    # Sort for deterministic processing order - O(m log m)
    return sorted(images)


def load_image(image_path: Path) -> Image.Image:
    """
    Load image and convert to RGB for consistent processing.
    
    Time complexity: O(w*h) where w,h are image dimensions
    Space complexity: O(w*h*3) for RGB pixel data
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image in RGB mode
    """
    # Convert to RGB to ensure 3-channel consistency
    # This handles RGBA, grayscale, palette modes uniformly
    return Image.open(image_path).convert("RGB")


def split_into_grid(img: Image.Image) -> List[Tuple[int, int, Image.Image]]:
    """
    Split image into GRID_ROWS × GRID_COLS equal regions.
    
    Uses integer division for precise cell boundaries.
    Last row/col includes remainder pixels for exact coverage.
    
    Time complexity: O(GRID_ROWS * GRID_COLS * cell_pixels) for cropping
    Space complexity: O(GRID_ROWS * GRID_COLS * cell_pixels) for crops
    
    Args:
        img: PIL Image to split
        
    Returns:
        List of (row, col, cropped_image) tuples
    """
    w, h = img.size
    
    # Calculate cell dimensions - O(1)
    cell_w = w // GRID_COLS
    cell_h = h // GRID_ROWS
    
    # Pre-allocate list for known size - O(1) allocation
    regions = []
    
    # Iterate through grid - O(GRID_ROWS * GRID_COLS)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # Calculate cell boundaries
            x0 = c * cell_w
            y0 = r * cell_h
            
            # Last column/row takes remainder pixels
            x1 = (c + 1) * cell_w if c < GRID_COLS - 1 else w
            y1 = (r + 1) * cell_h if r < GRID_ROWS - 1 else h
            
            # Crop creates a new image - O(cell_w * cell_h)
            crop = img.crop((x0, y0, x1, y1))
            regions.append((r, c, crop))
    
    return regions


def encode_image_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    """
    Encode PIL image as base64 data URL for API transmission.
    
    Uses in-memory buffer to avoid disk I/O.
    PNG format for lossless transmission; can use JPEG for smaller size.
    
    Time complexity: O(w*h) for encoding + O(n) for base64 where n is byte size
    Space complexity: O(n) for encoded bytes
    
    Args:
        img: PIL Image to encode
        fmt: Image format for encoding (PNG, JPEG)
        
    Returns:
        Data URL string (data:image/...;base64,...)
    """
    # In-memory buffer avoids file I/O - O(1) allocation
    buf = BytesIO()
    
    # Encode image to buffer - O(w*h)
    img.save(buf, format=fmt)
    
    # Encode to base64 - O(n) where n is buffer size
    b64_bytes = base64.b64encode(buf.getvalue())
    b64_str = b64_bytes.decode("utf-8")
    
    # Construct data URL with proper MIME type
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"


def get_cell_center(row: int, col: int, img_width: int, img_height: int) -> Tuple[int, int]:
    """
    Calculate pixel coordinates for center of a grid cell.
    
    Time complexity: O(1)
    Space complexity: O(1)
    
    Args:
        row: Cell row index (0 to GRID_ROWS-1)
        col: Cell column index (0 to GRID_COLS-1)
        img_width: Total image width in pixels
        img_height: Total image height in pixels
        
    Returns:
        (center_x, center_y) tuple in pixel coordinates
    """
    cell_w = img_width // GRID_COLS
    cell_h = img_height // GRID_ROWS
    
    # Center is at (col + 0.5) * cell_w, (row + 0.5) * cell_h
    center_x = int((col + 0.5) * cell_w)
    center_y = int((row + 0.5) * cell_h)
    
    return center_x, center_y
