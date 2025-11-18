"""
Visualization and overlay module for HYGO project.

Handles drawing grid lines and severity markers on images.
Optimized for clean visual output with minimal computational overhead.
"""

from typing import Dict, Any, Tuple

from PIL import Image, ImageDraw

from .image_utils import GRID_ROWS, GRID_COLS, get_cell_center
from .analyzer import compute_severity


# Severity color mapping (RGB tuples)
# Yellow for mild issues, red for severe issues
SEVERITY_COLORS = {
    "mild": (255, 215, 0),     # Gold/Yellow
    "severe": (255, 0, 0),      # Red
}

# Grid line color - white for visibility on most images
GRID_LINE_COLOR = (255, 255, 255)

# Marker outline color for better visibility
MARKER_OUTLINE_COLOR = (0, 0, 0)


def calculate_marker_radius(img_width: int, img_height: int) -> int:
    """
    Calculate appropriate marker radius based on image size.
    
    Scales marker size to be visible but not overwhelming.
    Time complexity: O(1)
    Space complexity: O(1)
    
    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Marker radius in pixels
    """
    # Cell dimensions
    cell_w = img_width // GRID_COLS
    cell_h = img_height // GRID_ROWS
    
    # Marker should be roughly 1/15 of cell size, min 3px
    radius = max(3, min(cell_w, cell_h) // 15)
    
    return radius


def calculate_line_width(img_width: int, img_height: int) -> int:
    """
    Calculate appropriate line width for grid based on image size.
    
    Scales line width to be visible but not obtrusive.
    Time complexity: O(1)
    Space complexity: O(1)
    
    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Line width in pixels
    """
    # Line should be roughly 1/300 of smallest dimension, min 1px
    width = max(1, min(img_width, img_height) // 300)
    
    return width


def draw_grid_lines(draw: ImageDraw.ImageDraw, img_width: int, img_height: int) -> None:
    """
    Draw 3x3 grid lines on image.
    
    Draws vertical and horizontal lines to show cell boundaries.
    Time complexity: O(GRID_ROWS + GRID_COLS)
    Space complexity: O(1)
    
    Args:
        draw: PIL ImageDraw object
        img_width: Image width in pixels
        img_height: Image height in pixels
    """
    cell_w = img_width // GRID_COLS
    cell_h = img_height // GRID_ROWS
    
    line_width = calculate_line_width(img_width, img_height)
    
    # Draw vertical lines - O(GRID_COLS)
    for c in range(1, GRID_COLS):
        x = c * cell_w
        draw.line(
            [(x, 0), (x, img_height)],
            fill=GRID_LINE_COLOR,
            width=line_width
        )
    
    # Draw horizontal lines - O(GRID_ROWS)
    for r in range(1, GRID_ROWS):
        y = r * cell_h
        draw.line(
            [(0, y), (img_width, y)],
            fill=GRID_LINE_COLOR,
            width=line_width
        )


def draw_severity_marker(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    center_y: int,
    radius: int,
    severity: str,
) -> None:
    """
    Draw a filled circle marker at specified location.
    
    Uses ellipse with equal width/height for perfect circle.
    Time complexity: O(1)
    Space complexity: O(1)
    
    Args:
        draw: PIL ImageDraw object
        center_x: X coordinate of marker center
        center_y: Y coordinate of marker center
        radius: Marker radius in pixels
        severity: Severity level ("mild" or "severe")
    """
    # Get color for severity level
    color = SEVERITY_COLORS.get(severity)
    if color is None:
        return  # Skip if severity not recognized
    
    # Calculate bounding box for circle
    bbox = [
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    ]
    
    # Draw filled circle with outline for visibility
    draw.ellipse(bbox, fill=color, outline=MARKER_OUTLINE_COLOR)


def create_annotated_image(
    img: Image.Image,
    cell_results: Dict[Tuple[int, int], Dict[str, Any]],
) -> Image.Image:
    """
    Create annotated copy of image with grid and severity markers.
    
    Draws grid lines showing 3x3 cell boundaries and colored dots
    indicating severity level of detected issues per cell.
    
    Time complexity: O(w*h) for image copy + O(GRID_ROWS*GRID_COLS) for markers
    Space complexity: O(w*h*3) for new image
    
    Args:
        img: Original PIL Image
        cell_results: Dict mapping (row, col) to analysis results
        
    Returns:
        New PIL Image with annotations
    """
    # Create copy to preserve original - O(w*h)
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    
    w, h = annotated.size
    
    # Draw grid lines - O(GRID_ROWS + GRID_COLS)
    draw_grid_lines(draw, w, h)
    
    # Calculate marker size once - O(1)
    radius = calculate_marker_radius(w, h)
    
    # Draw severity markers - O(GRID_ROWS * GRID_COLS)
    for (row, col), result in cell_results.items():
        issue_count = result.get("issue_count", 0)
        
        # Skip cells with no issues
        if issue_count <= 0:
            continue
        
        # Compute severity - O(1)
        severity = compute_severity(issue_count)
        
        # Skip if no marker needed
        if severity == "none":
            continue
        
        # Get cell center coordinates - O(1)
        center_x, center_y = get_cell_center(row, col, w, h)
        
        # Draw marker - O(1)
        draw_severity_marker(draw, center_x, center_y, radius, severity)
    
    return annotated


def print_analysis_summary(
    image_name: str,
    cell_results: Dict[Tuple[int, int], Dict[str, Any]],
) -> None:
    """
    Print human-readable summary of analysis results.
    
    Outputs to console showing which cells had issues.
    Time complexity: O(n) where n is number of cells
    Space complexity: O(1)
    
    Args:
        image_name: Name of image file
        cell_results: Dict mapping (row, col) to analysis results
    """
    print(f"\nImage: {image_name}")
    
    # Count total issues across all cells
    total_issues = sum(r.get("issue_count", 0) for r in cell_results.values())
    
    if total_issues == 0:
        print("  No issues detected in any cell.")
        return
    
    # Print details for cells with issues
    for (row, col), result in sorted(cell_results.items()):
        issue_count = result.get("issue_count", 0)
        if issue_count > 0:
            issues = result.get("issues", [])
            severity = compute_severity(issue_count)
            print(f"  Cell ({row},{col}) [{severity}]: "
                  f"issue_count={issue_count}, issues={issues}")
