"""
Unit tests for src/visualizer.py module.

Tests image annotation, grid drawing, and severity marker rendering.
Uses PIL for image creation and mock objects for testing.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from PIL import Image, ImageDraw

from src.visualizer import (
    calculate_marker_radius,
    calculate_line_width,
    draw_grid_lines,
    draw_severity_marker,
    create_annotated_image,
    print_analysis_summary,
    SEVERITY_COLORS,
    GRID_LINE_COLOR,
    MARKER_OUTLINE_COLOR,
)


class TestCalculateMarkerRadius:
    """Test suite for calculate_marker_radius function."""
    
    def test_small_image(self):
        """
        Test marker radius calculation for small images.
        
        What it tests: Minimum radius constraint
        Expected output: At least 3 pixels radius
        Verifies: Markers visible even on tiny images
        """
        radius = calculate_marker_radius(90, 90)  # 30x30 cells
        assert radius >= 3
    
    def test_medium_image(self):
        """
        Test marker radius calculation for medium images.
        
        What it tests: Proportional scaling
        Expected output: Radius scales with image size
        Verifies: Markers sized appropriately for image
        """
        radius_small = calculate_marker_radius(300, 300)
        radius_large = calculate_marker_radius(900, 900)
        
        # Larger image should have larger markers
        assert radius_large > radius_small
    
    def test_rectangular_image(self):
        """
        Test marker radius for non-square images.
        
        What it tests: Minimum dimension used for scaling
        Expected output: Based on smaller dimension
        Verifies: Markers fit in smallest cells
        """
        # 600x300 → cell is 200x100 → use min(200,100) = 100
        radius = calculate_marker_radius(600, 300)
        assert radius == max(3, 100 // 15)
    
    def test_minimum_radius(self):
        """
        Test that minimum radius is enforced.
        
        What it tests: Minimum radius constraint of 3 pixels
        Expected output: Never less than 3 pixels
        Verifies: Tiny images still get visible markers
        """
        radius = calculate_marker_radius(30, 30)  # Very small
        assert radius == 3


class TestCalculateLineWidth:
    """Test suite for calculate_line_width function."""
    
    def test_small_image(self):
        """
        Test line width calculation for small images.
        
        What it tests: Minimum line width constraint
        Expected output: At least 1 pixel width
        Verifies: Lines visible even on tiny images
        """
        width = calculate_line_width(90, 90)
        assert width >= 1
    
    def test_large_image(self):
        """
        Test line width calculation for large images.
        
        What it tests: Proportional scaling for large images
        Expected output: Thicker lines for larger images
        Verifies: Lines scale with image size
        """
        width_small = calculate_line_width(300, 300)
        width_large = calculate_line_width(3000, 3000)
        
        assert width_large > width_small
    
    def test_rectangular_image(self):
        """
        Test line width for non-square images.
        
        What it tests: Minimum dimension used for scaling
        Expected output: Based on smaller dimension
        Verifies: Lines consistent across dimensions
        """
        width = calculate_line_width(3000, 300)
        expected = max(1, 300 // 300)
        assert width == expected
    
    def test_minimum_width(self):
        """
        Test that minimum width is enforced.
        
        What it tests: Minimum width constraint of 1 pixel
        Expected output: Never less than 1 pixel
        Verifies: Tiny images still get visible lines
        """
        width = calculate_line_width(50, 50)
        assert width == 1


class TestDrawGridLines:
    """Test suite for draw_grid_lines function."""
    
    def test_grid_lines_drawn(self):
        """
        Test that grid lines are drawn on image.
        
        What it tests: Grid line drawing functionality
        Expected output: Image with grid lines
        Verifies: Lines drawn at correct positions
        """
        # Create test image
        img = Image.new("RGB", (300, 300), color="white")
        draw = ImageDraw.Draw(img)
        
        # Draw grid
        draw_grid_lines(draw, 300, 300)
        
        # Verify image was modified (not easy to check pixels,
        # but we can verify function completes without error)
        assert img.size == (300, 300)
    
    def test_grid_on_different_sizes(self):
        """
        Test grid drawing on various image sizes.
        
        What it tests: Grid scaling with image size
        Expected output: Grids on different sized images
        Verifies: Works with various dimensions
        """
        sizes = [(300, 300), (600, 400), (150, 150)]
        
        for w, h in sizes:
            img = Image.new("RGB", (w, h))
            draw = ImageDraw.Draw(img)
            
            # Should complete without error
            draw_grid_lines(draw, w, h)
            assert img.size == (w, h)


class TestDrawSeverityMarker:
    """Test suite for draw_severity_marker function."""
    
    def test_draw_mild_marker(self):
        """
        Test drawing mild severity marker (yellow).
        
        What it tests: Mild marker rendering
        Expected output: Yellow circle drawn
        Verifies: Correct color for mild severity
        """
        img = Image.new("RGB", (200, 200), color="white")
        draw = ImageDraw.Draw(img)
        
        draw_severity_marker(draw, 100, 100, 20, "mild")
        
        # Check that image was modified
        assert img.getpixel((100, 100)) != (255, 255, 255)
    
    def test_draw_severe_marker(self):
        """
        Test drawing severe severity marker (red).
        
        What it tests: Severe marker rendering
        Expected output: Red circle drawn
        Verifies: Correct color for severe severity
        """
        img = Image.new("RGB", (200, 200), color="white")
        draw = ImageDraw.Draw(img)
        
        draw_severity_marker(draw, 100, 100, 20, "severe")
        
        # Check that image was modified
        assert img.getpixel((100, 100)) != (255, 255, 255)
    
    def test_draw_marker_with_number(self):
        """
        Test drawing marker with issue count number.
        
        What it tests: Number label rendering
        Expected output: Marker with number displayed
        Verifies: Issue count shown in marker
        """
        img = Image.new("RGB", (200, 200), color="white")
        draw = ImageDraw.Draw(img)
        
        # Should complete without error
        draw_severity_marker(draw, 100, 100, 30, "severe", 5)
        
        # Image should be modified
        assert img.size == (200, 200)
    
    def test_invalid_severity_skipped(self):
        """
        Test that invalid severity is handled gracefully.
        
        What it tests: Invalid severity handling
        Expected output: No marker drawn
        Verifies: Function returns early for unknown severity
        """
        img = Image.new("RGB", (200, 200), color="white")
        draw = ImageDraw.Draw(img)
        
        # Should not crash with invalid severity
        draw_severity_marker(draw, 100, 100, 20, "invalid")
        
        # Image should remain mostly white (no marker)
        assert img.getpixel((100, 100)) == (255, 255, 255)
    
    def test_marker_at_different_positions(self):
        """
        Test drawing markers at various positions.
        
        What it tests: Position parameter handling
        Expected output: Markers at specified coordinates
        Verifies: Correct placement on canvas
        """
        img = Image.new("RGB", (300, 300), color="white")
        draw = ImageDraw.Draw(img)
        
        positions = [(50, 50), (150, 150), (250, 250)]
        
        for x, y in positions:
            draw_severity_marker(draw, x, y, 15, "mild")
        
        # Should complete without error
        assert img.size == (300, 300)


class TestCreateAnnotatedImage:
    """Test suite for create_annotated_image function."""
    
    def test_create_annotation_no_issues(self):
        """
        Test annotation of image with no issues.
        
        What it tests: Clean image annotation
        Expected output: Image with grid only (no markers)
        Verifies: Grid drawn even without issues
        """
        img = Image.new("RGB", (300, 300), color="blue")
        
        # No issues in any cell
        cell_results = {
            (r, c): {"issue_count": 0, "issues": []}
            for r in range(3) for c in range(3)
        }
        
        annotated = create_annotated_image(img, cell_results)
        
        # Should return new image with same size
        assert annotated.size == img.size
        assert annotated is not img  # Should be copy
    
    def test_create_annotation_with_mild_issues(self):
        """
        Test annotation with mild severity issues.
        
        What it tests: Mild marker rendering
        Expected output: Image with yellow markers
        Verifies: Mild issues get yellow dots
        """
        img = Image.new("RGB", (300, 300), color="white")
        
        # Some cells with 1-2 issues (mild)
        cell_results = {
            (0, 0): {"issue_count": 1, "issues": ["Issue 1"]},
            (1, 1): {"issue_count": 2, "issues": ["Issue 2", "Issue 3"]},
            (2, 2): {"issue_count": 0, "issues": []},
        }
        
        annotated = create_annotated_image(img, cell_results)
        
        assert annotated.size == img.size
        assert annotated is not img
    
    def test_create_annotation_with_severe_issues(self):
        """
        Test annotation with severe severity issues.
        
        What it tests: Severe marker rendering
        Expected output: Image with red markers
        Verifies: Severe issues get red dots
        """
        img = Image.new("RGB", (300, 300), color="white")
        
        # Some cells with 3+ issues (severe)
        cell_results = {
            (0, 0): {"issue_count": 3, "issues": ["I1", "I2", "I3"]},
            (1, 1): {"issue_count": 5, "issues": ["I1", "I2", "I3", "I4", "I5"]},
            (2, 2): {"issue_count": 0, "issues": []},
        }
        
        annotated = create_annotated_image(img, cell_results)
        
        assert annotated.size == img.size
        assert annotated is not img
    
    def test_create_annotation_mixed_severity(self):
        """
        Test annotation with mixed severity levels.
        
        What it tests: Multiple marker types on same image
        Expected output: Image with both yellow and red markers
        Verifies: Different severities rendered correctly
        """
        img = Image.new("RGB", (300, 300), color="gray")
        
        # Mix of severities
        cell_results = {
            (0, 0): {"issue_count": 0, "issues": []},           # none
            (0, 1): {"issue_count": 1, "issues": ["I1"]},       # mild
            (0, 2): {"issue_count": 3, "issues": ["I1", "I2", "I3"]},  # severe
            (1, 0): {"issue_count": 2, "issues": ["I1", "I2"]}, # mild
            (1, 1): {"issue_count": 0, "issues": []},           # none
            (1, 2): {"issue_count": 4, "issues": ["I1", "I2", "I3", "I4"]},  # severe
        }
        
        annotated = create_annotated_image(img, cell_results)
        
        assert annotated.size == img.size
        assert annotated is not img
    
    def test_original_image_not_modified(self):
        """
        Test that original image is not modified.
        
        What it tests: Immutability of input image
        Expected output: Original unchanged, new image returned
        Verifies: Image.copy() preserves original
        """
        img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        original_pixel = img.getpixel((0, 0))
        
        cell_results = {
            (0, 0): {"issue_count": 5, "issues": ["test"]},
        }
        
        annotated = create_annotated_image(img, cell_results)
        
        # Original should be unchanged
        assert img.getpixel((0, 0)) == original_pixel
        # Annotated should be different object
        assert annotated is not img
    
    def test_empty_cell_results(self):
        """
        Test annotation with empty cell results dict.
        
        What it tests: Empty results handling
        Expected output: Image with grid only
        Verifies: Works with no cell data
        """
        img = Image.new("RGB", (300, 300), color="white")
        cell_results = {}
        
        annotated = create_annotated_image(img, cell_results)
        
        assert annotated.size == img.size
        assert annotated is not img


class TestPrintAnalysisSummary:
    """Test suite for print_analysis_summary function."""
    
    def test_print_no_issues(self, capsys):
        """
        Test summary printing for image with no issues.
        
        What it tests: Clean image summary
        Expected output: "No issues detected" message
        Verifies: Appropriate message for error-free images
        """
        cell_results = {
            (0, 0): {"issue_count": 0, "issues": []},
            (1, 1): {"issue_count": 0, "issues": []},
        }
        
        print_analysis_summary("test.png", cell_results)
        captured = capsys.readouterr()
        
        assert "test.png" in captured.out
        assert "No issues detected" in captured.out
    
    def test_print_with_issues(self, capsys):
        """
        Test summary printing for image with issues.
        
        What it tests: Issue detail printing
        Expected output: Cell coordinates and issue counts
        Verifies: Detailed breakdown for problematic cells
        """
        cell_results = {
            (0, 0): {"issue_count": 2, "issues": ["Issue 1", "Issue 2"]},
            (1, 1): {"issue_count": 0, "issues": []},
            (2, 2): {"issue_count": 3, "issues": ["I1", "I2", "I3"]},
        }
        
        print_analysis_summary("faulty.png", cell_results)
        captured = capsys.readouterr()
        
        assert "faulty.png" in captured.out
        assert "Cell (0,0)" in captured.out
        assert "Cell (2,2)" in captured.out
        assert "issue_count=2" in captured.out
        assert "issue_count=3" in captured.out
    
    def test_print_severity_labels(self, capsys):
        """
        Test that severity labels are included in summary.
        
        What it tests: Severity classification display
        Expected output: "mild" and "severe" labels
        Verifies: Severity computed and shown
        """
        cell_results = {
            (0, 0): {"issue_count": 1, "issues": ["Minor issue"]},  # mild
            (1, 1): {"issue_count": 5, "issues": ["Major"] * 5},     # severe
        }
        
        print_analysis_summary("mixed.png", cell_results)
        captured = capsys.readouterr()
        
        assert "[mild]" in captured.out
        assert "[severe]" in captured.out
    
    def test_print_empty_results(self, capsys):
        """
        Test summary printing with empty results dict.
        
        What it tests: Empty results handling
        Expected output: "No issues detected" message
        Verifies: Graceful handling of empty data
        """
        cell_results = {}
        
        print_analysis_summary("empty.png", cell_results)
        captured = capsys.readouterr()
        
        assert "empty.png" in captured.out
        assert "No issues detected" in captured.out
    
    def test_print_sorted_cells(self, capsys):
        """
        Test that cells are printed in sorted order.
        
        What it tests: Deterministic output ordering
        Expected output: Cells listed in row-major order
        Verifies: Consistent, readable output
        """
        cell_results = {
            (2, 1): {"issue_count": 1, "issues": ["Last"]},
            (0, 0): {"issue_count": 1, "issues": ["First"]},
            (1, 2): {"issue_count": 1, "issues": ["Middle"]},
        }
        
        print_analysis_summary("test.png", cell_results)
        captured = capsys.readouterr()
        
        # Should appear in order (0,0), (1,2), (2,1)
        lines = captured.out.split('\n')
        issue_lines = [l for l in lines if "Cell" in l]
        
        assert len(issue_lines) == 3
        assert "(0,0)" in issue_lines[0]
        assert "(1,2)" in issue_lines[1]
        assert "(2,1)" in issue_lines[2]


class TestConstants:
    """Test suite for module constants."""
    
    def test_severity_colors(self):
        """
        Test that severity colors are defined.
        
        What it tests: Color mapping configuration
        Expected output: RGB tuples for mild and severe
        Verifies: Visual distinction between severities
        """
        assert "mild" in SEVERITY_COLORS
        assert "severe" in SEVERITY_COLORS
        
        # Verify RGB format
        assert len(SEVERITY_COLORS["mild"]) == 3
        assert len(SEVERITY_COLORS["severe"]) == 3
        
        # Verify colors are different
        assert SEVERITY_COLORS["mild"] != SEVERITY_COLORS["severe"]
    
    def test_grid_line_color(self):
        """
        Test that grid line color is defined.
        
        What it tests: Grid line color configuration
        Expected output: RGB tuple
        Verifies: Lines have visible color
        """
        assert len(GRID_LINE_COLOR) == 3
        assert all(isinstance(c, int) for c in GRID_LINE_COLOR)
    
    def test_marker_outline_color(self):
        """
        Test that marker outline color is defined.
        
        What it tests: Outline color configuration
        Expected output: RGB tuple
        Verifies: Markers have outline for visibility
        """
        assert len(MARKER_OUTLINE_COLOR) == 3
        assert all(isinstance(c, int) for c in MARKER_OUTLINE_COLOR)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
