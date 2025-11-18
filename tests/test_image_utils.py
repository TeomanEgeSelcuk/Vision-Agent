"""
Unit tests for src/image_utils.py module.

Tests image loading, grid splitting, encoding, and coordinate calculations.
Uses PIL for image creation and validation.
"""

import pytest
from pathlib import Path
from PIL import Image, ImageDraw
import base64
from io import BytesIO

from src.image_utils import (
    collect_images,
    load_image,
    split_into_grid,
    encode_image_to_data_url,
    get_cell_center,
    GRID_ROWS,
    GRID_COLS,
    SUPPORTED_EXTS,
)


class TestCollectImages:
    """Test suite for collect_images function."""
    
    def test_collect_supported_images(self, tmp_path):
        """
        Test collecting only supported image file types.
        
        What it tests: Image file filtering by extension
        Expected output: List of paths with supported extensions only
        Verifies: Only .png, .jpg, .jpeg, .webp, .bmp, .avif files collected
        """
        # Create test files
        (tmp_path / "image1.png").touch()
        (tmp_path / "image2.jpg").touch()
        (tmp_path / "image3.avif").touch()
        (tmp_path / "document.txt").touch()  # Should be ignored
        (tmp_path / "video.mp4").touch()  # Should be ignored
        
        images = collect_images(tmp_path)
        
        # Verify only image files collected
        assert len(images) == 3
        assert all(img.suffix.lower() in SUPPORTED_EXTS for img in images)
    
    def test_empty_directory(self, tmp_path):
        """
        Test collecting from empty directory.
        
        What it tests: Handling of empty directories
        Expected output: Empty list
        Verifies: No errors with empty directory
        """
        images = collect_images(tmp_path)
        assert images == []
    
    def test_sorted_order(self, tmp_path):
        """
        Test that collected images are sorted alphabetically.
        
        What it tests: Deterministic sorting of results
        Expected output: Alphabetically sorted list
        Verifies: Consistent processing order across runs
        """
        # Create files in non-alphabetical order
        (tmp_path / "zebra.png").touch()
        (tmp_path / "apple.png").touch()
        (tmp_path / "middle.png").touch()
        
        images = collect_images(tmp_path)
        names = [img.name for img in images]
        
        assert names == sorted(names)
        assert names == ["apple.png", "middle.png", "zebra.png"]
    
    def test_case_insensitive_extensions(self, tmp_path):
        """
        Test that file extensions are matched case-insensitively.
        
        What it tests: Case-insensitive extension matching
        Expected output: Files with .PNG, .Jpg, etc. are collected
        Verifies: Mixed-case extensions are recognized
        """
        (tmp_path / "image1.PNG").touch()
        (tmp_path / "image2.Jpg").touch()
        (tmp_path / "image3.AVIF").touch()
        
        images = collect_images(tmp_path)
        assert len(images) == 3
    
    def test_subdirectories_ignored(self, tmp_path):
        """
        Test that subdirectories are not included in results.
        
        What it tests: Directory filtering
        Expected output: Only files, no directories
        Verifies: is_file() check works correctly
        """
        (tmp_path / "image.png").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.png").touch()
        
        images = collect_images(tmp_path)
        
        # Should only find image in root, not subdirectory
        assert len(images) == 1
        assert images[0].name == "image.png"


class TestLoadImage:
    """Test suite for load_image function."""
    
    def test_load_rgb_image(self, tmp_path):
        """
        Test loading an RGB image.
        
        What it tests: Loading and mode conversion for RGB images
        Expected output: PIL Image in RGB mode
        Verifies: RGB images load correctly
        """
        # Create test RGB image
        img_path = tmp_path / "test.png"
        test_img = Image.new("RGB", (100, 100), color="red")
        test_img.save(img_path)
        
        loaded = load_image(img_path)
        
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)
    
    def test_load_rgba_image(self, tmp_path):
        """
        Test loading an RGBA image (with alpha channel).
        
        What it tests: Conversion of RGBA to RGB
        Expected output: PIL Image in RGB mode (alpha removed)
        Verifies: Alpha channel is properly removed
        """
        # Create test RGBA image
        img_path = tmp_path / "test.png"
        test_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        test_img.save(img_path)
        
        loaded = load_image(img_path)
        
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)
    
    def test_load_grayscale_image(self, tmp_path):
        """
        Test loading a grayscale image.
        
        What it tests: Conversion of grayscale to RGB
        Expected output: PIL Image in RGB mode (3 channels)
        Verifies: Grayscale images converted to RGB
        """
        # Create test grayscale image
        img_path = tmp_path / "test.png"
        test_img = Image.new("L", (100, 100), color=128)
        test_img.save(img_path)
        
        loaded = load_image(img_path)
        
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)
    
    def test_load_nonexistent_file(self):
        """
        Test error handling for nonexistent file.
        
        What it tests: FileNotFoundError handling
        Expected output: Exception raised
        Verifies: Proper error on missing file
        """
        with pytest.raises(FileNotFoundError):
            load_image(Path("nonexistent.png"))


class TestSplitIntoGrid:
    """Test suite for split_into_grid function."""
    
    def test_split_square_image(self):
        """
        Test splitting a square image into 3x3 grid.
        
        What it tests: Grid splitting with evenly divisible dimensions
        Expected output: 9 regions with correct coordinates
        Verifies: All cells created with proper row/col indices
        """
        # Create 300x300 test image (evenly divisible by 3)
        img = Image.new("RGB", (300, 300))
        
        regions = split_into_grid(img)
        
        # Should have 9 regions (3x3)
        assert len(regions) == 9
        
        # Verify all row/col combinations present
        coords = {(r, c) for r, c, _ in regions}
        expected_coords = {(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)}
        assert coords == expected_coords
        
        # Verify all crops are 100x100
        for _, _, crop in regions:
            assert crop.size == (100, 100)
    
    def test_split_rectangular_image(self):
        """
        Test splitting a rectangular image into 3x3 grid.
        
        What it tests: Grid splitting with non-square images
        Expected output: 9 regions with different width/height
        Verifies: Each cell has correct dimensions
        """
        # Create 600x300 test image
        img = Image.new("RGB", (600, 300))
        
        regions = split_into_grid(img)
        
        assert len(regions) == 9
        
        # Cell size should be 200x100
        for _, _, crop in regions:
            assert crop.size == (200, 100)
    
    def test_split_with_remainder(self):
        """
        Test splitting image with dimensions not evenly divisible by 3.
        
        What it tests: Remainder pixel handling
        Expected output: Last row/col gets extra pixels
        Verifies: All pixels covered, no gaps or overlaps
        """
        # Create 302x301 test image (not evenly divisible)
        img = Image.new("RGB", (302, 301))
        
        regions = split_into_grid(img)
        
        assert len(regions) == 9
        
        # Check edge cells get remainder pixels
        for r, c, crop in regions:
            w, h = crop.size
            
            # Most cells should be 100x100 (302//3 = 100, 301//3 = 100)
            # Last column (c=2) should be 102 wide (includes remainder)
            # Last row (r=2) should be 101 tall (includes remainder)
            if c == 2:
                assert w == 102
            else:
                assert w == 100
            
            if r == 2:
                assert h == 101
            else:
                assert h == 100
    
    def test_region_order(self):
        """
        Test that regions are returned in row-major order.
        
        What it tests: Region ordering (top-left to bottom-right)
        Expected output: List ordered by row then column
        Verifies: Consistent iteration order
        """
        img = Image.new("RGB", (300, 300))
        regions = split_into_grid(img)
        
        # Extract coordinates
        coords = [(r, c) for r, c, _ in regions]
        
        # Should be in order: (0,0), (0,1), (0,2), (1,0), ...
        expected = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
        ]
        assert coords == expected
    
    def test_small_image(self):
        """
        Test splitting a very small image.
        
        What it tests: Handling of small dimensions
        Expected output: 9 tiny regions
        Verifies: Works with images smaller than grid size
        """
        # Create 9x9 test image (3x3 pixels per cell)
        img = Image.new("RGB", (9, 9))
        
        regions = split_into_grid(img)
        
        assert len(regions) == 9
        
        # Each cell should be 3x3
        for _, _, crop in regions:
            assert crop.size == (3, 3)


class TestEncodeImageToDataUrl:
    """Test suite for encode_image_to_data_url function."""
    
    def test_encode_png(self):
        """
        Test encoding image as PNG data URL.
        
        What it tests: PNG encoding with base64
        Expected output: data URL string with correct MIME type
        Verifies: Proper data URL format with image/png
        """
        # Create test image
        img = Image.new("RGB", (10, 10), color="blue")
        
        data_url = encode_image_to_data_url(img, fmt="PNG")
        
        # Verify format
        assert data_url.startswith("data:image/png;base64,")
        
        # Verify it's valid base64
        b64_data = data_url.split(",")[1]
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0
    
    def test_encode_jpeg(self):
        """
        Test encoding image as JPEG data URL.
        
        What it tests: JPEG encoding with base64
        Expected output: data URL string with image/jpeg MIME type
        Verifies: Proper data URL format with image/jpeg
        """
        img = Image.new("RGB", (10, 10), color="green")
        
        data_url = encode_image_to_data_url(img, fmt="JPEG")
        
        # Verify format
        assert data_url.startswith("data:image/jpeg;base64,")
        
        # Verify it's valid base64
        b64_data = data_url.split(",")[1]
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0
    
    def test_encoded_image_decodable(self):
        """
        Test that encoded image can be decoded back.
        
        What it tests: Round-trip encoding/decoding
        Expected output: Decoded image matches original dimensions
        Verifies: Data integrity through encoding cycle
        """
        # Create test image with specific color
        original = Image.new("RGB", (50, 50), color="red")
        
        # Encode
        data_url = encode_image_to_data_url(original, fmt="PNG")
        
        # Decode
        b64_data = data_url.split(",")[1]
        decoded_bytes = base64.b64decode(b64_data)
        decoded_img = Image.open(BytesIO(decoded_bytes))
        
        # Verify dimensions preserved
        assert decoded_img.size == original.size
        assert decoded_img.mode == "RGB"
    
    def test_different_sizes(self):
        """
        Test encoding images of various sizes.
        
        What it tests: Size handling in encoding
        Expected output: Larger images produce longer data URLs
        Verifies: Encoding scales with image size
        """
        small = Image.new("RGB", (10, 10))
        large = Image.new("RGB", (100, 100))
        
        small_url = encode_image_to_data_url(small)
        large_url = encode_image_to_data_url(large)
        
        # Larger image should have longer encoded string
        assert len(large_url) > len(small_url)


class TestGetCellCenter:
    """Test suite for get_cell_center function."""
    
    def test_center_calculation_square(self):
        """
        Test cell center calculation for square image.
        
        What it tests: Center coordinate calculation
        Expected output: Coordinates at cell midpoints
        Verifies: Centers align with cell boundaries
        """
        # 300x300 image → 100x100 cells
        # Cell (0,0) center should be at (50, 50)
        # Cell (1,1) center should be at (150, 150)
        
        cx, cy = get_cell_center(0, 0, 300, 300)
        assert cx == 50
        assert cy == 50
        
        cx, cy = get_cell_center(1, 1, 300, 300)
        assert cx == 150
        assert cy == 150
        
        cx, cy = get_cell_center(2, 2, 300, 300)
        assert cx == 250
        assert cy == 250
    
    def test_center_calculation_rectangular(self):
        """
        Test cell center calculation for rectangular image.
        
        What it tests: Center calculation with non-square dimensions
        Expected output: Different x and y cell sizes handled correctly
        Verifies: Width and height processed independently
        """
        # 600x300 image → 200x100 cells
        
        cx, cy = get_cell_center(0, 0, 600, 300)
        assert cx == 100  # (0 + 0.5) * 200
        assert cy == 50   # (0 + 0.5) * 100
        
        cx, cy = get_cell_center(1, 2, 600, 300)
        assert cx == 500  # (2 + 0.5) * 200
        assert cy == 150  # (1 + 0.5) * 100
    
    def test_all_cell_centers(self):
        """
        Test center calculation for all 9 grid cells.
        
        What it tests: Complete grid center mapping
        Expected output: 9 unique center coordinates
        Verifies: No duplicate or overlapping centers
        """
        img_w, img_h = 300, 300
        centers = []
        
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                center = get_cell_center(r, c, img_w, img_h)
                centers.append(center)
        
        # Should have 9 unique centers
        assert len(centers) == 9
        assert len(set(centers)) == 9  # All unique
    
    def test_center_within_bounds(self):
        """
        Test that all cell centers are within image bounds.
        
        What it tests: Center coordinates validity
        Expected output: All centers within image dimensions
        Verifies: No out-of-bounds coordinates
        """
        img_w, img_h = 400, 300
        
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cx, cy = get_cell_center(r, c, img_w, img_h)
                
                # Centers should be within image bounds
                assert 0 <= cx < img_w
                assert 0 <= cy < img_h


class TestConstants:
    """Test suite for module constants."""
    
    def test_grid_dimensions(self):
        """
        Test that grid dimensions are set correctly.
        
        What it tests: Grid size constants
        Expected output: 3x3 grid configuration
        Verifies: Standard grid dimensions used
        """
        assert GRID_ROWS == 3
        assert GRID_COLS == 3
    
    def test_supported_extensions(self):
        """
        Test that all expected extensions are supported.
        
        What it tests: Supported file types list
        Expected output: Set containing common image formats
        Verifies: All standard formats included
        """
        expected_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}
        assert SUPPORTED_EXTS == expected_exts
        
        # Verify all are lowercase (for case-insensitive matching)
        assert all(ext.islower() for ext in SUPPORTED_EXTS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
