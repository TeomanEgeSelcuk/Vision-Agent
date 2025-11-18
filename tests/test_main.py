"""
Unit tests for src/main.py module.

Tests the main pipeline orchestration and integration.
Uses mocking extensively to avoid API calls and file I/O.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from PIL import Image

from src import main


class TestProcessSingleImage:
    """Test suite for process_single_image function."""
    
    @patch('src.main.load_image')
    @patch('src.main.split_into_grid')
    @patch('src.main.analyze_image_grid')
    @patch('src.main.create_annotated_image')
    @patch('src.main.print_analysis_summary')
    def test_complete_pipeline(
        self, mock_summary, mock_annotate, mock_analyze, mock_split, mock_load
    ):
        """
        Test complete single image processing pipeline.
        
        What it tests: End-to-end image processing
        Expected output: All pipeline steps executed
        Verifies: Workflow coordination and data flow
        """
        # Setup mocks
        mock_img = Image.new("RGB", (300, 300))
        mock_load.return_value = mock_img
        
        mock_regions = [(0, 0, mock_img), (0, 1, mock_img), (0, 2, mock_img)]
        mock_split.return_value = mock_regions
        
        mock_results = {
            (0, 0): {"issue_count": 1, "issues": ["Test"]},
        }
        mock_analyze.return_value = mock_results
        
        mock_annotated = Image.new("RGB", (300, 300))
        mock_annotate.return_value = mock_annotated
        
        # Create temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Run pipeline
            mock_client = Mock()
            main.process_single_image(
                mock_client,
                "openai/gpt-4o-mini",
                Path("test.png"),
                output_dir
            )
            
            # Verify all steps called
            mock_load.assert_called_once()
            mock_split.assert_called_once_with(mock_img)
            mock_analyze.assert_called_once()
            mock_annotate.assert_called_once()
            mock_summary.assert_called_once()
    
    @patch('src.main.load_image')
    @patch('src.main.split_into_grid')
    @patch('src.main.analyze_image_grid')
    @patch('src.main.create_annotated_image')
    def test_output_directory_creation(
        self, mock_annotate, mock_analyze, mock_split, mock_load
    ):
        """
        Test that output directory is created if missing.
        
        What it tests: Directory creation
        Expected output: Output directory exists after processing
        Verifies: mkdir(parents=True, exist_ok=True) called
        """
        mock_img = Image.new("RGB", (300, 300))
        mock_load.return_value = mock_img
        mock_split.return_value = [(0, 0, mock_img)]
        mock_analyze.return_value = {(0, 0): {"issue_count": 0, "issues": []}}
        mock_annotate.return_value = mock_img
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "output"
            
            assert not output_dir.exists()
            
            mock_client = Mock()
            main.process_single_image(
                mock_client,
                "openai/gpt-4o-mini",
                Path("test.png"),
                output_dir
            )
            
            assert output_dir.exists()
    
    @patch('src.main.load_image')
    @patch('src.main.split_into_grid')
    @patch('src.main.analyze_image_grid')
    @patch('src.main.create_annotated_image')
    def test_database_recording(
        self, mock_annotate, mock_analyze, mock_split, mock_load
    ):
        """
        Test that results are saved to database when service provided.
        
        What it tests: Database integration
        Expected output: save_analysis called with correct data
        Verifies: Results persisted to database
        """
        mock_img = Image.new("RGB", (300, 300))
        mock_load.return_value = mock_img
        mock_split.return_value = [(0, 0, mock_img)]
        
        cell_results = {(0, 0): {"issue_count": 2, "issues": ["I1", "I2"]}}
        mock_analyze.return_value = cell_results
        mock_annotate.return_value = mock_img
        
        # Mock database service
        mock_db = Mock()
        mock_db.save_analysis = Mock()
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            mock_client = Mock()
            main.process_single_image(
                mock_client,
                "openai/gpt-4o-mini",
                Path("test.png"),
                output_dir,
                db_service=mock_db,
                dataset_id=1,
                model_test_id=2
            )
            
            # Verify database call
            mock_db.save_analysis.assert_called_once()
            call_args = mock_db.save_analysis.call_args
            assert call_args[0][0] == 1  # dataset_id
            assert call_args[0][1] == 2  # model_test_id
            assert cell_results in call_args[0]


class TestProcessImageBatch:
    """Test suite for process_image_batch function."""
    
    @patch('src.main.process_single_image')
    def test_batch_processing(self, mock_process):
        """
        Test processing multiple images in batch.
        
        What it tests: Batch iteration
        Expected output: Each image processed once
        Verifies: All images in batch handled
        """
        image_paths = [Path("img1.png"), Path("img2.png"), Path("img3.png")]
        
        mock_client = Mock()
        output_dir = Path("/output")
        
        main.process_image_batch(
            mock_client,
            "openai/gpt-4o-mini",
            image_paths,
            output_dir,
            "Testing"
        )
        
        # Verify each image processed
        assert mock_process.call_count == 3
        
        # Verify correct paths passed
        processed_paths = [call[0][2] for call in mock_process.call_args_list]
        assert set(processed_paths) == set(image_paths)
    
    @patch('src.main.process_single_image')
    def test_empty_batch(self, mock_process):
        """
        Test processing empty batch.
        
        What it tests: Empty list handling
        Expected output: No processing occurs
        Verifies: Handles empty input gracefully
        """
        image_paths = []
        
        mock_client = Mock()
        main.process_image_batch(
            mock_client,
            "openai/gpt-4o-mini",
            image_paths,
            Path("/output"),
            "Empty"
        )
        
        # Should not process anything
        mock_process.assert_not_called()
    
    @patch('src.main.process_single_image')
    def test_database_propagation(self, mock_process):
        """
        Test that database parameters are passed to each image.
        
        What it tests: Parameter propagation in batch
        Expected output: Database IDs passed to all calls
        Verifies: Consistent database tracking
        """
        image_paths = [Path("img1.png"), Path("img2.png")]
        
        mock_client = Mock()
        mock_db = Mock()
        
        main.process_image_batch(
            mock_client,
            "openai/gpt-4o-mini",
            image_paths,
            Path("/output"),
            "Testing",
            db_service=mock_db,
            dataset_id=5,
            model_test_id=10
        )
        
        # Verify database params passed to all calls
        for call_obj in mock_process.call_args_list:
            kwargs = call_obj[1]
            assert kwargs.get("db_service") == mock_db
            assert kwargs.get("dataset_id") == 5
            assert kwargs.get("model_test_id") == 10


class TestMainFunction:
    """Test suite for main() entry point."""
    
    @patch('src.main.load_config')
    @patch('src.main.DatabaseService')
    @patch('src.main.validate_openrouter_api_key')
    @patch('src.main.create_openrouter_client')
    @patch('src.main.collect_images')
    @patch('src.main.process_image_batch')
    def test_main_successful_run(
        self, mock_batch, mock_collect, mock_client, mock_validate, mock_db, mock_config
    ):
        """
        Test successful main execution.
        
        What it tests: Complete main workflow
        Expected output: All stages execute without error
        Verifies: Integration of all components
        """
        # Setup mocks
        mock_config.return_value = {
            "api_key": "sk-or-test",
            "model": "openai/gpt-4o-mini",
            "referer": "http://test.com",
            "app_title": "test",
            "test_all": False,
            "db_path": "test.db"
        }
        
        mock_db_instance = Mock()
        mock_db_instance.get_or_create_dataset = Mock(return_value=1)
        mock_db.return_value = mock_db_instance
        
        mock_validate.return_value = True
        mock_client.return_value = Mock()
        
        # Mock images found
        mock_collect.side_effect = [
            [Path("correct1.png")],  # correct images
            [Path("faulty1.png")]     # faulty images
        ]
        
        # Run main
        main.main()
        
        # Verify key steps
        mock_config.assert_called_once()
        mock_validate.assert_called_once()
        mock_client.assert_called_once()
        assert mock_batch.call_count == 2  # Once for each dataset
    
    @patch('src.main.load_config')
    def test_main_config_error(self, mock_config):
        """
        Test main handling of configuration errors.
        
        What it tests: Config error handling
        Expected output: Error message and early exit
        Verifies: Graceful failure on bad config
        """
        mock_config.side_effect = RuntimeError("Config error")
        
        # Should exit with error message (would normally exit, but we catch)
        with pytest.raises(RuntimeError):
            main.main()
    
    @patch('src.main.load_config')
    @patch('src.main.DatabaseService')
    @patch('src.main.validate_openrouter_api_key')
    def test_main_invalid_api_key(self, mock_validate, mock_db, mock_config):
        """
        Test main handling of invalid API key.
        
        What it tests: API key validation failure
        Expected output: Early exit with error
        Verifies: Validation prevents execution
        """
        mock_config.return_value = {
            "api_key": "invalid",
            "model": "openai/gpt-4o-mini",
            "referer": "http://test.com",
            "app_title": "test",
            "test_all": False,
            "db_path": "test.db"
        }
        
        mock_db.return_value = Mock()
        mock_validate.return_value = False
        
        # Should exit early
        with pytest.raises(SystemExit):
            main.main()
    
    @patch('src.main.load_config')
    @patch('src.main.DatabaseService')
    @patch('src.main.validate_openrouter_api_key')
    @patch('src.main.create_openrouter_client')
    @patch('src.main.collect_images')
    def test_main_no_images_found(
        self, mock_collect, mock_client, mock_validate, mock_db, mock_config
    ):
        """
        Test main when no images are found.
        
        What it tests: Empty dataset handling
        Expected output: Error message and exit
        Verifies: Handles missing input gracefully
        """
        mock_config.return_value = {
            "api_key": "sk-or-test",
            "model": "openai/gpt-4o-mini",
            "referer": "http://test.com",
            "app_title": "test",
            "test_all": False,
            "db_path": "test.db"
        }
        
        mock_db.return_value = Mock()
        mock_validate.return_value = True
        mock_client.return_value = Mock()
        
        # No images found
        mock_collect.return_value = []
        
        # Should exit with error
        with pytest.raises(SystemExit):
            main.main()


class TestConstants:
    """Test suite for module constants."""
    
    def test_dataset_paths_defined(self):
        """
        Test that dataset directory constants are defined.
        
        What it tests: Path constants configuration
        Expected output: Path objects for dataset/output
        Verifies: Directory structure defined
        """
        assert hasattr(main, 'DATASET_ROOT')
        assert hasattr(main, 'OUTPUT_ROOT')
        assert isinstance(main.DATASET_ROOT, Path)
        assert isinstance(main.OUTPUT_ROOT, Path)
    
    def test_subdirectory_names(self):
        """
        Test that subdirectory names are defined.
        
        What it tests: Subdirectory constants
        Expected output: String names for correct/faulty
        Verifies: Dataset categories defined
        """
        assert hasattr(main, 'CORRECT_SUBDIR')
        assert hasattr(main, 'FAULTY_SUBDIR')
        assert isinstance(main.CORRECT_SUBDIR, str)
        assert isinstance(main.FAULTY_SUBDIR, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
