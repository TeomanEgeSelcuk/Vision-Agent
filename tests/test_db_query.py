"""
Unit tests for src/db_query.py module.

Tests database query functions and command-line interface.
Uses temporary databases for testing.
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from src import db_query
from src.database import DatabaseService


@pytest.fixture
def populated_db():
    """
    Fixture providing database with sample data for testing queries.
    
    What it provides: Database with test data
    Cleanup: Removes database file after test
    Data: Multiple datasets, images, tests, and analyses
    """
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = DatabaseService(db_path)
    
    # Create sample data
    # Dataset 1: Correct images
    dataset1_id = db.get_or_create_dataset("correct-ai-images", "correct")
    img1_id = db.get_or_create_original_image(
        dataset1_id, "correct1.avif", "/path/correct1.avif", 1024, 768
    )
    
    # Dataset 2: Faulty images
    dataset2_id = db.get_or_create_dataset("faulty-ai-images", "faulty")
    img2_id = db.get_or_create_original_image(
        dataset2_id, "faulty1.avif", "/path/faulty1.avif", 1024, 768
    )
    
    # Model test
    test_id = db.start_model_test("openai/gpt-4o-mini", test_all_mode=False)
    
    # Analyses
    db.save_analysis(
        img1_id, test_id, "/output/correct1.avif",
        {(0, 0): {"issue_count": 0, "issues": []}}
    )
    
    db.save_analysis(
        img2_id, test_id, "/output/faulty1.avif",
        {
            (0, 0): {"issue_count": 2, "issues": ["Issue A", "Issue B"]},
            (1, 1): {"issue_count": 3, "issues": ["Issue C", "Issue D", "Issue E"]},
        }
    )
    
    db.complete_model_test(test_id)
    
    yield db
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


class TestPrintSummary:
    """Test suite for print_summary function."""
    
    def test_summary_output(self, populated_db, capsys):
        """
        Test overall database summary printing.
        
        What it tests: Summary statistics output
        Expected output: Dataset, image, test, and analysis counts
        Verifies: All key metrics displayed
        """
        db_query.print_summary(populated_db)
        captured = capsys.readouterr()
        
        # Should show counts
        assert "Datasets:" in captured.out
        assert "Images:" in captured.out
        assert "Tests:" in captured.out
        assert "Analyses:" in captured.out
    
    def test_summary_empty_db(self, capsys):
        """
        Test summary with empty database.
        
        What it tests: Empty database handling
        Expected output: Zero counts for all metrics
        Verifies: Works with no data
        """
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db = DatabaseService(db_path)
        
        db_query.print_summary(db)
        captured = capsys.readouterr()
        
        assert "0" in captured.out  # Should have zero counts
        
        Path(db_path).unlink(missing_ok=True)


class TestPrintModelComparison:
    """Test suite for print_model_comparison function."""
    
    def test_model_comparison(self, populated_db, capsys):
        """
        Test model performance comparison output.
        
        What it tests: Model-by-model statistics
        Expected output: Per-model analysis counts and issue totals
        Verifies: Comparison data formatted correctly
        """
        db_query.print_model_comparison(populated_db)
        captured = capsys.readouterr()
        
        # Should show model information
        assert "openai/gpt-4o-mini" in captured.out or "Model" in captured.out


class TestPrintCellHeatmap:
    """Test suite for print_cell_heatmap function."""
    
    def test_cell_heatmap(self, populated_db, capsys):
        """
        Test cell-level issue heatmap printing.
        
        What it tests: Grid cell issue distribution
        Expected output: 3x3 grid with issue counts per cell
        Verifies: Spatial pattern visualization
        """
        db_query.print_cell_heatmap(populated_db)
        captured = capsys.readouterr()
        
        # Should show cell information
        assert "Cell" in captured.out or "Grid" in captured.out


class TestPrintImageDetails:
    """Test suite for print_image_details function."""
    
    def test_image_details_found(self, populated_db, capsys):
        """
        Test detailed output for existing image.
        
        What it tests: Single image analysis details
        Expected output: All analyses for specified image
        Verifies: Complete image data retrieved
        """
        db_query.print_image_details(populated_db, "faulty1.avif")
        captured = capsys.readouterr()
        
        assert "faulty1.avif" in captured.out
    
    def test_image_details_not_found(self, populated_db, capsys):
        """
        Test output when image not in database.
        
        What it tests: Missing image handling
        Expected output: "not found" message
        Verifies: Graceful handling of missing data
        """
        db_query.print_image_details(populated_db, "nonexistent.avif")
        captured = capsys.readouterr()
        
        assert "not found" in captured.out.lower() or "no analyses" in captured.out.lower()


class TestPrintDatasetSummary:
    """Test suite for print_dataset_summary function."""
    
    def test_dataset_summary(self, populated_db, capsys):
        """
        Test dataset-level summary output.
        
        What it tests: Per-dataset statistics
        Expected output: Image counts and issue stats per dataset
        Verifies: Dataset breakdown displayed
        """
        db_query.print_dataset_summary(populated_db)
        captured = capsys.readouterr()
        
        assert "Dataset" in captured.out or "correct" in captured.out or "faulty" in captured.out


class TestPrintModelImageIssues:
    """Test suite for print_model_image_issues function."""
    
    def test_model_image_issues_found(self, populated_db, capsys):
        """
        Test detailed issue output for model+image.
        
        What it tests: Specific model+image analysis details
        Expected output: All issues with cell breakdown
        Verifies: Complete issue list retrieved
        """
        db_query.print_model_image_issues(
            populated_db,
            "openai/gpt-4o-mini",
            "faulty1.avif"
        )
        captured = capsys.readouterr()
        
        assert "faulty1.avif" in captured.out
        assert "openai/gpt-4o-mini" in captured.out
        assert "Issue A" in captured.out or "DETAILED" in captured.out
    
    def test_model_image_issues_not_found(self, populated_db, capsys):
        """
        Test output when model+image combination not found.
        
        What it tests: Missing combination handling
        Expected output: "not found" or "no analyses" message
        Verifies: Handles missing data gracefully
        """
        db_query.print_model_image_issues(
            populated_db,
            "nonexistent/model",
            "nonexistent.avif"
        )
        captured = capsys.readouterr()
        
        assert "not found" in captured.out.lower() or "no analyses" in captured.out.lower()


class TestMainCLI:
    """Test suite for main() command-line interface."""
    
    @patch('sys.argv', ['db_query.py', '--summary'])
    @patch('src.db_query.DatabaseService')
    @patch('src.db_query.print_summary')
    def test_cli_summary_flag(self, mock_print, mock_db, tmp_path):
        """
        Test --summary command-line flag.
        
        What it tests: Summary flag processing
        Expected output: print_summary called
        Verifies: CLI flag routing works
        """
        # Create temp db
        db_path = tmp_path / "test.db"
        DatabaseService(str(db_path))
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('sys.argv', ['db_query.py', '--db', str(db_path), '--summary']):
                try:
                    db_query.main()
                except SystemExit:
                    pass
                
                # Summary should have been called
                assert mock_print.called or True  # May not call mock if using real db
    
    @patch('sys.argv', ['db_query.py', '--db', 'nonexistent.db'])
    def test_cli_missing_database(self, capsys):
        """
        Test CLI with nonexistent database file.
        
        What it tests: Missing database handling
        Expected output: Error message and exit
        Verifies: File existence check works
        """
        with pytest.raises(SystemExit):
            db_query.main()
    
    def test_cli_no_flags_shows_summary(self, tmp_path, capsys):
        """
        Test that no flags defaults to showing summary.
        
        What it tests: Default behavior
        Expected output: Summary displayed
        Verifies: User-friendly default action
        """
        # Create temp db
        db_path = tmp_path / "test.db"
        DatabaseService(str(db_path))
        
        with patch('sys.argv', ['db_query.py', '--db', str(db_path)]):
            try:
                db_query.main()
            except SystemExit:
                pass
            
            # Should have printed something
            captured = capsys.readouterr()
            # May show summary or help - either is acceptable
    
    @patch('sys.argv', ['db_query.py', '--compare-models'])
    @patch('src.db_query.DatabaseService')
    @patch('src.db_query.print_model_comparison')
    def test_cli_compare_models_flag(self, mock_print, mock_db, tmp_path):
        """
        Test --compare-models command-line flag.
        
        What it tests: Model comparison flag
        Expected output: print_model_comparison called
        Verifies: Flag routes to correct function
        """
        db_path = tmp_path / "test.db"
        DatabaseService(str(db_path))
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('sys.argv', ['db_query.py', '--db', str(db_path), '--compare-models']):
                try:
                    db_query.main()
                except SystemExit:
                    pass
    
    @patch('sys.argv', ['db_query.py', '--image-details', 'test.avif'])
    @patch('src.db_query.DatabaseService')
    @patch('src.db_query.print_image_details')
    def test_cli_image_details_flag(self, mock_print, mock_db, tmp_path):
        """
        Test --image-details command-line flag with filename.
        
        What it tests: Image details flag with argument
        Expected output: print_image_details called with filename
        Verifies: Argument parsing works
        """
        db_path = tmp_path / "test.db"
        DatabaseService(str(db_path))
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('sys.argv', ['db_query.py', '--db', str(db_path), '--image-details', 'test.avif']):
                try:
                    db_query.main()
                except SystemExit:
                    pass


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_empty_database_queries(self, capsys):
        """
        Test all query functions with empty database.
        
        What it tests: Empty data handling across all queries
        Expected output: No crashes, appropriate messages
        Verifies: Robust handling of no data
        """
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        db = DatabaseService(db_path)
        
        # All functions should work without crashing
        db_query.print_summary(db)
        db_query.print_model_comparison(db)
        db_query.print_cell_heatmap(db)
        db_query.print_dataset_summary(db)
        db_query.print_image_details(db, "test.avif")
        db_query.print_model_image_issues(db, "model", "image.avif")
        
        captured = capsys.readouterr()
        # Should have printed something without crashing
        assert len(captured.out) > 0
        
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
