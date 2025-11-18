"""
Unit tests for src/database.py module.

Tests database models, service operations, and data persistence.
Uses temporary databases for test isolation.
"""

import pytest
import json
from datetime import datetime
from pathlib import Path
import tempfile

from src.database import (
    DatabaseService,
    Dataset,
    OriginalImage,
    ModelTest,
    ImageAnalysis,
    CellResult,
    Base,
)


@pytest.fixture
def temp_db():
    """
    Fixture providing temporary database for testing.
    
    What it provides: Clean database instance per test
    Cleanup: Removes database file after test
    Ensures: Test isolation and no state leakage
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Create database service
    db = DatabaseService(db_path)
    
    yield db
    
    # Cleanup - close all connections first
    db.engine.dispose()
    import time
    time.sleep(0.1)  # Brief delay for Windows file lock release
    try:
        Path(db_path).unlink(missing_ok=True)
    except PermissionError:
        pass  # File still locked, will be cleaned by temp directory


class TestDatabaseService:
    """Test suite for DatabaseService class."""
    
    def test_database_initialization(self, temp_db):
        """
        Test database creation and table schema.
        
        What it tests: Database initialization
        Expected output: Database file created with all tables
        Verifies: Schema properly set up
        """
        # Database should exist
        session = temp_db.Session()
        
        # All tables should exist (no errors on query)
        assert session.query(Dataset).count() == 0
        assert session.query(OriginalImage).count() == 0
        assert session.query(ModelTest).count() == 0
        assert session.query(ImageAnalysis).count() == 0
        assert session.query(CellResult).count() == 0
        
        session.close()
    
    def test_get_or_create_dataset_new(self, temp_db):
        """
        Test creating a new dataset.
        
        What it tests: Dataset creation
        Expected output: New dataset ID returned
        Verifies: Dataset record inserted
        """
        dataset_id = temp_db.get_or_create_dataset("test-dataset", "correct")
        
        assert dataset_id is not None
        assert isinstance(dataset_id, int)
        
        # Verify in database
        session = temp_db.Session()
        dataset = session.query(Dataset).filter_by(id=dataset_id).first()
        assert dataset is not None
        assert dataset.name == "test-dataset"
        assert dataset.category == "correct"
        session.close()
    
    def test_get_or_create_dataset_existing(self, temp_db):
        """
        Test retrieving existing dataset.
        
        What it tests: Dataset retrieval without duplication
        Expected output: Same ID returned for existing dataset
        Verifies: No duplicate datasets created
        """
        # Create dataset
        id1 = temp_db.get_or_create_dataset("test-dataset", "correct")
        
        # Try to create again
        id2 = temp_db.get_or_create_dataset("test-dataset", "correct")
        
        # Should return same ID
        assert id1 == id2
        
        # Verify only one record exists
        session = temp_db.Session()
        count = session.query(Dataset).filter_by(name="test-dataset").count()
        assert count == 1
        session.close()
    
    def test_get_or_create_original_image_new(self, temp_db):
        """
        Test creating a new original image record.
        
        What it tests: Image record creation
        Expected output: New image ID returned
        Verifies: Image metadata stored correctly
        """
        dataset_id = temp_db.get_or_create_dataset("test-dataset", "correct")
        
        image_id = temp_db.get_or_create_original_image(
            dataset_id=dataset_id,
            filename="test.png",
            file_path="/path/to/test.png",
            width=1024,
            height=768
        )
        
        assert image_id is not None
        
        # Verify in database
        session = temp_db.Session()
        image = session.query(OriginalImage).filter_by(id=image_id).first()
        assert image.filename == "test.png"
        assert image.file_path == "/path/to/test.png"
        assert image.width == 1024
        assert image.height == 768
        session.close()
    
    def test_get_or_create_original_image_existing(self, temp_db):
        """
        Test retrieving existing original image.
        
        What it tests: Image retrieval without duplication
        Expected output: Same ID for existing image
        Verifies: No duplicate image records
        """
        dataset_id = temp_db.get_or_create_dataset("test-dataset", "correct")
        
        id1 = temp_db.get_or_create_original_image(
            dataset_id, "test.png", "/path/test.png", 100, 100
        )
        
        id2 = temp_db.get_or_create_original_image(
            dataset_id, "test.png", "/path/test.png", 100, 100
        )
        
        assert id1 == id2
        
        # Verify only one record
        session = temp_db.Session()
        count = session.query(OriginalImage).filter_by(filename="test.png").count()
        assert count == 1
        session.close()
    
    def test_start_model_test(self, temp_db):
        """
        Test starting a new model test run.
        
        What it tests: Model test initialization
        Expected output: Test ID with start timestamp
        Verifies: Test record created correctly
        """
        test_id = temp_db.start_model_test("openai/gpt-4o-mini", test_all_mode=False)
        
        assert test_id is not None
        
        # Verify in database
        session = temp_db.Session()
        test = session.query(ModelTest).filter_by(id=test_id).first()
        assert test.model_name == "openai/gpt-4o-mini"
        assert test.test_all_mode is False
        assert test.started_at is not None
        assert test.completed_at is None  # Not yet completed
        session.close()
    
    def test_complete_model_test(self, temp_db):
        """
        Test marking model test as completed.
        
        What it tests: Test completion timestamp
        Expected output: completed_at field populated
        Verifies: Test completion tracked
        """
        test_id = temp_db.start_model_test("openai/gpt-4o-mini")
        
        temp_db.complete_model_test(test_id)
        
        # Verify completion timestamp
        session = temp_db.Session()
        test = session.query(ModelTest).filter_by(id=test_id).first()
        assert test.completed_at is not None
        assert test.completed_at >= test.started_at
        session.close()
    
    def test_save_analysis_new(self, temp_db):
        """
        Test saving a new analysis with cell results.
        
        What it tests: Complete analysis persistence
        Expected output: Analysis and cell records created
        Verifies: All data stored correctly
        """
        # Setup
        dataset_id = temp_db.get_or_create_dataset("test-dataset", "faulty")
        image_id = temp_db.get_or_create_original_image(
            dataset_id, "test.png", "/path/test.png", 300, 300
        )
        test_id = temp_db.start_model_test("openai/gpt-4o-mini")
        
        # Cell results
        cell_results = {
            (0, 0): {"issue_count": 2, "issues": ["Issue 1", "Issue 2"]},
            (1, 1): {"issue_count": 0, "issues": []},
            (2, 2): {"issue_count": 3, "issues": ["I1", "I2", "I3"]},
        }
        
        analysis_id = temp_db.save_analysis(
            image_id, test_id, "/output/test.png", cell_results
        )
        
        assert analysis_id is not None
        
        # Verify analysis
        session = temp_db.Session()
        analysis = session.query(ImageAnalysis).filter_by(id=analysis_id).first()
        assert analysis.total_issues == 5  # 2 + 0 + 3
        assert analysis.output_path == "/output/test.png"
        
        # Verify cell results
        cells = session.query(CellResult).filter_by(analysis_id=analysis_id).all()
        assert len(cells) == 3
        
        # Check specific cell
        cell_00 = next(c for c in cells if c.row == 0 and c.col == 0)
        assert cell_00.issue_count == 2
        assert cell_00.severity == "mild"
        issues = json.loads(cell_00.issues_json)
        assert len(issues) == 2
        
        session.close()
    
    def test_save_analysis_replaces_existing(self, temp_db):
        """
        Test that saving analysis replaces previous results.
        
        What it tests: Analysis replacement (upsert behavior)
        Expected output: Old analysis deleted, new one saved
        Verifies: No duplicate analyses for same image+model
        """
        # Setup
        dataset_id = temp_db.get_or_create_dataset("test-dataset", "faulty")
        image_id = temp_db.get_or_create_original_image(
            dataset_id, "test.png", "/path/test.png", 300, 300
        )
        test_id = temp_db.start_model_test("openai/gpt-4o-mini")
        
        # First analysis
        cell_results_1 = {
            (0, 0): {"issue_count": 1, "issues": ["Old issue"]},
        }
        analysis_id_1 = temp_db.save_analysis(
            image_id, test_id, "/output/v1.png", cell_results_1
        )
        
        # Second analysis (should replace first)
        cell_results_2 = {
            (0, 0): {"issue_count": 2, "issues": ["New issue 1", "New issue 2"]},
        }
        analysis_id_2 = temp_db.save_analysis(
            image_id, test_id, "/output/v2.png", cell_results_2
        )
        
        # Should be different IDs (or same if replacement logic changed)
        # The key test is that only ONE analysis exists
        
        # Only one analysis should exist
        session = temp_db.Session()
        count = session.query(ImageAnalysis).filter_by(
            original_image_id=image_id, model_test_id=test_id
        ).count()
        assert count == 1
        
        # Should be the new one
        analysis = session.query(ImageAnalysis).filter_by(id=analysis_id_2).first()
        assert analysis.total_issues == 2
        assert analysis.output_path == "/output/v2.png"
        
        session.close()
    
    def test_get_latest_analysis(self, temp_db):
        """
        Test retrieving most recent analysis.
        
        What it tests: Latest analysis query
        Expected output: Most recent analysis data
        Verifies: Correct sorting by timestamp
        """
        # Setup
        dataset_id = temp_db.get_or_create_dataset("test-dataset", "faulty")
        image_id = temp_db.get_or_create_original_image(
            dataset_id, "test.png", "/path/test.png", 300, 300
        )
        test_id = temp_db.start_model_test("openai/gpt-4o-mini")
        
        cell_results = {
            (0, 0): {"issue_count": 3, "issues": ["I1", "I2", "I3"]},
        }
        
        temp_db.save_analysis(image_id, test_id, "/output/test.png", cell_results)
        
        # Retrieve
        result = temp_db.get_latest_analysis(image_id, "openai/gpt-4o-mini")
        
        assert result is not None
        assert result["total_issues"] == 3
        assert result["output_path"] == "/output/test.png"
        assert (0, 0) in result["cell_results"]
        assert result["cell_results"][(0, 0)]["issue_count"] == 3
    
    def test_get_latest_analysis_not_found(self, temp_db):
        """
        Test query for non-existent analysis.
        
        What it tests: Missing data handling
        Expected output: None returned
        Verifies: Graceful handling of missing records
        """
        result = temp_db.get_latest_analysis(999, "nonexistent/model")
        assert result is None
    
    def test_get_all_test_results(self, temp_db):
        """
        Test retrieving all test results.
        
        What it tests: Bulk result retrieval
        Expected output: List of all analyses
        Verifies: Multiple results returned correctly
        """
        # Setup multiple analyses
        dataset_id = temp_db.get_or_create_dataset("test-dataset", "faulty")
        image_id_1 = temp_db.get_or_create_original_image(
            dataset_id, "img1.png", "/path/img1.png", 300, 300
        )
        image_id_2 = temp_db.get_or_create_original_image(
            dataset_id, "img2.png", "/path/img2.png", 300, 300
        )
        
        test_id = temp_db.start_model_test("openai/gpt-4o-mini", test_all_mode=True)
        
        cell_results = {(0, 0): {"issue_count": 1, "issues": ["Test"]}}
        
        temp_db.save_analysis(image_id_1, test_id, "/output/img1.png", cell_results)
        temp_db.save_analysis(image_id_2, test_id, "/output/img2.png", cell_results)
        
        # Retrieve all
        results = temp_db.get_all_test_results(test_all_mode=True)
        
        assert len(results) >= 2
        assert any(r["image"] == "img1.png" for r in results)
        assert any(r["image"] == "img2.png" for r in results)
    
    def test_get_model_image_issues(self, temp_db):
        """
        Test retrieving detailed issues for model+image.
        
        What it tests: Detailed issue retrieval
        Expected output: Complete analysis with all issues
        Verifies: All data accessible in single query
        """
        # Setup
        dataset_id = temp_db.get_or_create_dataset("test-dataset", "faulty")
        image_id = temp_db.get_or_create_original_image(
            dataset_id, "test.avif", "/path/test.avif", 300, 300
        )
        test_id = temp_db.start_model_test("openai/gpt-4o-mini")
        
        cell_results = {
            (0, 0): {"issue_count": 2, "issues": ["Issue A", "Issue B"]},
            (1, 1): {"issue_count": 1, "issues": ["Issue C"]},
        }
        
        temp_db.save_analysis(image_id, test_id, "/output/test.avif", cell_results)
        
        # Retrieve details
        result = temp_db.get_model_image_issues("openai/gpt-4o-mini", "test.avif")
        
        assert result is not None
        assert result["model_name"] == "openai/gpt-4o-mini"
        assert result["image_filename"] == "test.avif"
        assert result["total_issues"] == 3
        assert len(result["all_issues"]) == 3
        assert "Issue A" in result["all_issues"]
        assert "Issue C" in result["all_issues"]
        assert len(result["cell_details"]) == 2


class TestDatabaseModels:
    """Test suite for SQLAlchemy model classes."""
    
    def test_dataset_model(self, temp_db):
        """
        Test Dataset model attributes and relationships.
        
        What it tests: Dataset model structure
        Expected output: Model with correct fields
        Verifies: Schema matches requirements
        """
        session = temp_db.Session()
        
        dataset = Dataset(name="test-dataset", category="correct")
        session.add(dataset)
        session.commit()
        
        # Verify fields
        assert dataset.id is not None
        assert dataset.name == "test-dataset"
        assert dataset.category == "correct"
        assert dataset.created_at is not None
        
        session.close()
    
    def test_original_image_model(self, temp_db):
        """
        Test OriginalImage model attributes.
        
        What it tests: OriginalImage model structure
        Expected output: Model with metadata fields
        Verifies: Image data properly stored
        """
        session = temp_db.Session()
        
        dataset = Dataset(name="test", category="correct")
        session.add(dataset)
        session.flush()
        
        image = OriginalImage(
            dataset_id=dataset.id,
            filename="test.png",
            file_path="/path/test.png",
            width=1920,
            height=1080
        )
        session.add(image)
        session.commit()
        
        assert image.id is not None
        assert image.width == 1920
        assert image.height == 1080
        
        session.close()
    
    def test_model_test_model(self, temp_db):
        """
        Test ModelTest model attributes.
        
        What it tests: ModelTest model structure
        Expected output: Model with timing fields
        Verifies: Test tracking properly configured
        """
        session = temp_db.Session()
        
        test = ModelTest(
            model_name="openai/gpt-4o-mini",
            test_all_mode=True
        )
        session.add(test)
        session.commit()
        
        assert test.id is not None
        assert test.model_name == "openai/gpt-4o-mini"
        assert test.test_all_mode is True
        assert test.started_at is not None
        assert test.completed_at is None
        
        session.close()
    
    def test_relationships(self, temp_db):
        """
        Test model relationships work correctly.
        
        What it tests: Foreign key relationships
        Expected output: Related objects accessible
        Verifies: ORM relationships configured
        """
        session = temp_db.Session()
        
        # Create dataset with image
        dataset = Dataset(name="test", category="correct")
        session.add(dataset)
        session.flush()
        
        image = OriginalImage(
            dataset_id=dataset.id,
            filename="test.png",
            file_path="/path/test.png",
            width=100,
            height=100
        )
        session.add(image)
        session.commit()
        
        # Access relationship
        assert image.dataset == dataset
        assert image in dataset.images
        
        session.close()


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_save_analysis_empty_cell_results(self, temp_db):
        """
        Test saving analysis with no cell results.
        
        What it tests: Empty results handling
        Expected output: Analysis with zero issues
        Verifies: Works with empty dict
        """
        dataset_id = temp_db.get_or_create_dataset("test", "correct")
        image_id = temp_db.get_or_create_original_image(
            dataset_id, "test.png", "/path/test.png", 100, 100
        )
        test_id = temp_db.start_model_test("openai/gpt-4o-mini")
        
        cell_results = {}
        
        analysis_id = temp_db.save_analysis(
            image_id, test_id, "/output/test.png", cell_results
        )
        
        assert analysis_id is not None
        
        session = temp_db.Session()
        analysis = session.query(ImageAnalysis).filter_by(id=analysis_id).first()
        assert analysis.total_issues == 0
        session.close()
    
    def test_unicode_handling(self, temp_db):
        """
        Test handling of unicode characters in issues.
        
        What it tests: Unicode string storage
        Expected output: Unicode preserved correctly
        Verifies: Database handles UTF-8
        """
        dataset_id = temp_db.get_or_create_dataset("test", "faulty")
        image_id = temp_db.get_or_create_original_image(
            dataset_id, "test.png", "/path/test.png", 100, 100
        )
        test_id = temp_db.start_model_test("openai/gpt-4o-mini")
        
        # Unicode issues
        cell_results = {
            (0, 0): {
                "issue_count": 2,
                "issues": ["Issue with émojis 🚨", "中文 characters"]
            }
        }
        
        analysis_id = temp_db.save_analysis(
            image_id, test_id, "/output/test.png", cell_results
        )
        
        # Retrieve and verify
        result = temp_db.get_latest_analysis(image_id, "openai/gpt-4o-mini")
        issues = result["cell_results"][(0, 0)]["issues"]
        
        assert "émojis 🚨" in issues[0]
        assert "中文" in issues[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
