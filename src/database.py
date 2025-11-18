"""
Database models and service for HYGO project.

Handles SQLite database operations for tracking model tests and results.
Optimized for efficient storage and retrieval of image analysis data.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Boolean,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import json


Base = declarative_base()


class Dataset(Base):
    """
    Represents a dataset of images (correct or faulty).
    
    Tracks metadata about image datasets used for testing.
    """
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)  # e.g., "correct-ai-images"
    category = Column(String(50), nullable=False)  # "correct" or "faulty"
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to images in this dataset
    images = relationship("OriginalImage", back_populates="dataset")


class OriginalImage(Base):
    """
    Represents an original input image from a dataset.
    
    Stores metadata about source images before analysis.
    """
    __tablename__ = "original_images"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)  # Absolute path
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="images")
    analyses = relationship("ImageAnalysis", back_populates="original_image")


class ModelTest(Base):
    """
    Represents a test run with specific model configuration.
    
    Tracks when and how models were tested (single or TEST_ALL mode).
    """
    __tablename__ = "model_tests"
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), nullable=False)
    test_all_mode = Column(Boolean, default=False)  # True if part of TEST_ALL run
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    analyses = relationship("ImageAnalysis", back_populates="model_test")


class ImageAnalysis(Base):
    """
    Represents a single image analysis result.
    
    Stores overall analysis metadata and links to cell-level details.
    """
    __tablename__ = "image_analyses"
    
    id = Column(Integer, primary_key=True)
    original_image_id = Column(Integer, ForeignKey("original_images.id"), nullable=False)
    model_test_id = Column(Integer, ForeignKey("model_tests.id"), nullable=False)
    output_path = Column(String(1000), nullable=False)  # Path to annotated image
    total_issues = Column(Integer, default=0)  # Sum across all cells
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    original_image = relationship("OriginalImage", back_populates="analyses")
    model_test = relationship("ModelTest", back_populates="analyses")
    cell_results = relationship("CellResult", back_populates="analysis")


class CellResult(Base):
    """
    Represents analysis results for a single grid cell.
    
    Stores issue count, severity, and individual issue descriptions per cell.
    """
    __tablename__ = "cell_results"
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("image_analyses.id"), nullable=False)
    row = Column(Integer, nullable=False)  # 0-2
    col = Column(Integer, nullable=False)  # 0-2
    issue_count = Column(Integer, default=0)
    severity = Column(String(20), nullable=False)  # "none", "mild", "severe"
    issues_json = Column(Text, nullable=True)  # JSON array of issue descriptions
    
    # Relationship
    analysis = relationship("ImageAnalysis", back_populates="cell_results")


class DatabaseService:
    """
    Service for managing database operations.
    
    Provides high-level API for storing and retrieving analysis data.
    Thread-safe session management with automatic commit/rollback.
    """
    
    def __init__(self, db_path: str = "hygo_results.db"):
        """
        Initialize database service with SQLite backend.
        
        Creates database file and tables if they don't exist.
        Time complexity: O(1) for connection, O(n) for schema creation where n is tables
        Space complexity: O(1)
        
        Args:
            db_path: Path to SQLite database file
        """
        # Use NullPool to avoid persistent connections that can keep file handles open on Windows
        from sqlalchemy.pool import NullPool
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        # Track db path for debugging / cleanup
        self._db_path = db_path

    def close(self):
        """
        Dispose of Engine bindings and release file handles.
        Call this when you're done with the DatabaseService to ensure resources are released
        and files can be removed on Windows during test teardown.
        """
        try:
            # Dispose engine to release SQLite file locks
            if getattr(self, 'engine', None) is not None:
                self.engine.dispose()
        except Exception:
            pass

    def __del__(self):
        """Best-effort cleanup: dispose engine on object deletion."""
        try:
            self.close()
        except Exception:
            pass
    
    def get_or_create_dataset(self, name: str, category: str) -> int:
        """
        Get existing dataset or create new one.
        
        Time complexity: O(1) for query
        Space complexity: O(1)
        
        Args:
            name: Dataset name (e.g., "correct-ai-images")
            category: Dataset category ("correct" or "faulty")
            
        Returns:
            Dataset ID
        """
        session = self.Session()
        try:
            dataset = session.query(Dataset).filter_by(name=name).first()
            if not dataset:
                dataset = Dataset(name=name, category=category)
                session.add(dataset)
                session.commit()
            return dataset.id
        finally:
            session.close()
    
    def get_or_create_original_image(
        self,
        dataset_id: int,
        filename: str,
        file_path: str,
        width: int,
        height: int,
    ) -> int:
        """
        Get existing original image or create new one.
        
        Time complexity: O(1) for query
        Space complexity: O(1)
        
        Args:
            dataset_id: ID of parent dataset
            filename: Image filename
            file_path: Absolute path to image file
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            OriginalImage ID
        """
        session = self.Session()
        try:
            img = session.query(OriginalImage).filter_by(
                dataset_id=dataset_id, filename=filename
            ).first()
            if not img:
                img = OriginalImage(
                    dataset_id=dataset_id,
                    filename=filename,
                    file_path=file_path,
                    width=width,
                    height=height,
                )
                session.add(img)
                session.commit()
            return img.id
        finally:
            session.close()
    
    def start_model_test(self, model_name: str, test_all_mode: bool = False) -> int:
        """
        Start a new model test run.
        
        Time complexity: O(1)
        Space complexity: O(1)
        
        Args:
            model_name: Name of model being tested
            test_all_mode: Whether this is part of TEST_ALL run
            
        Returns:
            ModelTest ID
        """
        session = self.Session()
        try:
            test = ModelTest(model_name=model_name, test_all_mode=test_all_mode)
            session.add(test)
            session.commit()
            return test.id
        finally:
            session.close()
    
    def complete_model_test(self, test_id: int):
        """
        Mark model test as completed.
        
        Time complexity: O(1)
        Space complexity: O(1)
        
        Args:
            test_id: ID of model test to complete
        """
        session = self.Session()
        try:
            test = session.query(ModelTest).filter_by(id=test_id).first()
            if test:
                test.completed_at = datetime.utcnow()
                session.commit()
        finally:
            session.close()
    
    def save_analysis(
        self,
        original_image_id: int,
        model_test_id: int,
        output_path: str,
        cell_results: Dict[tuple, Dict[str, Any]],
    ) -> int:
        """
        Save image analysis with cell-level results.
        
        Replaces existing analysis for same image+model if present.
        Time complexity: O(n) where n is number of cells
        Space complexity: O(n)
        
        Args:
            original_image_id: ID of original image
            model_test_id: ID of model test run
            output_path: Path to annotated output image
            cell_results: Dict mapping (row, col) to analysis results
            
        Returns:
            ImageAnalysis ID
        """
        session = self.Session()
        try:
            # Delete existing analysis for this image+model if present
            existing = session.query(ImageAnalysis).filter_by(
                original_image_id=original_image_id,
                model_test_id=model_test_id,
            ).first()
            
            if existing:
                # Delete old cell results
                session.query(CellResult).filter_by(analysis_id=existing.id).delete()
                session.delete(existing)
                session.commit()
            
            # Calculate total issues
            total_issues = sum(r.get("issue_count", 0) for r in cell_results.values())
            
            # Create new analysis record
            analysis = ImageAnalysis(
                original_image_id=original_image_id,
                model_test_id=model_test_id,
                output_path=output_path,
                total_issues=total_issues,
            )
            session.add(analysis)
            session.flush()  # Get analysis.id
            
            # Save cell results
            for (row, col), result in cell_results.items():
                from .analyzer import compute_severity
                
                issue_count = result.get("issue_count", 0)
                issues = result.get("issues", [])
                severity = compute_severity(issue_count)
                
                cell = CellResult(
                    analysis_id=analysis.id,
                    row=row,
                    col=col,
                    issue_count=issue_count,
                    severity=severity,
                    issues_json=json.dumps(issues),
                )
                session.add(cell)
            
            session.commit()
            return analysis.id
        finally:
            session.close()
    
    def get_latest_analysis(self, original_image_id: int, model_name: str) -> Optional[Dict]:
        """
        Get most recent analysis for image and model.
        
        Time complexity: O(log n) for query + O(k) for k cells
        Space complexity: O(k)
        
        Args:
            original_image_id: ID of original image
            model_name: Name of model
            
        Returns:
            Dict with analysis data or None if not found
        """
        session = self.Session()
        try:
            analysis = (
                session.query(ImageAnalysis)
                .join(ModelTest)
                .filter(
                    ImageAnalysis.original_image_id == original_image_id,
                    ModelTest.model_name == model_name,
                )
                .order_by(ImageAnalysis.analyzed_at.desc())
                .first()
            )
            
            if not analysis:
                return None
            
            # Build cell results dict
            cells = {}
            for cell in analysis.cell_results:
                cells[(cell.row, cell.col)] = {
                    "issue_count": cell.issue_count,
                    "severity": cell.severity,
                    "issues": json.loads(cell.issues_json) if cell.issues_json else [],
                }
            
            return {
                "id": analysis.id,
                "output_path": analysis.output_path,
                "total_issues": analysis.total_issues,
                "analyzed_at": analysis.analyzed_at,
                "cell_results": cells,
            }
        finally:
            session.close()
    
    def get_all_test_results(self, test_all_mode: bool = True) -> List[Dict]:
        """
        Get all results from TEST_ALL runs.
        
        Time complexity: O(n) where n is number of analyses
        Space complexity: O(n)
        
        Args:
            test_all_mode: Filter for TEST_ALL runs only
            
        Returns:
            List of dicts with test results
        """
        session = self.Session()
        try:
            query = (
                session.query(ImageAnalysis)
                .join(ModelTest)
                .join(OriginalImage)
            )
            
            if test_all_mode:
                query = query.filter(ModelTest.test_all_mode == True)
            
            results = []
            for analysis in query.all():
                results.append({
                    "model": analysis.model_test.model_name,
                    "image": analysis.original_image.filename,
                    "total_issues": analysis.total_issues,
                    "output_path": analysis.output_path,
                    "analyzed_at": analysis.analyzed_at,
                })
            
            return results
        finally:
            session.close()
    
    def get_model_image_issues(self, model_name: str, image_filename: str) -> Optional[Dict]:
        """
        Get all issues for a specific model and image combination.
        
        Returns detailed information including:
        - All detected issues with descriptions
        - Cell-by-cell breakdown
        - Raw image path
        - Annotated image path
        - Total issue count
        - Analysis timestamp
        
        Time complexity: O(log n) for query + O(k) for k cells
        Space complexity: O(k)
        
        Args:
            model_name: Name of the model (e.g., "openai/gpt-4o-mini")
            image_filename: Name of the image file (e.g., "image.avif")
            
        Returns:
            Dict with complete analysis data or None if not found
        """
        session = self.Session()
        try:
            # Find the most recent analysis for this model+image combo
            analysis = (
                session.query(ImageAnalysis)
                .join(ModelTest)
                .join(OriginalImage)
                .filter(
                    ModelTest.model_name == model_name,
                    OriginalImage.filename == image_filename,
                )
                .order_by(ImageAnalysis.analyzed_at.desc())
                .first()
            )
            
            if not analysis:
                return None
            
            # Build detailed cell results with all issues
            cell_details = []
            for cell in sorted(analysis.cell_results, key=lambda c: (c.row, c.col)):
                issues_list = json.loads(cell.issues_json) if cell.issues_json else []
                cell_details.append({
                    "row": cell.row,
                    "col": cell.col,
                    "issue_count": cell.issue_count,
                    "severity": cell.severity,
                    "issues": issues_list,
                })
            
            # Collect all unique issues across cells
            all_issues = []
            for cell in analysis.cell_results:
                if cell.issues_json:
                    issues_list = json.loads(cell.issues_json)
                    all_issues.extend(issues_list)
            
            return {
                "model_name": analysis.model_test.model_name,
                "image_filename": analysis.original_image.filename,
                "raw_image_path": analysis.original_image.file_path,
                "annotated_image_path": analysis.output_path,
                "total_issues": analysis.total_issues,
                "analyzed_at": analysis.analyzed_at,
                "all_issues": all_issues,
                "cell_details": cell_details,
            }
        finally:
            session.close()
