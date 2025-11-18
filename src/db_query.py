"""
Database query utility for HYGO project.

Provides convenient functions to query and analyze results from the database.
Run from project root: python -m src.db_query

Examples:
    python -m src.db_query --summary
    python -m src.db_query --compare-models
    python -m src.db_query --image-details "image.avif"
"""

import argparse
from pathlib import Path

from .database import DatabaseService


def print_summary(db: DatabaseService):
    """Print overall database summary."""
    from sqlalchemy import func
    from .database import Dataset, OriginalImage, ModelTest, ImageAnalysis
    
    session = db.Session()
    try:
        # Count datasets
        dataset_count = session.query(Dataset).count()
        
        # Count images
        image_count = session.query(OriginalImage).count()
        
        # Count tests
        test_count = session.query(ModelTest).count()
        test_all_count = session.query(ModelTest).filter_by(test_all_mode=True).count()
        
        # Count analyses
        analysis_count = session.query(ImageAnalysis).count()
        
        # Average issues
        avg_issues = session.query(func.avg(ImageAnalysis.total_issues)).scalar()
        
        print("\n" + "=" * 60)
        print("DATABASE SUMMARY")
        print("=" * 60)
        print(f"Datasets:        {dataset_count}")
        print(f"Images:          {image_count}")
        print(f"Model Tests:     {test_count} ({test_all_count} in TEST_ALL mode)")
        print(f"Analyses:        {analysis_count}")
        print(f"Avg Issues:      {avg_issues:.2f}" if avg_issues else "Avg Issues:      N/A")
        print("=" * 60 + "\n")
        
    finally:
        session.close()


def print_model_comparison(db: DatabaseService):
    """Compare performance across different models."""
    from sqlalchemy import func
    from .database import ModelTest, ImageAnalysis
    
    session = db.Session()
    try:
        # Query model performance
        results = (
            session.query(
                ModelTest.model_name,
                func.count(ImageAnalysis.id).label('image_count'),
                func.avg(ImageAnalysis.total_issues).label('avg_issues'),
                func.min(ImageAnalysis.total_issues).label('min_issues'),
                func.max(ImageAnalysis.total_issues).label('max_issues'),
            )
            .join(ImageAnalysis)
            .group_by(ModelTest.model_name)
            .order_by(func.avg(ImageAnalysis.total_issues))
            .all()
        )
        
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE COMPARISON")
        print("=" * 80)
        print(f"{'Model':<40} {'Images':<10} {'Avg':<10} {'Min':<10} {'Max':<10}")
        print("-" * 80)
        
        for row in results:
            print(f"{row.model_name:<40} {row.image_count:<10} "
                  f"{row.avg_issues:<10.2f} {row.min_issues:<10} {row.max_issues:<10}")
        
        print("=" * 80 + "\n")
        
    finally:
        session.close()


def print_cell_heatmap(db: DatabaseService):
    """Show which grid cells have most issues."""
    from sqlalchemy import func
    from .database import CellResult
    
    session = db.Session()
    try:
        # Query cell statistics
        results = (
            session.query(
                CellResult.row,
                CellResult.col,
                func.avg(CellResult.issue_count).label('avg_issues'),
                func.count(CellResult.id).label('count'),
            )
            .group_by(CellResult.row, CellResult.col)
            .all()
        )
        
        # Build 3x3 grid
        grid = {}
        for row in results:
            grid[(row.row, row.col)] = (row.avg_issues, row.count)
        
        print("\n" + "=" * 60)
        print("GRID CELL HEATMAP (Average Issues per Cell)")
        print("=" * 60)
        print("Layout: (Row, Col) = Avg Issues [Sample Count]\n")
        
        for r in range(3):
            row_str = ""
            for c in range(3):
                if (r, c) in grid:
                    avg, count = grid[(r, c)]
                    row_str += f"  ({r},{c})={avg:4.2f}[{count:3d}]  "
                else:
                    row_str += f"  ({r},{c})= N/A [  0]  "
            print(row_str)
        
        print("=" * 60 + "\n")
        
    finally:
        session.close()


def print_image_details(db: DatabaseService, filename: str):
    """Print detailed results for a specific image."""
    from sqlalchemy import desc
    from .database import OriginalImage, ImageAnalysis, ModelTest, CellResult
    import json
    
    session = db.Session()
    try:
        # Find image
        image = session.query(OriginalImage).filter(
            OriginalImage.filename.like(f"%{filename}%")
        ).first()
        
        if not image:
            print(f"\n[ERROR] Image matching '{filename}' not found in database.\n")
            return
        
        print("\n" + "=" * 60)
        print(f"IMAGE DETAILS: {image.filename}")
        print("=" * 60)
        print(f"Path:       {image.file_path}")
        print(f"Size:       {image.width}x{image.height}")
        print(f"Dataset:    {image.dataset.name} ({image.dataset.category})")
        print(f"Created:    {image.created_at}")
        print("-" * 60)
        
        # Get all analyses for this image
        analyses = (
            session.query(ImageAnalysis)
            .join(ModelTest)
            .filter(ImageAnalysis.original_image_id == image.id)
            .order_by(desc(ImageAnalysis.analyzed_at))
            .all()
        )
        
        if not analyses:
            print("No analyses found for this image.")
        else:
            print(f"\nANALYSES ({len(analyses)} total):\n")
            
            for idx, analysis in enumerate(analyses, 1):
                print(f"  [{idx}] Model: {analysis.model_test.model_name}")
                print(f"      Total Issues: {analysis.total_issues}")
                print(f"      Analyzed: {analysis.analyzed_at}")
                print(f"      Output: {analysis.output_path}")
                
                # Show cell details
                cells_with_issues = [c for c in analysis.cell_results if c.issue_count > 0]
                if cells_with_issues:
                    print(f"      Cells with Issues:")
                    for cell in cells_with_issues:
                        issues = json.loads(cell.issues_json) if cell.issues_json else []
                        print(f"        ({cell.row},{cell.col}): {cell.issue_count} issues - {cell.severity}")
                        for issue in issues:
                            print(f"          - {issue}")
                else:
                    print(f"      No issues detected in any cell")
                
                print()
        
        print("=" * 60 + "\n")
        
    finally:
        session.close()


def print_dataset_summary(db: DatabaseService):
    """Print summary per dataset."""
    from sqlalchemy import func
    from .database import Dataset, OriginalImage, ImageAnalysis
    
    session = db.Session()
    try:
        datasets = session.query(Dataset).all()
        
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        
        for dataset in datasets:
            image_count = session.query(OriginalImage).filter_by(dataset_id=dataset.id).count()
            
            # Get analysis stats
            avg_issues = (
                session.query(func.avg(ImageAnalysis.total_issues))
                .join(OriginalImage)
                .filter(OriginalImage.dataset_id == dataset.id)
                .scalar()
            )
            
            print(f"\nDataset: {dataset.name} ({dataset.category})")
            print(f"  Images:      {image_count}")
            print(f"  Avg Issues:  {avg_issues:.2f}" if avg_issues else "  Avg Issues:  N/A")
        
        print("\n" + "=" * 60 + "\n")
        
    finally:
        session.close()


def print_model_image_issues(db: DatabaseService, model_name: str, image_filename: str):
    """Print all issues for a specific model and image."""
    result = db.get_model_image_issues(model_name, image_filename)
    
    if not result:
        print(f"\n[ERROR] No analysis found for model '{model_name}' and image '{image_filename}'\n")
        return
    
    print("\n" + "=" * 80)
    print("MODEL + IMAGE DETAILED ANALYSIS")
    print("=" * 80)
    print(f"Model:          {result['model_name']}")
    print(f"Image:          {result['image_filename']}")
    print(f"Analyzed:       {result['analyzed_at']}")
    print(f"Total Issues:   {result['total_issues']}")
    print(f"\nRaw Image:      {result['raw_image_path']}")
    print(f"Annotated:      {result['annotated_image_path']}")
    
    print("\n" + "-" * 80)
    print("ALL DETECTED ISSUES:")
    print("-" * 80)
    if result['all_issues']:
        for idx, issue in enumerate(result['all_issues'], 1):
            print(f"  {idx}. {issue}")
    else:
        print("  (No issues detected)")
    
    print("\n" + "-" * 80)
    print("CELL-BY-CELL BREAKDOWN (3x3 Grid):")
    print("-" * 80)
    for cell in result['cell_details']:
        if cell['issue_count'] > 0:
            print(f"\nCell [{cell['row']}, {cell['col']}] - {cell['severity'].upper()} ({cell['issue_count']} issues):")
            for idx, issue in enumerate(cell['issues'], 1):
                print(f"    {idx}. {issue}")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Command-line interface for database queries."""
    parser = argparse.ArgumentParser(
        description="Query HYGO analysis database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--db",
        default="hygo_results.db",
        help="Path to database file (default: hygo_results.db)"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show overall database summary"
    )
    
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare performance across models"
    )
    
    parser.add_argument(
        "--cell-heatmap",
        action="store_true",
        help="Show grid cell issue heatmap"
    )
    
    parser.add_argument(
        "--datasets",
        action="store_true",
        help="Show dataset summary"
    )
    
    parser.add_argument(
        "--image-details",
        metavar="FILENAME",
        help="Show detailed results for specific image"
    )
    
    parser.add_argument(
        "--model-image",
        nargs=2,
        metavar=("MODEL", "IMAGE"),
        help="Get all issues for specific model+image (e.g., --model-image 'openai/gpt-4o-mini' 'image.avif')"
    )
    
    args = parser.parse_args()
    
    # Initialize database
    if not Path(args.db).exists():
        print(f"\n[ERROR] Database not found: {args.db}")
        print("Run the main pipeline first to create the database.\n")
        return
    
    db = DatabaseService(args.db)
    
    # Execute requested queries
    if args.summary:
        print_summary(db)
    
    if args.compare_models:
        print_model_comparison(db)
    
    if args.cell_heatmap:
        print_cell_heatmap(db)
    
    if args.datasets:
        print_dataset_summary(db)
    
    if args.image_details:
        print_image_details(db, args.image_details)
    
    if args.model_image:
        print_model_image_issues(db, args.model_image[0], args.model_image[1])
    
    # If no arguments, show summary
    if not any([args.summary, args.compare_models, args.cell_heatmap, 
                args.datasets, args.image_details, args.model_image]):
        print_summary(db)


if __name__ == "__main__":
    main()
