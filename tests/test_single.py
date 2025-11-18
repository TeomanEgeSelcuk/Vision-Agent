"""Quick test script to verify model fallback chain works with single image."""

from pathlib import Path
import sys
from pathlib import Path as _Path

# Ensure imports work when running this test as a script
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.image_utils import load_image, split_into_grid
from src.analyzer import create_openrouter_client, analyze_image_grid
from src.visualizer import create_annotated_image

def main():
    print("Testing single image analysis...")
    
    # Load config
    cfg = load_config()
    print(f"Model: {cfg['model']}")
    
    # Create client
    client = create_openrouter_client(
        cfg["api_key"],
        cfg["referer"],
        cfg["app_title"]
    )
    
    # Test with one faulty image
    test_path = Path("dataset/faulty-ai-images/hand.png")
    if not test_path.exists():
        print(f"Test image not found: {test_path}")
        return
    
    print(f"\nAnalyzing: {test_path.name}")
    
    # Load and split image
    img = load_image(test_path)
    regions = split_into_grid(img)
    print(f"Split into {len(regions)} regions")
    
    # Analyze each region
    results = analyze_image_grid(client, cfg["model"], regions)
    
    # Print results
    print("\nResults:")
    total_issues = 0
    for (row, col), result in sorted(results.items()):
        count = result["issue_count"]
        total_issues += count
        cell_num = row * 3 + col + 1  # Convert to 1-9 numbering
        if count > 0:
            print(f"  Cell {cell_num} ({row},{col}): {count} issue(s)")
            for issue in result["issues"]:
                print(f"    - {issue}")
    
    if total_issues == 0:
        print("  No issues detected in any cell")
    else:
        print(f"\nTotal: {total_issues} issues across all cells")
    
    # Create annotated output
    annotated = create_annotated_image(img, results)
    output_path = Path("test_output.png")
    annotated.save(output_path)
    print(f"\nSaved annotated image to: {output_path}")

if __name__ == "__main__":
    main()
