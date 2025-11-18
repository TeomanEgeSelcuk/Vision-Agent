"""
Main pipeline orchestration script for HYGO project.

Coordinates the complete image analysis workflow:
1. Load configuration
2. Initialize OpenRouter client
3. Process images from dataset folders
4. Generate annotated outputs

Run from project root: python -m src.main
"""

from pathlib import Path

from tqdm import tqdm

from .config import load_config, validate_openrouter_api_key
from .analyzer import create_openrouter_client, analyze_image_grid
from .image_utils import collect_images, load_image, split_into_grid
from .visualizer import create_annotated_image, print_analysis_summary


# Dataset and output directory structure
DATASET_ROOT = Path("dataset")
OUTPUT_ROOT = Path("output")

# Subdirectories for image categories
CORRECT_SUBDIR = "correct-ai-images"
FAULTY_SUBDIR = "faulty-ai-images"


def process_single_image(
    client,
    model: str,
    image_path: Path,
    output_dir: Path,
) -> None:
    """
    Complete analysis pipeline for a single image.
    
    Steps:
    1. Load image
    2. Split into 3x3 grid
    3. Analyze each cell with LLM
    4. Draw grid and severity markers
    5. Save annotated image
    6. Print summary
    
    Time complexity: O(API_calls * API_latency + image_processing)
    Space complexity: O(image_size * num_crops)
    
    Args:
        client: OpenRouter API client
        model: Model name to use for analysis
        image_path: Path to input image
        output_dir: Directory for output image
    """
    # Load and prepare image - O(w*h)
    img = load_image(image_path)
    
    # Split into 3x3 grid - O(9 * cell_size)
    regions = split_into_grid(img)
    
    # Analyze all regions - O(9 * API_latency)
    cell_results = analyze_image_grid(client, model, regions)
    
    # Create annotated image - O(w*h)
    annotated = create_annotated_image(img, cell_results)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save annotated image - O(w*h)
    output_path = output_dir / image_path.name
    annotated.save(output_path)
    
    # Print summary to console
    print_analysis_summary(image_path.name, cell_results)
    print(f"  → Saved to: {output_path}")


def process_image_batch(
    client,
    model: str,
    image_paths: list,
    output_dir: Path,
    description: str,
) -> None:
    """
    Process a batch of images with progress tracking.
    
    Time complexity: O(n * single_image_processing)
    Space complexity: O(single_image_size)
    
    Args:
        client: OpenRouter API client
        model: Model name for analysis
        image_paths: List of image paths to process
        output_dir: Output directory for batch
        description: Description for progress bar
    """
    # Use tqdm for progress tracking
    for img_path in tqdm(image_paths, desc=description):
        try:
            process_single_image(client, model, img_path, output_dir)
        except Exception as e:
            # Log error but continue with other images
            print(f"\nERROR processing {img_path.name}: {e}")
            continue


def main() -> None:
    """
    Main entry point for HYGO image analysis pipeline.
    
    Orchestrates complete workflow:
    - Configuration loading
    - API key validation
    - Client initialization
    - Batch processing of all images
    - Output organization
    """
    print("=" * 60)
    print("HYGO - AI Image Error Detection")
    print("=" * 60)
    
    # Load configuration from .env
    print("\n[1/6] Loading configuration...")
    try:
        cfg = load_config()
        print(f"  [OK] Model: {cfg['model']}")
    except Exception as e:
        print(f"  [ERROR] Configuration error: {e}")
        return
    
    # Validate OpenRouter API key
    print("\n[2/6] Validating OpenRouter API key...")
    try:
        is_valid = validate_openrouter_api_key(cfg["api_key"])
        if not is_valid:
            print("  [ERROR] API key validation failed. Please check your .env file.")
            return
    except Exception as e:
        print(f"  [ERROR] API key validation error: {e}")
        return
    
    # Initialize OpenRouter client
    print("\n[3/6] Initializing OpenRouter client...")
    try:
        client = create_openrouter_client(
            cfg["api_key"],
            cfg["referer"],
            cfg["app_title"]
        )
        print("  [OK] Client ready")
    except Exception as e:
        print(f"  [ERROR] Client initialization error: {e}")
        return
    
    # Collect images from dataset
    print("\n[4/6] Scanning dataset directories...")
    correct_dir = DATASET_ROOT / CORRECT_SUBDIR
    faulty_dir = DATASET_ROOT / FAULTY_SUBDIR
    
    correct_images = collect_images(correct_dir) if correct_dir.exists() else []
    faulty_images = collect_images(faulty_dir) if faulty_dir.exists() else []
    
    total_images = len(correct_images) + len(faulty_images)
    
    if total_images == 0:
        print(f"  [ERROR] No images found in {DATASET_ROOT}")
        print(f"    Expected: {correct_dir} or {faulty_dir}")
        return
    
    print(f"  [OK] Found {len(correct_images)} correct images")
    print(f"  [OK] Found {len(faulty_images)} faulty images")
    print(f"  [OK] Total: {total_images} images")
    
    # Process faulty images
    if faulty_images:
        print("\n[5/6] Processing faulty images...")
        faulty_output = OUTPUT_ROOT / FAULTY_SUBDIR
        process_image_batch(
            client,
            cfg["model"],
            faulty_images,
            faulty_output,
            "Faulty images"
        )
    
    # Process correct images
    if correct_images:
        print("\n[6/6] Processing correct images...")
        correct_output = OUTPUT_ROOT / CORRECT_SUBDIR
        process_image_batch(
            client,
            cfg["model"],
            correct_images,
            correct_output,
            "Correct images"
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Annotated images saved to: {OUTPUT_ROOT}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
