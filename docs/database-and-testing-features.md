# Database and Testing Features

## Overview
HYGO now includes comprehensive database tracking and multi-model testing capabilities. All analysis results are automatically stored in a SQLite database, and a TEST_ALL mode enables systematic evaluation of multiple vision models.

## Database Integration

### Schema
The database tracks five main entities:

1. **Datasets** - Image collections (correct-ai-images, faulty-ai-images)
2. **OriginalImages** - Source images with metadata (filename, path, dimensions)
3. **ModelTests** - Test runs with specific model configurations
4. **ImageAnalyses** - Overall analysis results per image+model
5. **CellResults** - Per-cell (3x3 grid) issue counts and descriptions

### Features
- **Automatic recording**: All analyses stored in database automatically
- **Duplicate handling**: Re-running analysis replaces existing results for same image+model
- **Result persistence**: Database preserves full history of model tests
- **Query support**: Retrieve past analyses, compare model performance
- **Structured storage**: Cell-level issues stored as JSON for flexible analysis

### Database Location
Default: `hygo_results.db` in project root
Configurable via `DATABASE_PATH` in `.env` file

## TEST_ALL Mode

### Purpose
Systematically test all fallback models to compare performance across different vision LLMs.

### Models Tested (in order)
1. `openai/gpt-4o-mini` - Cost-effective baseline
2. `google/gemini-flash-1.5` - Fast alternative
3. `anthropic/claude-3-haiku` - Reliable fallback
4. `openai/gpt-4o` - High-capability option

### Enabling TEST_ALL
Set in `.env` file:
```env
TEST_ALL=true
```

### Behavior
- Processes all images with each model sequentially
- Creates separate output directories per model: `output/correct-ai-images/openai_gpt-4o-mini/`
- Records all results in database with `test_all_mode=True` flag
- Continues with next model if one fails (non-fatal errors)
- Provides progress tracking for each model

### Output Structure (TEST_ALL mode)
```
output/
├── correct-ai-images/
│   ├── openai_gpt-4o-mini/
│   │   ├── image1.avif
│   │   └── image2.avif
│   ├── google_gemini-flash-1.5/
│   │   ├── image1.avif
│   │   └── image2.avif
│   └── ...
└── faulty-ai-images/
    ├── openai_gpt-4o-mini/
    └── ...
```

### Normal Mode (TEST_ALL=false)
- Uses only `OPENROUTER_MODEL` from config
- Standard output structure: `output/correct-ai-images/image.avif`
- Still records in database for historical tracking
- Replaces previous results for same image+model

## Numbered Issue Markers

### Enhancement
Issue markers (colored dots) now display the issue count inside the circle.

### Visual Encoding
- **Yellow dot with "2"** = 2 mild issues detected (1-2 issues)
- **Red dot with "5"** = 5 severe issues detected (3+ issues)
- **No dot** = 0 issues detected

### Implementation
- Font size scales with marker radius for readability
- White text on colored background for contrast
- Falls back to default font if TrueType fonts unavailable
- Numbers correspond to `issue_count` from LLM responses

## Configuration Reference

### Required Settings
```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

### Optional Settings
```env
# Test all fallback models (default: false)
TEST_ALL=true

# Database file path (default: hygo_results.db)
DATABASE_PATH=results/hygo.db

# OpenRouter tracking (optional)
OPENROUTER_HTTP_REFERER=http://localhost
OPENROUTER_APP_TITLE=hygo-vision-agent
```

## Database Queries (Examples)

### Using Command-Line Interface

The `db_query.py` module provides convenient CLI commands for querying results:

#### 1. Overall Summary
Get a quick overview of all database contents:
```bash
python -m src.db_query --summary
```

Output includes:
- Number of datasets
- Total images analyzed
- Number of model tests run
- Total analyses performed
- Average issues per image

#### 2. Compare Models
See performance comparison across different models:
```bash
python -m src.db_query --compare-models
```

Shows for each model:
- Number of images analyzed
- Average issues detected
- Min/max issue counts
- Sorted by average issues (best to worst)

#### 3. Get All Issues for Specific Model + Image
Retrieve detailed analysis for a specific model and image combination:
```bash
python -m src.db_query --model-image "openai/gpt-4o-mini" "image.avif"
```

This returns comprehensive information including:
- **Model name** and **image filename**
- **Analysis timestamp**
- **Total issue count**
- **Raw image path** (original input file)
- **Annotated image path** (output with markers)
- **Complete list of all detected issues** across all cells
- **Cell-by-cell breakdown** showing issues per grid cell with severity levels

Example output:
```
================================================================================
MODEL + IMAGE DETAILED ANALYSIS
================================================================================
Model:          openai/gpt-4o-mini
Image:          3d-rendered-photos-super-car.avif
Analyzed:       2025-11-18 14:23:45
Total Issues:   7

Raw Image:      dataset/correct-ai-images/3d-rendered-photos-super-car.avif
Annotated:      output/correct-ai-images/3d-rendered-photos-super-car.avif

--------------------------------------------------------------------------------
ALL DETECTED ISSUES:
--------------------------------------------------------------------------------
  1. Wheel geometry appears distorted
  2. Shadow direction inconsistent
  3. Reflection angle physically impossible
  4. Window glass has rendering artifacts
  5. Door handle proportions incorrect
  6. Headlight shape anatomically wrong
  7. Ground texture repeating unnaturally

--------------------------------------------------------------------------------
CELL-BY-CELL BREAKDOWN (3x3 Grid):
--------------------------------------------------------------------------------

Cell [0, 0] - MILD (2 issues):
    1. Wheel geometry appears distorted
    2. Shadow direction inconsistent

Cell [1, 1] - SEVERE (3 issues):
    1. Reflection angle physically impossible
    2. Window glass has rendering artifacts
    3. Door handle proportions incorrect

Cell [2, 2] - MILD (2 issues):
    1. Headlight shape anatomically wrong
    2. Ground texture repeating unnaturally
================================================================================
```

#### 4. Cell Heatmap
View which grid cells tend to have the most issues:
```bash
python -m src.db_query --cell-heatmap
```

#### 5. Dataset Summary
See statistics per dataset:
```bash
python -m src.db_query --datasets
```

#### 6. Image Details
Get all analyses for a specific image (across all models):
```bash
python -m src.db_query --image-details "image.avif"
```

### Using Python
```python
from src.database import DatabaseService

# Initialize
db = DatabaseService("hygo_results.db")

# Get latest analysis for an image
results = db.get_latest_analysis(
    original_image_id=1,
    model_name="openai/gpt-4o-mini"
)

# Get all TEST_ALL results
test_results = db.get_all_test_results(test_all_mode=True)
for result in test_results:
    print(f"{result['model']}: {result['image']} - {result['total_issues']} issues")

# Get detailed issues for specific model + image
details = db.get_model_image_issues(
    model_name="openai/gpt-4o-mini",
    image_filename="image.avif"
)
if details:
    print(f"Total issues: {details['total_issues']}")
    print(f"Raw image: {details['raw_image_path']}")
    print(f"Annotated: {details['annotated_image_path']}")
    print("All issues:", details['all_issues'])
    for cell in details['cell_details']:
        print(f"Cell [{cell['row']}, {cell['col']}]: {cell['issue_count']} issues")
```

### Direct SQL (SQLite)
```sql
-- Compare model performance
SELECT 
    mt.model_name,
    AVG(ia.total_issues) as avg_issues,
    COUNT(*) as images_analyzed
FROM image_analyses ia
JOIN model_tests mt ON ia.model_test_id = mt.id
WHERE mt.test_all_mode = 1
GROUP BY mt.model_name
ORDER BY avg_issues;

-- Get most problematic cells across all images
SELECT 
    cr.row,
    cr.col,
    AVG(cr.issue_count) as avg_issues,
    COUNT(*) as occurrences
FROM cell_results cr
GROUP BY cr.row, cr.col
ORDER BY avg_issues DESC;
```

## Performance Considerations

### Normal Mode
- **Time**: ~9 API calls per image (one per cell)
- **Cost**: Depends on `OPENROUTER_MODEL` pricing
- **Storage**: ~1-2 KB per image in database

### TEST_ALL Mode
- **Time**: ~9 × N API calls per image (N = number of models, typically 4)
- **Cost**: Higher due to multiple model calls
- **Storage**: ~1-2 KB × N per image
- **Recommended**: Use for evaluation only, not routine processing

## Workflow Recommendations

### Development/Testing
1. Use TEST_ALL=false with gpt-4o-mini (cost-effective)
2. Iterate on prompts and analysis logic
3. Check database for result consistency

### Model Evaluation
1. Set TEST_ALL=true
2. Run full dataset through all models
3. Query database to compare performance
4. Analyze which models best detect issues

### Production
1. Use TEST_ALL=false with chosen model
2. Database automatically tracks results
3. Re-running updates existing results
4. Query database for trend analysis

## Migration Notes

### Updating Environment
```bash
# Update conda environment with new dependencies
conda env update -f environment.yaml --prune -n vision-agent

# Activate environment
conda activate vision-agent
```

### Database Creation
Database is automatically created on first run. No manual setup required.

### Backward Compatibility
- Existing code works without changes
- Database features are optional (work without database if service not passed)
- Numbered markers render automatically (no code changes needed)
- TEST_ALL mode is opt-in via config

## Troubleshooting

### Database Locked
If you see "database is locked" errors:
- Close any SQLite browser tools
- Ensure only one process accesses database at a time

### Font Rendering
If numbers don't display in markers:
- System falls back to default font automatically
- Install TrueType fonts for better rendering (arial.ttf, DejaVuSans, etc.)

### TEST_ALL Taking Too Long
- Reduce dataset size for testing
- Use fewer models (edit `CHEAP_VISION_MODELS` in `analyzer.py`)
- Consider running overnight for full evaluation
