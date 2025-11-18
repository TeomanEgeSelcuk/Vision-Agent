# HYGO - AI-Generated Image Error Detection

A Python project for detecting visual errors in AI-generated images using vision language models. The system splits images into a 3x3 grid, analyzes each cell with a multimodal LLM via OpenRouter, and overlays colored severity markers on the original image.

## Features

- **Grid-based Analysis**: Splits images into 3×3 regions for focused error detection
- **Vision LLM Integration**: Uses OpenRouter API for multimodal analysis
- **Severity Classification**: Categorizes issues as mild (yellow) or severe (red)
- **Numbered Markers**: Displays issue count inside colored dots for clarity
- **Database Tracking**: Stores all analysis results in SQLite database
- **TEST_ALL Mode**: Systematically test multiple models and compare performance
- **Batch Processing**: Handles multiple images efficiently with progress tracking
- **Result Persistence**: Query historical analyses and track model performance
- **Clean Output**: Annotated images with grid lines and numbered severity markers

## Setup (minimum steps)

1) Update conda environment and activate:

```bash
conda env update -f environment.yaml --prune -n vision-agent
conda activate vision-agent
```

2) Register the kernel (optional but recommended to run notebooks):

```bash
conda run -n vision-agent python -m ipykernel install --user --name vision-agent --display-name "vision-agent"
```

3) Create `.env` from the example and add your OpenRouter key:

```bash
cp .env.example .env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

4) Add images to `dataset/` folders and run pipeline:

```bash
python -m src.main
```

Note: The default database location is `hygo_results.db` in the project root — the notebook scripts do not move the main database under `notebooks/`.
Edit `.env` and add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini

# Optional: Enable TEST_ALL mode to test all models
TEST_ALL=false

# Optional: Customize database location
DATABASE_PATH=hygo_results.db
```

Get your API key from: https://openrouter.ai/keys

Add images to `dataset/correct-ai-images/` (expected to be correct) and `dataset/faulty-ai-images/` (with known visual issues). Supported formats: PNG, JPG, JPEG, WEBP, BMP, AVIF

## Developer Notebooks

- `notebooks/run_pipeline.ipynb` — optional interactive steps for env update, kernel registration, and a short verification script. Use the `vision-agent` kernel when running it.

## Usage

### Basic Analysis

Run the analysis pipeline:

```bash
python -m src.main
```

This will:
1. Load configuration from `.env`
2. Initialize database (creates `hygo_results.db` on first run)
3. Initialize OpenRouter client
4. Process all images in dataset folders
5. Save annotated images to `output/` directory
6. Store results in database
7. Print analysis summary to console

### TEST_ALL Mode

To test all fallback models sequentially:

1. Set `TEST_ALL=true` in `.env`
2. Run: `python -m src.main`
3. Results stored separately per model in database
4. Output organized by model: `output/correct-ai-images/openai_gpt-4o-mini/`

### Query Database

Basic query examples:

```bash
python -m src.db_query --summary
python -m src.db_query --compare-models
python -m src.db_query --image-details "image.avif"
```

## Output

For each input image, the system generates:

- **Annotated Image**: Original image with:
  - White 3×3 grid lines showing analysis cells
  - Yellow dots with numbers (●1, ●2) for cells with 1-2 issues (mild)
  - Red dots with numbers (●3, ●5) for cells with 3+ issues (severe)
  - Numbers indicate exact count of detected issues
  
- **Console Summary**: Per-cell issue counts and descriptions

- **Database Record**: Full analysis results including:
  - Original image metadata
  - Model used
  - Per-cell issue counts and descriptions
  - Output path
  - Timestamp

### Normal Mode Output
```
output/
├── correct-ai-images/
│   └── image1.avif (annotated with numbered markers)
└── faulty-ai-images/
    └── image2.avif (annotated with numbered markers)
```

### TEST_ALL Mode Output
```
output/
├── correct-ai-images/
│   ├── openai_gpt-4o-mini/
│   │   └── image1.avif
│   ├── google_gemini-flash-1.5/
│   │   └── image1.avif
│   └── anthropic_claude-3-haiku/
│       └── image1.avif
└── faulty-ai-images/
    └── (similar structure)
```

## Architecture

The system follows a modular design optimized for efficiency:

1. **config.py**: Loads and validates configuration from `.env` file
2. **database.py**: SQLAlchemy models and database service for result tracking
3. **image_utils.py**: Handles image loading, grid splitting, and encoding
4. **analyzer.py**: Manages OpenRouter API calls and response parsing
5. **visualizer.py**: Creates annotated images with grid and numbered markers
6. **main.py**: Orchestrates the complete pipeline with database integration
7. **db_query.py**: Utility for querying and analyzing database results

All modules include comprehensive docstrings and inline comments explaining logic and complexity.

## Coding Standards

- **Efficiency**: All code optimized for best space and time complexity
- **Documentation**: Comprehensive docstrings for all functions and classes
- **Comments**: Inline comments explaining logic in simple, short terms
- **Organization**: Logical structure following patterns in `docs/` folder

## Strategy

The detection strategy is documented in `docs/strategy.md`. Key points:

- 3×3 grid splitting for focused analysis
- JSON-mode LLM responses for reliable parsing
- Severity thresholds: 0 (none), 1-2 (mild), 3+ (severe)
- Cost-efficient: Minimal API calls with structured prompts

See `docs/ai-generated-image-error-detection-strategy-outline-and-research.md` for detailed research and background.

## Development

To modify the pipeline:

1. **Configuration**: Edit `src/config.py` for new config options
2. **Database**: Extend `src/database.py` for new tables or queries
3. **Image Processing**: Extend `src/image_utils.py` for new image operations
4. **Analysis Logic**: Modify `src/analyzer.py` for different prompts or models
5. **Visualization**: Update `src/visualizer.py` for custom overlays
6. **Queries**: Add new queries to `src/db_query.py` for analysis

Refer to `.github/copilot-instructions.md` for AI agent guidance on coding patterns.

### New Features Documentation

For detailed information on database integration and TEST_ALL mode, see:
- `docs/database-and-testing-features.md`

This includes:
- Database schema and query examples
- TEST_ALL mode configuration and workflow
- Performance considerations
- Troubleshooting tips

## License

See project license file for terms.

## References

- OpenRouter API: https://openrouter.ai/docs
- Strategy documentation: `docs/strategy.md`
- Research: `docs/ai-generated-image-error-detection-strategy-outline-and-research.md`
