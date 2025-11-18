# HYGO - AI-Generated Image Error Detection

A Python project for detecting visual errors in AI-generated images using vision language models. The system splits images into a 3x3 grid, analyzes each cell with a multimodal LLM via OpenRouter, and overlays colored severity markers on the original image.

## Features

- **Grid-based Analysis**: Splits images into 3×3 regions for focused error detection
- **Vision LLM Integration**: Uses OpenRouter API for multimodal analysis
- **Severity Classification**: Categorizes issues as mild (yellow) or severe (red)
- **Batch Processing**: Handles multiple images efficiently with progress tracking
- **Clean Output**: Annotated images with grid lines and severity markers

## Project Structure

```
HYGO/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management
│   ├── image_utils.py      # Image processing utilities
│   ├── analyzer.py         # LLM client and analysis logic
│   ├── visualizer.py       # Visualization and overlay
│   └── main.py             # Main pipeline orchestration
├── dataset/
│   ├── correct-ai-images/  # Images expected to be correct
│   └── faulty-ai-images/   # Images with known issues
├── output/                 # Generated annotated images
├── docs/                   # Strategy and research documentation
├── .github/
│   └── copilot-instructions.md  # AI agent guidelines
├── environment.yaml        # Conda environment specification
├── .env.example           # Configuration template
└── README.md              # This file
```

## Setup

### 1. Create Conda Environment

```bash
conda env update -f environment.yaml --prune -n vision-agent
conda activate vision-agent
```

### 2. Configure OpenRouter API

Copy the example configuration:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Get your API key from: https://openrouter.ai/keys

### 3. Add Images

Place images in the dataset directories:

- `dataset/correct-ai-images/` - Images expected to have no issues
- `dataset/faulty-ai-images/` - Images with known visual errors

Supported formats: PNG, JPG, JPEG, WEBP, BMP, AVIF

## Usage

Run the analysis pipeline:

```bash
python -m src.main
```

This will:
1. Load configuration from `.env`
2. Initialize OpenRouter client
3. Process all images in dataset folders
4. Save annotated images to `output/` directory
5. Print analysis summary to console

## Output

For each input image, the system generates:

- **Annotated Image**: Original image with:
  - White 3×3 grid lines showing analysis cells
  - Yellow dots (●) for cells with 1-2 issues (mild)
  - Red dots (●) for cells with 3+ issues (severe)
  
- **Console Summary**: Per-cell issue counts and descriptions

Example output structure:
```
output/
├── correct-ai-images/
│   └── image1.avif (annotated)
└── faulty-ai-images/
    └── image2.avif (annotated)
```

## Architecture

The system follows a modular design optimized for efficiency:

1. **config.py**: Loads and validates configuration from `.env` file
2. **image_utils.py**: Handles image loading, grid splitting, and encoding
3. **analyzer.py**: Manages OpenRouter API calls and response parsing
4. **visualizer.py**: Creates annotated images with grid and markers
5. **main.py**: Orchestrates the complete pipeline

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
2. **Image Processing**: Extend `src/image_utils.py` for new image operations
3. **Analysis Logic**: Modify `src/analyzer.py` for different prompts or models
4. **Visualization**: Update `src/visualizer.py` for custom overlays

Refer to `.github/copilot-instructions.md` for AI agent guidance on coding patterns.

## License

See project license file for terms.

## References

- OpenRouter API: https://openrouter.ai/docs
- Strategy documentation: `docs/strategy.md`
- Research: `docs/ai-generated-image-error-detection-strategy-outline-and-research.md`
