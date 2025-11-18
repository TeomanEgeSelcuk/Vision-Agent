# Copilot Instructions for HYGO Project

## Project Overview
HYGO is a Python project for detecting errors in AI-generated images using vision language models. It splits images into a 3x3 grid, analyzes each cell with a multimodal LLM via OpenRouter, counts visual anomalies, and overlays colored dots with numbered labels (yellow for mild, red for severe issues) on the original image. All results are stored in a SQLite database for historical tracking and model comparison.

## Architecture
- **Core Pipeline**: Image grid splitting (strategy.md), LLM analysis per cell, severity-based overlay with numbered markers, database storage.
- **Data Flow**: Input images from `dataset/` folders, output annotated images to `output/` folders, results stored in `hygo_results.db`.
- **Key Components**: OpenAI-compatible client for OpenRouter, JSON-mode responses for structured issue detection, SQLAlchemy database models, TEST_ALL mode for multi-model evaluation.
- **Database**: SQLite with tables for datasets, images, model tests, analyses, and cell results. Automatic replacement of existing results when re-running same image+model.

## Development Setup
- Use conda environment defined in `environment.yaml`.
- Update environment: `conda env update -f environment.yaml --prune -n vision-agent`
- Activate: `conda activate vision-agent`
- Dependencies: LangChain for orchestration, OpenAI SDK with OpenRouter base_url, Pillow/OpenCV for image handling, SQLAlchemy for database, Alembic for migrations.

## Coding Patterns
- **LLM Integration**: Use `openai.OpenAI(base_url="https://openrouter.ai/api/v1")` for vision models. Encode images as base64 data URLs.
- **Prompts**: Structured prompts for error detection, e.g., "List clear, objective visual errors... Return JSON with issue_count and issues array."
- **Response Handling**: Always use `response_format={"type": "json_object"}` for parseable JSON output.
- **Image Processing**: Split images into 3x3 grid using width//3, height//3. Process cells independently.
- **Output**: Draw grid lines and filled circles at cell centers with issue count numbers based on thresholds (0: none, 1-2: yellow with number, >=3: red with number).
- **Database**: Use `DatabaseService` for all result storage. Pass to `process_single_image` and `process_image_batch`. Automatic upsert on re-runs.
- **TEST_ALL**: When enabled, iterate through `get_all_fallback_models()`, create separate model test records, organize outputs by model name.

## Coding Standards
- **Efficiency**: Optimize all code for the best possible space and time complexity.
- **Documentation**: Write comprehensive docstrings for functions and classes. Include inline comments in simple, short terms throughout the code.
- **Organization**: Structure code logically, referencing logic and implementations documented in the `docs/` folder.

## Workflows
- **Image Analysis**: For each image, crop 9 cells, send to LLM, aggregate issues, annotate original with numbered markers, save to database.
- **Dataset**: `dataset/faulty-ai-images/` and `dataset/correct-ai-images/` contain .avif files; mirror to `output/` with annotations.
- **Database Recording**: All analyses stored in `hygo_results.db`. Re-running same image+model replaces existing record.
- **TEST_ALL Mode**: Set `TEST_ALL=true` in .env to test all models. Outputs organized by model: `output/correct-ai-images/openai_gpt-4o-mini/`.
- **Querying**: Use `python -m src.db_query` with flags like `--summary`, `--compare-models`, `--image-details`.
- **Debugging**: Check LLM responses for issue arrays; verify base64 encoding and JSON parsing; inspect database records.

## Conventions
- Use absolute paths for file operations.
- Handle .avif images with Pillow (may need additional libs if issues).
- Focus on cost-efficiency: Minimize API calls, use local models where possible (though current strategy uses OpenRouter).

Reference: `strategy.md` and `docs/` for pipeline details and logic, `environment.yaml` for deps.