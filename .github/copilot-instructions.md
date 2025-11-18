# Copilot Instructions for HYGO Project

## Project Overview
HYGO is a Python project for detecting errors in AI-generated images using vision language models. It splits images into a 3x3 grid, analyzes each cell with a multimodal LLM via OpenRouter, counts visual anomalies, and overlays colored dots (yellow for mild, red for severe issues) on the original image.

## Architecture
- **Core Pipeline**: Image grid splitting (strategy.md), LLM analysis per cell, severity-based overlay.
- **Data Flow**: Input images from `dataset/` folders, output annotated images to `output/` folders.
- **Key Components**: OpenAI-compatible client for OpenRouter, JSON-mode responses for structured issue detection.

## Development Setup
- Use conda environment defined in `environment.yaml`.
- Update environment: `conda env update -f environment.yaml --prune -n vision-agent`
- Activate: `conda activate vision-agent`
- Dependencies: LangChain for orchestration, OpenAI SDK with OpenRouter base_url, Pillow/OpenCV for image handling.

## Coding Patterns
- **LLM Integration**: Use `openai.OpenAI(base_url="https://openrouter.ai/api/v1")` for vision models. Encode images as base64 data URLs.
- **Prompts**: Structured prompts for error detection, e.g., "List clear, objective visual errors... Return JSON with issue_count and issues array."
- **Response Handling**: Always use `response_format={"type": "json_object"}` for parseable JSON output.
- **Image Processing**: Split images into 3x3 grid using width//3, height//3. Process cells independently.
- **Output**: Draw grid lines and filled circles at cell centers based on issue_count thresholds (0: none, 1-2: yellow, >=3: red).

## Coding Standards
- **Efficiency**: Optimize all code for the best possible space and time complexity.
- **Documentation**: Write comprehensive docstrings for functions and classes. Include inline comments in simple, short terms throughout the code.
- **Organization**: Structure code logically, referencing logic and implementations documented in the `docs/` folder.

## Workflows
- **Image Analysis**: For each image, crop 9 cells, send to LLM, aggregate issues, annotate original.
- **Dataset**: `dataset/faulty-ai-images/` and `dataset/correct-ai-images/` contain .avif files; mirror to `output/` with annotations.
- **Debugging**: Check LLM responses for issue arrays; verify base64 encoding and JSON parsing.

## Conventions
- Use absolute paths for file operations.
- Handle .avif images with Pillow (may need additional libs if issues).
- Focus on cost-efficiency: Minimize API calls, use local models where possible (though current strategy uses OpenRouter).

Reference: `strategy.md` and `docs/` for pipeline details and logic, `environment.yaml` for deps.