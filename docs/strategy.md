## Strategy

Given your current dataset and goal, we can simplify the original “pose / OCR / multiple tools” pipeline into:

1. **Split each image into a 3×3 grid**

   * Compute `cell_w = width // 3`, `cell_h = height // 3`.
   * Crop 9 regions: `(row, col)` in `{0,1,2} × {0,1,2}`.
   * Each crop is analyzed independently.

2. **Use a vision LLM (via OpenRouter) on each cell**

   * For every cell, send a prompt like:

     > “You are an expert at spotting visual mistakes in AI-generated images. You will see a small crop from a larger image. List clear, objective visual errors such as anatomical impossibilities, warped geometry, nonsensical text, unnatural artifacts, or other implausible details. Return JSON with keys: `issue_count` (integer) and `issues` (array of short strings). If there are no obvious errors, use `issue_count: 0` and `issues: []`.”

   * Encode the crop as a **base64 data URL** (e.g. `"data:image/jpeg;base64,..."`) and send it using the OpenAI Python client’s multimodal interface, with `base_url="https://openrouter.ai/api/v1"` so it talks to OpenRouter instead.([OpenAI Platform][1])

   * Use **JSON mode** (`response_format={"type": "json_object"}`) so the model is constrained to output valid JSON you can parse safely.([Stack Overflow][2])

3. **Compute severity per cell from `issue_count`**

   For each cell:

   * `issue_count == 0` → **no dot** (no obvious issues).
   * `1 ≤ issue_count ≤ 2` → **yellow dot** (mild anomalies).
   * `issue_count ≥ 3` → **red dot** (severe / many anomalies).

   (You can tweak thresholds later, but this matches your “between one and three vs three or more mistakes” idea.)

4. **Draw overlay on the original image**

   * Draw **grid lines** (light color) so it’s obvious where the 3×3 cells are.
   * For each cell with `issue_count > 0`, draw a **filled circle** at the cell center:

     * Yellow for mild (1–2 issues),
     * Red for severe (≥3 issues).
   * Optionally, write the numeric issue count next to the dot.

5. **Output per-image artifacts**

   For each image in:

   * `dataset/faulty-ai-images/`
   * `dataset/correct-ai-images/`

   Create a corresponding annotated copy in:

   * `output/faulty-ai-images/`
   * `output/correct-ai-images/`

   plus a console summary like:

   ```text
   Image: horse-extra-legs.jpg
   Cell (row=1, col=2): issue_count=3, issues=['Too many legs', 'Anatomically impossible joints', 'Deformed hooves']
   ...
   ```

   This gives you a clean demo: **no extra dataset**, just your current folders, and you visually show where the model thinks problems live.

---

