# AI-Generated Image Error Detection Strategy Outline and Research

## Overview and Objectives

Analyzing AI-generated images for errors involves identifying visual anomalies and precisely marking their location. The goal is to flag mistakes (e.g. extra limbs, distorted text, unrealistic artifacts) and overlay a marker on the image indicating where the error is, along with a brief description. This must be done cost-efficiently using Python and possibly LangChain, leveraging available vision models or APIs. The project will handle all image types (portraits, landscapes, product shots, etc.), so the strategy must be robust to various content. A key challenge is that there are **no reference images** for comparison - the system must rely on the AI model's own knowledge of what is plausible (for example, a person should have two hands and each hand five fingers). Below, we outline a comprehensive strategy addressing these requirements.

## Key Challenges and Considerations

- **Variety of Image Types:** The system should handle **portraits, landscapes, and product images** (the answer to question 2 was "all of the above"). Different image types have different common errors:
- _Portraits:_ look for anatomical errors (extra/missing fingers, limbs, or facial distortions), unnatural textures, or overly "perfect" features[\[1\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf).
- _Landscapes:_ look for physical inconsistencies (impossible shadows, melting or blobby structures) and strange artifacts or repetitive patterns in nature.
- _Product Images:_ check for text/logo errors on packaging, warped geometry (e.g. a mug with a detached handle), or inconsistent reflections.
- **No Reference Images:** Since we cannot compare against a ground-truth image (question 3), the system must use **knowledge-based anomaly detection**. It should leverage common sense and learned norms of reality:
- For example, a human should have five fingers per hand - if an image shows a hand with six fingers, it's an anomaly[\[2\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf)[\[3\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf).
- Text in real images is usually readable - if an image has gibberish or blurred text on a sign or label, it's likely an AI error[\[4\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf).
- Physical laws should hold - e.g. a candle burning inside a fully sealed jar is implausible[\[5\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf), indicating an error.
- **Minimizing Cost:** The strategy should minimize API calls and heavy computations. This can be achieved by:
- Using local or open-source models (for vision processing) where possible instead of expensive cloud APIs.
- Applying efficient checks first (simple rules or lightweight detectors) to filter obvious errors before invoking large models.
- Optimizing image processing (e.g. resizing images to the needed resolution for analysis, focusing on regions of interest to save computation).

## Strategy for Error Detection and Analysis

To effectively detect errors, a **hybrid approach** can be used, combining specific computer vision tools with a large language model's reasoning. LangChain can orchestrate this pipeline, where the LLM can call vision processing tools and then reason about their outputs.

**1\. Initial Image Analysis (Content Detection):** First, determine what the image contains to decide which error-checks to run: - Use an image classification or tagging model to identify if the image has a **person**, **text**, **animals**, etc. - Alternatively, utilize an LLM with vision capabilities (e.g. GPT-4 Vision or an open-source multimodal model) to get a quick description of the image content. This description can hint at possible errors (for example, an LLM might note "the person appears to have an extra finger" in its description).

**2\. Rule-Based Anomaly Checks:** For known common errors, implement targeted detectors: - **Anatomical Anomalies:** If a person is present, check for extra or missing body parts. For instance: - Apply a **pose/hand keypoint detector** (like MediaPipe or OpenCV pose estimation) to count fingers and hands. A rule can flag if a hand has more than 5 fingertip keypoints or if a person has more than two arms. This addresses scenarios like extra fingers or limbs. - This approach is inspired by suggestions in the community to use object recognition or a compact model to detect wrong finger counts[\[6\]](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1821). For example, one idea is to train a small model specifically to recognize when a hand region has the wrong number of fingers[\[6\]](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1821). - **Facial Oddities:** Use face detection and optionally face-landmark models. They can verify if two eyes, one nose, one mouth are present and reasonably placed. Unusually arranged or duplicated facial features would be flagged. - **Text Anomalies:** If the image contains text (on signs, products, etc.), use **OCR (Optical Character Recognition)** to read it. - If OCR fails or returns garbled text that doesn't form words, that indicates the text is nonsensical - a known hallmark of AI-generated images[\[4\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf). The system should then flag those text regions as errors (e.g. "illegible text on sign"). - _Cost note:_ OCR can be done with open-source libraries (Tesseract or EasyOCR) to avoid API costs. - **General Visual Artifacts:** Implement some image processing checks for blatant glitches: - Check for areas that are abnormally **blurry or distorted** compared to the rest of the image. AI images often have localized blurs or "blob-like" artifacts[\[7\]](https://www.academia.edu/108091171/Detection_of_Images_generated_by_AI_implementing_attention_based_techniques). Simple methods like edge detectors or variance of Laplacian (for blur detection) can identify overly blurry regions. - Look for **incongruous shapes or disconnections**, e.g. an object that fades into the background or a limb that isn't properly attached. Some of these can be caught by segmentation masks or by an LLM's description. For example, an AI model with attention can focus on image parts with inconsistent texture or shape to detect anomalies[\[7\]](https://www.academia.edu/108091171/Detection_of_Images_generated_by_AI_implementing_attention_based_techniques). - **Common-Sense Violations:** For more subtle errors that defy real-world logic (like the candle-in-jar physics example), an LLM is useful. The LLM can be prompted with a description of the scene to reason about feasibility. While not every demo image will require this, it's good to note that advanced models (like GPT-4 or specialized systems) can spot these issues[\[5\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf).

**3\. Leveraging Vision-Language Models:** A multimodal LLM can serve as an "error describer" to complement the rule-based checks: - Models such as GPT-4 with Vision or open models (e.g. LLaVA, mPLUG-Owl, etc.) can take an image and output a description of anomalies. Research has shown that fine-tuned vision-language models can produce human-like explanations of AI image errors[\[3\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf)[\[1\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf). For instance, the AIGI-Holmes model (an ICCV 2025 work) outputs explanations like _"The hands have an incorrect number of fingers…suggesting AI generation"_[\[3\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf) or _"The woman's facial features are almost too perfect…often a sign of AI generation"_[\[1\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf). - **Strategy:** Use an LLM (via OpenRouter or an open alternative) to analyze the image after initial checks. You could feed it either the image directly (if the model supports vision input) or a compiled summary of what was detected (from the previous steps). For example, you might prompt: _"The system detected an unusual hand shape in this image. Describe the issue in detail."_ The LLM can then articulate the problem (e.g. "the person's left hand has six fingers, which is anatomically incorrect"). - Using LangChain, you can set up an agent that first calls detection tools (OCR, pose detector, etc.), and passes their findings into an LLM prompt. The LLM can then confirm the errors and add any others it infers. This **agent approach** ensures that the heavy vision lifting is done by specialized tools, and the LLM focuses on reasoning and explanation (conserving tokens and cost).

**4\. Integrating Findings:** Combine the outputs of rule-based detectors and the LLM's analysis to form a list of errors found. Each error entry should have: - A description (e.g. "Extra finger on right hand" or "Unreadable text on sign in background"). - The location (region in image) where it occurs (details on pinpointing location are below).

By using both targeted computer vision checks and an LLM for general reasoning, the system covers both obvious _and_ subtle anomalies. Notably, this approach aligns with recent research trends: explainable detection frameworks use **structured anomaly annotations** covering **anatomical errors, text rendering flaws, physical implausibilities, etc**[\[8\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf) as well as commonsense reasoning about the image. Our hybrid strategy is essentially implementing these ideas in a practical, cost-conscious way.

## Pinpointing Error Locations (Highlighting)

Once an error is identified, the next step is to mark its location on the image (e.g. draw a bounding box or highlight). Several techniques can achieve this:

- **Bounding Box via Detection Models:** If a specific detector was used:
- For a hand with extra fingers, a hand-detection model or pose landmarks give coordinates of the hand region. You can compute a bounding box around the hand keypoints and draw a rectangle or circle to highlight it.
- For OCR-detected text, most OCR libraries provide bounding boxes around detected text regions. Use those coordinates to highlight the area of gibberish text.
- If using an object detector for other anomalies (say a YOLO model that might detect "hand" or "face"), you can leverage its bounding box output.
- **Segmentation Approaches:** For more precision, segmentation models like **SAM (Segment Anything Model)** can be used. If you know a rough area (e.g. a point on the erroneous hand or region of weird texture), SAM can generate a mask for that object/area, which you can then overlay as a translucent highlight. This could be triggered by either user input or automatically choosing a point in the region of interest (perhaps via the attention map or keypoints).
- **Attention/Saliency Maps:** If you trained or use a classifier to detect AI vs real images, you can apply **Grad-CAM or attention visualization** to see what parts of the image the model considered "fake-like." This often highlights the anomaly. For example, an attention-based fake image detector can naturally focus on blobs, blurs, or incongruous shapes[\[7\]](https://www.academia.edu/108091171/Detection_of_Images_generated_by_AI_implementing_attention_based_techniques). You could generate a heatmap and threshold it to mark regions. In a demo, this can be an insightful visualization (though it may highlight multiple areas rather than giving a neat box).
- **LLM Guidance to Localization:** If the LLM describes the location in words ("on the person's left hand" or "center of the tablet screen"), you can interpret that:
- Use a body part detector to find "left hand of person" in coordinates.
- Or use template matching: for instance, find the largest object that looks like a hand on the left side of the image.
- While not pixel-perfect, this can place an approximate marker. In a demo context, even a rough arrow pointing to the described area can suffice if exact segmentation is too involved.

In summary, for each detected error, we will have either direct coordinates from detectors or we derive them via additional processing. We then **draw markers** on the image: e.g. red bounding boxes or circles around the error region, perhaps with a label number. The description of the error can be listed in text alongside or below the image (with a legend like "1 - Extra finger on left hand").

_Important:_ Avoid cluttering the image with too many markers; highlight only the key mistake regions. If an image has no detected issues, the output for that image would simply state "No obvious AI-generation errors detected" and show the original image with no markers.

## Cost Minimization Techniques

To keep the solution cost-effective, consider the following optimizations:

- **Use Local/Open-Source Models:** Wherever possible, use local models. For example:
- Use **MediaPipe** or **OpenCV** for pose and face detection (these run locally and are free).
- Use **Tesseract OCR** for text (free offline).
- Use an open-source **Vision-Language model** for analysis if GPT-4's Vision API is costly. Models like _LLaVA_ or _mPLUG-Owl_ (7B variants) can run on a GPU. They might not be as fluent as GPT-4, but fine-tuned versions have shown decent anomaly explanation capabilities[\[9\]](https://arxiv.org/html/2510.10231v1)[\[10\]](https://arxiv.org/html/2510.10231v1).
- If using OpenRouter with a model like GPT-4, use it sparingly (e.g. only on images where simpler checks found something or as a final step to verify and explain).
- **Batch Processing & Caching:** If analyzing many images, process them in batch where possible. Also cache results of heavy computations:
- For example, if using a large model to get image features or descriptions, cache that so you don't recompute on the same image during development.
- **Resolution Trade-offs:** Use a reasonable image size for analysis (not too high resolution which slows models, and not too low that details are lost). You can downscale large images to, say, 512px or 720px on the long side for processing, which is often sufficient for detecting anomalies like extra fingers or blurry text.
- **Progressive Analysis:** Adopt a cascade: quick cheap checks first, then expensive ones:
- **Quick pass:** e.g. run OCR and pose detector (fast) - if they already flag errors, you might not need to call the LLM except to generate the explanation text.
- If quick pass finds nothing obvious, then use the LLM to do a more thorough analysis of the image (so you pay for the LLM only on the harder cases).
- **Limit LLM Token Usage:** If using LangChain with an LLM, craft concise prompts and instruct it to be brief in its explanation. Also, you might give the LLM structured context (like "Flag = 1 if error present in category X") rather than raw image data to reduce how much it needs to process. The structured approach of AIGI-Holmes, for example, uses a **structured output (Name, Phenomenon, Reasoning, Severity)** for anomalies[\[11\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf), which our system can simplify to just a name and short description to keep the conversation short.

By combining these strategies, we ensure the pipeline runs efficiently. The heavy lifting of image analysis is done by specialized tools (which are faster and cheaper than a general LLM), and the LLM is used mainly for its strength in reasoning and language (which we use in a limited, targeted manner).

## Benchmarking and Evaluation

Since this is a project/demonstration, we should still define some way to evaluate if the approach works (question 4 touches on benchmarks). Here are some strategies: - **Create a Test Set of Labeled Images:** Use a collection of images with known errors and some without errors. You might source some "bad" AI-generated images (e.g. from Stable Diffusion or Midjourney fail cases posted online, or the MalHand dataset for hand anomalies[\[12\]](https://arxiv.org/html/2411.04332v1)[\[13\]](https://arxiv.org/html/2411.04332v1)) and label the errors they contain. Also include some real or well-generated images that should pass as having "no error." - **Measure Detection Accuracy:** Run your pipeline on this test set and see: - How many known errors it successfully catches (true positives). - How many false alarms it raises on images that are actually fine (false positives). - This can be summarized in precision/recall terms if you have enough samples. For example, if 10 images have extra fingers and your system flagged 9 of them, that's 90% recall for that error type. - **Quality of Explanations:** Because this project emphasizes explaining the error, you can evaluate the explanations qualitatively: - Are the descriptions factually correct and pinpoint the issue? (e.g. it correctly says "six fingers on one hand" vs. a vague "the hand looks odd"). - One way to benchmark explanation quality is by comparison to a ground-truth explanation. In research, metrics like BLEU or METEOR have been used to compare model explanations to reference text[\[14\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf)[\[15\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf). For our demo, simply ensure the explanations are **understandable and verifiable by a human** - i.e., someone looking at the image can clearly see the described issue. - **Use of Attention Maps (Optional):** If we incorporate a classifier with attention or Grad-CAM, we can use this as a form of evaluation/explanation. For each "fake" image in the test set, check if the classifier's attention highlights the error region (this validates that our system is focusing on the right cues). For example, if an image has an extra arm and the attention map lights up around that arm, that confirms our system's detection mechanism aligns with the known anomaly[\[7\]](https://www.academia.edu/108091171/Detection_of_Images_generated_by_AI_implementing_attention_based_techniques). - **Benchmark Against Past Work (if feasible):** If data is available, you could see how your approach compares to known benchmarks. For instance, AIGI-Holmes dataset (Holmes-Set) contains images with annotated anomalies[\[8\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf). Without doing a full ML training, you might simply test a few images from such datasets to see if your pipeline finds the same issues noted in the annotations.

For a demo, a formal benchmark isn't absolutely required, but having a **small curated evaluation** helps validate the approach. You can present a table of results like "Out of X images with errors, the system caught Y% of them" and maybe show a couple of failure cases to analyze why something was missed (this might lead to improving the rules or prompts).

## Demo Implementation Plan

Finally, to demonstrate this project, here's how you could set up the demo:

- **Prepare Demo Images:** Gather a set of images, some with obvious AI mistakes and some that are clean. For example:
- Portrait of a person with an extra finger or strange hands.
- A landscape with a warped building or unnatural artifact.
- A product photo with scrambled text on the label.
- A few _normal_ images (real or very well-generated) for contrast. Ensure each image is labeled behind-the-scenes with what the "issue" is (for your reference during the demo).
- **Develop the Pipeline (Python):** Write a script or notebook that:
- **Loads an image** and displays it.
- Runs the detection checks:
  - Pose/hand analysis, face analysis, OCR, etc.
  - If using LangChain, instantiate an agent that can call these tools. For example, define tools like detect_hands(image) -> regions or read_text(image) -> text. The LangChain agent can decide to use these and then call the LLM for interpretation.
  - Alternatively, run the tools sequentially in code and compile their findings into a simple list (e.g. issues = \["Hand: 6 fingers detected on left hand", "Text: OCR failed on sign text"\]).
- If issues were found by the tools, optionally verify or elaborate with an LLM:
  - You might prompt: _"Image analysis found: 1) extra finger on left hand; 2) unreadable text on sign. Explain these issues briefly."_ The LLM would then produce a nice concise explanation confirming those points.
  - If no issues were found by tools, you could still query the LLM: _"Do you notice any oddities in this image?"_ as a safety net, but instruct it to answer "No obvious issues" if none come to mind.
- **Draw on the image:** Using OpenCV or PIL, draw circles or boxes at the locations of each issue:
  - Use the coordinates from the detection stage. For example, if the hand keypoint detector found an extra finger, you have the hand's coordinates to highlight[\[6\]](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1821). If OCR flagged text, you have the text bounding box.
  - Make the markers clear (red or yellow with some transparency perhaps) and not too small.
- **Display the result:** Show the annotated image next to or below the original, with a caption or list of error descriptions. Each description can be numbered to match the markers on the image.
- **User Interface for Demo:** Depending on context, this could be:
- A **Jupyter Notebook** where you step through images one by one, showing the outputs. You can use Markdown cells to narrate and the notebook to show images with Matplotlib or PIL.
- A **Streamlit or Gradio app** where the user can click through examples (or even upload their own image) and see the analysis. This makes it interactive - LangChain can be integrated in a backend to handle the analysis when a new image is uploaded.
- Slides or a report with before/after images: If live demo is risky, prepare slides showing each image and the detected annotations, alongside your explanation text. For instance, slide 1: original vs highlighted image with bullet points of detected errors.
- **Demonstrate Various Cases:** In the demo, show that the system works on all requested types:
- **Portrait example:** highlight an anatomical error (e.g., extra finger). Explain: "The system marked this hand because it found six fingers[\[3\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf)."
- **Landscape example:** perhaps highlight a weird object or a physics defying element (like a suspiciously blurred shape in the sky). Explain how the model knows it's unusual (maybe "this building's reflection is missing, indicating a possible generation flaw" - if such an error was present).
- **Product image example:** highlight the text area that is gibberish. The demo can show the OCR output vs what a real product label should look like, and the system's flagged region on the label with a note "text is not readable"[\[4\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf).
- **No-error example:** show a normal image and that the system correctly reports no issues. This proves it's not flagging false positives on everything.
- **Discuss Performance (if relevant):** If you did the mini-benchmark, you can conclude the demo by mentioning how many errors were caught in testing and how the approach can be improved. For instance, maybe note if the system missed anything (and why), or how using a more powerful model could help catch subtler anomalies.

Throughout the demo, emphasize the **explanation** part. This is a key differentiator - not only does the system detect an error, but it describes it. This builds trust with users, since they can see **why** the system thinks it's an error (similar to providing human-verifiable explanations in research[\[16\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf)[\[17\]](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf)).

## Citations

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

Too many fingers - it maybe can be solved with a bit of a performance loss ?  AUTOMATIC1111 stable-diffusion-webui  Discussion #1821  GitHub

https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1821

(PDF) Detection of Images generated by AI implementing attention based techniques

https://www.academia.edu/108091171/Detection_of_Images_generated_by_AI_implementing_attention_based_techniques

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images

https://arxiv.org/html/2510.10231v1

Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images

https://arxiv.org/html/2510.10231v1

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models

https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_AIGI-Holmes_Towards_Explainable_and_Generalizable_AI-Generated_Image_Detection_via_Multimodal_ICCV_2025_paper.pdf

HandCraft: Anatomically Correct Restoration of Malformed Hands in Diffusion Generated Images

https://arxiv.org/html/2411.04332v1

HandCraft: Anatomically Correct Restoration of Malformed Hands in Diffusion Generated Images

https://arxiv.org/html/2411.04332v1

HandCraft: Anatomically Correct Restoration of Malformed Hands in Diffusion Generated Images

https://arxiv.org/html/2411.04332v1

All Sources

openaccess.thecvf
github
academia
arxiv

