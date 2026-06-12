from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

import gradio as gr


MODEL_ID = os.getenv("MODEL_ID", "openbmb/MiniCPM-V-4.6")
DEFAULT_PROMPT = (
    "Look at this food image and suggest a practical chef-style recipe. "
    "Include likely ingredients, prep steps, cooking method, timing, and tips."
)

_ESCAPED_NEWLINE_PATTERN = re.compile(
    r"(```[\s\S]*?```|`[^`]+`|\$\$[\s\S]*?\$\$|\$[^$]+\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\])"
    r"|(?<!\\)(?:\\r\\n|\\[nr])"
)


def normalize_response_text(text: str) -> str:
    if not isinstance(text, str) or "\\" not in text:
        return text
    return _ESCAPED_NEWLINE_PATTERN.sub(lambda match: match.group(1) or "\n", text)


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return processor, model


def _build_message(media_kind: str, media_path: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": media_kind, "url": media_path},
                {"type": "text", "text": prompt.strip() or DEFAULT_PROMPT},
            ],
        }
    ]


def generate_response(
    image_path: str | None,
    video_path: str | None,
    prompt: str,
    downsample_mode: str,
    max_new_tokens: int,
    max_slice_nums: int,
    max_num_frames: int,
    stack_frames: int,
) -> str:
    import torch

    if image_path and video_path:
        raise gr.Error("Use either an image or a video for one request.")
    if not image_path and not video_path:
        raise gr.Error("Upload an image or video first.")

    is_video = bool(video_path)
    media_path = video_path or image_path
    media_kind = "video" if is_video else "image"
    messages = _build_message(media_kind, media_path, prompt)

    processor, model = load_model()

    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "downsample_mode": downsample_mode,
        "max_slice_nums": 1 if is_video else max_slice_nums,
    }
    if is_video:
        template_kwargs.update(
            {
                "max_num_frames": max_num_frames,
                "stack_frames": stack_frames,
                "use_image_id": False,
            }
        )

    inputs = processor.apply_chat_template(messages, **template_kwargs).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            downsample_mode=downsample_mode,
            max_new_tokens=max_new_tokens,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return normalize_response_text(output_text).strip()


CSS = """
:root {
  --paper: #f4eee1;
  --paper-2: #efe7d4;
  --kraft: #e4d5b7;
  --kraft-deep: #d6c19a;
  --line: #cdbb95;
  --ink: #33312b;
  --ink-2: #4a4639;
  --ink-soft: rgba(51, 49, 43, 0.68);
  --forest: #3d6a55;
  --forest-deep: #2b4d3d;
  --amber: #e0913a;
  --clay: #bd5f37;
  --press: 5px 5px 0 var(--ink);
  --press-sm: 3px 3px 0 var(--ink);
}

.gradio-container {
  max-width: 1220px !important;
  margin: 0 auto !important;
  background:
    linear-gradient(90deg, rgba(51, 49, 43, 0.035) 1px, transparent 1px),
    linear-gradient(180deg, rgba(51, 49, 43, 0.025) 1px, transparent 1px),
    var(--paper) !important;
  background-size: 42px 42px, 42px 42px, auto !important;
  color: var(--ink) !important;
  font-family: Archivo, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  padding: 22px !important;
}

.gradio-container::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: 0.48;
  background-image: radial-gradient(rgba(51, 49, 43, 0.13) 0.65px, transparent 0.65px);
  background-size: 8px 8px;
}

.gradio-container > .main,
.gradio-container .contain {
  position: relative;
  z-index: 1;
}

.hero-card {
  position: relative;
  overflow: hidden;
  min-height: 280px;
  margin: 0 0 22px;
  padding: clamp(24px, 5vw, 46px);
  background: var(--paper-2);
  border: 2px solid var(--ink);
  box-shadow: var(--press);
}

.hero-card::before {
  content: "";
  position: absolute;
  inset: -12% -6%;
  opacity: 0.18;
  background:
    repeating-radial-gradient(ellipse at 78% 42%, transparent 0 24px, var(--line) 25px 27px, transparent 28px 46px),
    repeating-linear-gradient(135deg, transparent 0 18px, rgba(205, 187, 149, 0.48) 19px 21px);
}

.hero-content {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 24px;
  align-items: end;
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
  color: var(--ink-soft);
  font-family: "Spline Sans Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.eyebrow::before {
  content: "";
  width: 24px;
  height: 2px;
  background: var(--amber);
}

.hero-title {
  margin: 0;
  max-width: 780px;
  color: var(--ink);
  font-size: clamp(48px, 11vw, 126px);
  font-weight: 900;
  line-height: 0.9;
  letter-spacing: 0;
  text-transform: uppercase;
}

.hero-copy {
  max-width: 680px;
  margin: 18px 0 0;
  color: var(--ink-2);
  font-size: clamp(17px, 2vw, 22px);
  line-height: 1.3;
}

.hero-statline {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 24px;
}

.trail-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 34px;
  padding: 7px 12px;
  background: var(--paper);
  border: 2px solid var(--ink);
  box-shadow: var(--press-sm);
  color: var(--ink);
  font-family: "Spline Sans Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.stamp-mark {
  width: clamp(96px, 16vw, 150px);
  aspect-ratio: 1;
  display: grid;
  place-items: center;
  border: 2px dashed var(--paper);
  outline: 2px solid var(--ink);
  border-radius: 999px;
  background: var(--forest);
  color: var(--paper);
  box-shadow: var(--press);
  font-family: "Spline Sans Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-weight: 900;
  text-align: center;
}

.stamp-mark span {
  display: block;
  padding-top: 2px;
  font-size: clamp(22px, 4vw, 38px);
  line-height: 0.95;
}

.workbench-grid {
  align-items: stretch;
}

.field-panel {
  background: color-mix(in srgb, var(--paper) 92%, white);
  border: 2px solid var(--ink);
  box-shadow: var(--press);
  padding: 16px;
}

.output-panel {
  background: var(--kraft);
}

.settings-panel {
  background: var(--paper-2);
}

.gradio-container label,
.gradio-container .block-title,
.gradio-container .accordion-label {
  color: var(--ink) !important;
  font-family: "Spline Sans Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
  font-size: 12px !important;
  font-weight: 800 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container textarea,
.gradio-container input,
.gradio-container select {
  border-radius: 0 !important;
}

.gradio-container .block {
  border-color: var(--ink) !important;
}

.gradio-container textarea,
.gradio-container input {
  background: #fffaf0 !important;
  color: var(--ink) !important;
}

.gradio-container button.primary {
  min-height: 48px !important;
  border: 2px solid var(--ink) !important;
  border-radius: 0 !important;
  background: var(--forest) !important;
  color: #fff !important;
  box-shadow: var(--press-sm) !important;
  font-weight: 900 !important;
  letter-spacing: 0.02em !important;
  transition: transform 0.1s ease, box-shadow 0.1s ease !important;
}

.gradio-container button.primary:hover {
  transform: translate(2px, 2px);
  box-shadow: 1px 1px 0 var(--ink) !important;
}

.gradio-container .tabs {
  border-radius: 0 !important;
}

.gradio-container .tabitem {
  border-color: var(--ink) !important;
}

.gradio-container .wrap.svelte-1ipelgc,
.gradio-container .wrap {
  gap: 18px !important;
}

.response-title {
  margin: 0 0 10px;
  font-family: "Spline Sans Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-soft);
}

.examples {
  margin-top: 18px;
}

@media (max-width: 760px) {
  .gradio-container {
    padding: 14px !important;
  }

  .hero-content {
    grid-template-columns: 1fr;
  }

  .stamp-mark {
    width: 96px;
  }
}
"""

HERO_HTML = f"""
<section class="hero-card" aria-label="Recipe Lens Chef">
  <div class="hero-content">
    <div>
      <div class="eyebrow">Hugging Face x Gradio field notes</div>
      <h1 class="hero-title">Recipe Lens Chef</h1>
      <p class="hero-copy">
        Point the lens at a dish, pantry pile, or cooking clip. MiniCPM-V 4.6 turns the scene into
        a practical recipe, ingredient read, and step-by-step kitchen plan.
      </p>
      <div class="hero-statline" aria-label="App details">
        <span class="trail-chip">VLM recipe scout</span>
        <span class="trail-chip">{MODEL_ID}</span>
        <span class="trail-chip">Image + video</span>
      </div>
    </div>
    <div class="stamp-mark" aria-hidden="true"><span>RLC</span></div>
  </div>
</section>
"""


with gr.Blocks(
    title="Recipe Lens Chef",
    fill_height=True,
) as demo:
    gr.HTML(HERO_HTML)

    with gr.Row(equal_height=False, elem_classes=["workbench-grid"]):
        with gr.Column(scale=5, min_width=360, elem_classes=["field-panel"]):
            with gr.Tabs():
                with gr.Tab("Food photo"):
                    image_input = gr.Image(
                        label="Food snapshot",
                        type="filepath",
                        sources=["upload", "clipboard", "webcam"],
                        height=380,
                    )
                with gr.Tab("Cooking video"):
                    video_input = gr.Video(
                        label="Cooking clip",
                        sources=["upload", "webcam"],
                        height=300,
                    )
            prompt_input = gr.Textbox(
                label="Kitchen question",
                value=DEFAULT_PROMPT,
                lines=4,
                max_lines=8,
            )
            run_button = gr.Button("Cook this up", variant="primary")

        with gr.Column(scale=4, min_width=320, elem_classes=["field-panel", "output-panel"]):
            gr.HTML('<p class="response-title">Recipe field notes</p>')
            output_text = gr.Markdown(label="Chef plan", container=True)

        with gr.Column(scale=2, min_width=260, elem_classes=["field-panel", "settings-panel"]):
            downsample_input = gr.Radio(
                choices=["16x", "4x"],
                value="16x",
                label="Visual token compression",
                info="16x is faster; 4x keeps more visual detail.",
            )
            max_tokens_input = gr.Slider(
                minimum=64,
                maximum=2048,
                value=512,
                step=64,
                label="Max new tokens",
            )
            max_slice_input = gr.Slider(
                minimum=1,
                maximum=36,
                value=36,
                step=1,
                label="Image max slices",
            )
            max_frames_input = gr.Slider(
                minimum=8,
                maximum=128,
                value=64,
                step=8,
                label="Video max frames",
            )
            stack_frames_input = gr.Slider(
                minimum=1,
                maximum=5,
                value=1,
                step=1,
                label="Video stack frames",
            )

    gr.Examples(
        examples=[
            [None, None, "Create a recipe from this dish and explain how to cook it.", "16x", 512, 36, 64, 1],
            [None, None, "Identify the ingredients you can see and suggest a weeknight dinner recipe.", "4x", 512, 36, 64, 1],
            [None, None, "Watch the cooking steps and summarize the recipe timeline.", "16x", 1024, 36, 128, 1],
        ],
        inputs=[
            image_input,
            video_input,
            prompt_input,
            downsample_input,
            max_tokens_input,
            max_slice_input,
            max_frames_input,
            stack_frames_input,
        ],
    )

    run_button.click(
        fn=generate_response,
        inputs=[
            image_input,
            video_input,
            prompt_input,
            downsample_input,
            max_tokens_input,
            max_slice_input,
            max_frames_input,
            stack_frames_input,
        ],
        outputs=output_text,
        api_name="generate",
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
