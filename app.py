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
        "messages": messages,
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

    inputs = processor.apply_chat_template(**template_kwargs).to(model.device)

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
.gradio-container { max-width: 1180px !important; }
.header { margin-bottom: 14px; }
.settings-panel { border-left: 1px solid var(--border-color-primary); padding-left: 16px; }
"""


with gr.Blocks(
    title="Recipe Lens Chef",
    css=CSS,
    fill_height=True,
) as demo:
    gr.Markdown(
        "# Recipe Lens Chef\n"
        "Upload a food image or cooking video, ask what to make, and run inference with "
        f"`{MODEL_ID}`.",
        elem_classes=["header"],
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=360):
            image_input = gr.Image(
                label="Image",
                type="filepath",
                sources=["upload", "clipboard", "webcam"],
                height=340,
            )
            video_input = gr.Video(
                label="Video",
                sources=["upload", "webcam"],
                height=260,
            )
            prompt_input = gr.Textbox(
                label="Prompt",
                value=DEFAULT_PROMPT,
                lines=4,
                max_lines=8,
            )
            run_button = gr.Button("Run", variant="primary")

        with gr.Column(scale=4, min_width=320):
            output_text = gr.Markdown(label="Response", container=True)

        with gr.Column(scale=2, min_width=260, elem_classes=["settings-panel"]):
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
    demo.queue(default_concurrency_limit=1).launch()
