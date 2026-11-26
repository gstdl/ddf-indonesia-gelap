"""Box: Image: pHash, OCR, caption. Takes the "Image" arrow of "Media type?".

Output: image posts (media_type == "image") plus
    phash              perceptual hash of the image
    media_ocr          Florence-2 <OCR> text, NaN when empty
    media_caption      Florence-2 <CAPTION> text, NaN when empty
    media_transcript   NaN (images have no audio)
    media_text         non-empty OCR and caption joined with "\n", NaN when both are empty
    keyframe_paths     NaN
    audio_path         NaN
"""

from functools import cache

import imagehash
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from pipeline.hf import DEVICE
from pipeline.step import DATA, progress

# Florence-2 in native transformers format (no trust_remote_code), shared with the video box.
FLORENCE_MODEL = "florence-community/Florence-2-large"
FLORENCE_REVISION = "4271c66b88cdbc05735372ec13b2360108de5317"
# Generation budget for one OCR or caption answer.
FLORENCE_MAX_NEW_TOKENS = 1024
# Beam search width (deterministic decoding).
FLORENCE_NUM_BEAMS = 3


@cache
def florence() -> tuple[AutoProcessor, Florence2ForConditionalGeneration]:
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, revision=FLORENCE_REVISION)
    model = Florence2ForConditionalGeneration.from_pretrained(FLORENCE_MODEL, revision=FLORENCE_REVISION)
    return processor, model.to(DEVICE).eval()


def describe(image: Image.Image, task: str) -> str:
    """Run one Florence-2 task ("<OCR>" or "<CAPTION>") on an image and return its text answer."""
    processor, model = florence()
    inputs = processor(text=task, images=image.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=FLORENCE_MAX_NEW_TOKENS, num_beams=FLORENCE_NUM_BEAMS, do_sample=False)
    answer = processor.batch_decode(ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(answer, task=task, image_size=image.size)[task].strip()


def join_texts(*texts: str) -> str | None:
    """Non-empty texts joined with newlines; None when every text is empty or missing."""
    present = [text for text in texts if pd.notna(text) and text != ""]
    return "\n".join(present) if present else None


def run(posts: pd.DataFrame) -> pd.DataFrame:
    images = posts[posts["media_type"] == "image"]
    opened = [Image.open(DATA / path) for path in images["media_path"]]
    ocr = [describe(image, "<OCR>") or None for image in progress(opened, "image OCR")]
    caption = [describe(image, "<CAPTION>") or None for image in progress(opened, "image caption")]
    return images.assign(
        phash=[str(imagehash.phash(image)) for image in opened],
        media_ocr=ocr,
        media_caption=caption,
        media_transcript=None,
        media_text=[join_texts(*texts) for texts in zip(ocr, caption)],
        keyframe_paths=None,
        audio_path=None,
    )
