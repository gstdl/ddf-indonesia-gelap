"""Box: Image: deepfake classifier. Takes the "Image" arrow of "Content type?".

Output: one row per image post (media_type == "image") that is an original or a quote
(reposts copy their parent; replies are judged by classify_replies). Columns:
    post_id
    image_ai_probability_community_forensics   AI probability of the image (data/<media_path>) by community_forensics
    image_ai_probability_openfake              same, by openfake
    image_ai_probability_mkk_convnext          same, by mkk_convnext
    media_signal                               quorum() of the three: "synthetic", "real" or "disagree"
Models (ViT, Swin and a CNN, so they do not fail together):
    community_forensics  ViT-S/16 384px (CLIP-normalised), Park & Owens "Community Forensics" (CVPR 2025,
                         arXiv:2411.04125): trained on images of 4,803 generators (mostly Hugging Face Hub diffusion
                         models); its one logit -> sigmoid is the AI probability.
    openfake             SwinV2-base 256px, the OpenFake Demo detector (Livernoche et al. 2025, arXiv:2509.09495): trained
                         on OpenFake (real photos incl. LAION and Pexels vs. many proprietary and open-source diffusion
                         generators and Civitai fine-tunes; political, news and meme content). Licence CC-BY-NC-4.0
                         (research only). Known to flag real born-digital text graphics (screenshots, chyrons, memes)
                         as fake. The Demo divides logits by 2.0 before the softmax; we use plain softmax, which gives
                         the same 0.5 vote.
    mkk_convnext         ConvNeXt-Tiny CNN at the image's full resolution (no resize), mkk1135 graduation project, no
                         paper: trained on Community Forensics-Small (377 Stable Diffusion fine-tunes, GLIDE, DeepFloyd
                         IF, BigGAN; real COCO, FFHQ, ...) plus DataSeeds photographs; 81.5% on its held-out test set;
                         mostly misses generated images, weak on GPT Image / Nano Banana. Licence CC-BY-NC-SA-4.0.
image_ai_probabilities() is also the frame scorer of video_detector.
"""

from functools import cache

import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from pipeline.hf import DEVICE
from pipeline.layer2_label.track2_ai_generated.quorum import quorum
from pipeline.step import DATA, progress

# (short_name, model_id, revision, AI label in the model's config id2label). A one-logit model (sigmoid) has the
# default label "LABEL_0"; a multi-class model is read with softmax at its AI label.
IMAGE_MODELS = [
    (
        "community_forensics",
        "buildborderless/CommunityForensics-DeepfakeDet-ViT",
        "ac6ee457bea904a373065754107451793b56db00",
        "LABEL_0",
    ),
    ("openfake", "bendavidsteel/openfake-swinv2-base", "90837e19c45b9a49af435d35071114ba1448c239", "fake"),
    ("mkk_convnext", "mkk1135/ai-image-detector", "fb3e159796efe274f1139bd5cc3c967bc96735ce", "ai"),
]
# Post types that carry their own media: reposts copy the parent, replies go to classify_replies.
OWN_CONTENT_POST_TYPES = ["original", "quote"]


@cache
def image_model(model_id: str, revision: str):
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageClassification.from_pretrained(model_id, revision=revision)
    return processor, model.to(DEVICE).eval()


def image_ai_probabilities(path: str) -> dict[str, float]:
    """AI probability of one image file (path relative to data/) per model short_name."""
    with Image.open(DATA / path) as image:
        image = image.convert("RGB")
    probabilities = {}
    for short_name, model_id, revision, ai_label in IMAGE_MODELS:
        processor, model = image_model(model_id, revision)
        inputs = processor(image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        scores = torch.sigmoid(logits) if len(logits) == 1 else torch.softmax(logits, dim=-1)
        probabilities[short_name] = scores[model.config.label2id[ai_label]].item()
    return probabilities


def run(posts: pd.DataFrame) -> pd.DataFrame:
    images = posts[(posts["media_type"] == "image") & posts["post_type"].isin(OWN_CONTENT_POST_TYPES)]
    probabilities = pd.DataFrame(
        [image_ai_probabilities(path) for path in progress(images["media_path"].to_list(), "image detectors")],
        columns=[short_name for short_name, *_ in IMAGE_MODELS],
        dtype="float64",
    )
    table = probabilities.add_prefix("image_ai_probability_")
    table.insert(0, "post_id", images["post_id"].to_list())
    table["media_signal"] = [quorum(row) for row in probabilities.to_numpy().tolist()]
    return table.sort_values("post_id", ignore_index=True)
