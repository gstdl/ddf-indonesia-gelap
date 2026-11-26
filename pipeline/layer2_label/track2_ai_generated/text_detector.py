"""Box: Text: SynthID + AI-text detector. Takes the "Text" arrow of "Content type?".

Output: one row per text post (media_type == "text") that is an original or a quote
(reposts copy their parent; replies are judged by classify_replies). Columns:
    post_id
    text_ai_probability_multilingual_xlmr        probability that the post `text` was written by an AI model, by
                                                 multilingual_xlmr
    text_ai_probability_indobert                 same, by indobert
    text_ai_probability_german_academic_mmbert   same, by german_academic_mmbert
    media_signal                                 quorum() of the three: "synthetic", "real" or "disagree"
Models (XLM-RoBERTa, BERT and ModernBERT, so they do not fail together); AI probability = 1 - P(human label):
    multilingual_xlmr        XLM-RoBERTa-base (bibbbu), no paper: fine-tuned on only 900 English, Chinese and
                             Vietnamese QA pairs with AI answers from one generator (Qwen2.5-1.5B-Instruct); on
                             Indonesian it relies on cross-lingual transfer. Licence MIT.
    indobert                 IndoBERT-base-uncased (IndoLEM, Koto et al. 2020, arXiv:2011.00677; BERT) fine-tuned on
                             Indonesian human vs AI text (PlatinaSB), no paper; training data undocumented (92.5%
                             accuracy on its own held-out set). The only Indonesian-trained detector found; flags long
                             or list-like LLM prose but misses short LLM sentences. Licence MIT.
    german_academic_mmbert   mmBERT-small (Marone et al. 2025, arXiv:2509.06888; ModernBERT pretrained on 1,800+
                             languages incl. Indonesian) fine-tuned on German academic text (aliissa824), no paper:
                             classes human / ai_generated / ai_assisted (human text polished by AI), so its AI
                             probability counts both AI classes; on Indonesian it relies on cross-lingual transfer.
                             Licence CC-BY-SA-4.0.

SynthID is not built: detecting a SynthID text watermark needs the private watermarking keys of the model
that generated the text, so only Google can check text for a Gemini watermark.
Limitation: AI-text detectors are trained on long, formal text, and only indobert saw Indonesian; tweets are short and
casual, so each probability is weak evidence. English-only and English-trained detectors were tried and rejected
because they do not transfer to Indonesian: fakespot-ai/roberta-base-ai-text-detection-v1 and
GeorgeDrayson/modernbert-ai-detection-raid-mage scored casual Indonesian tweets 0.70-0.95 AI,
tabularisai/ai-text-detection scored every Indonesian text ~0.98 AI, and rasbt/ai-text-detector-qwen3-0.6b-variable,
andreas122001/bloomz-560m-mixed-detector, AICodexLab/answerdotai-ModernBERT-base-ai-detector and
almanach/camemberta-chatgptdetect-noisy scored every Indonesian LLM-style text below 0.05. The multilingual
trokhymovych/mbert-ai-bot-detector (social-bot detector) scored casual Indonesian tweets 0.48-0.94 AI and
hafsteinn/skemman-detector-ws-en-mdeberta-v3-base stayed near 0.5 on every Indonesian text.
"""

from functools import cache

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline.hf import DEVICE
from pipeline.layer2_label.track2_ai_generated.quorum import quorum

# (short_name, model_id, revision, human label in the model's config id2label). The AI probability is 1 - softmax at
# the human label, which is P(AI) for the binary models and P(ai_generated) + P(ai_assisted) for the 3-class mmBERT.
TEXT_MODELS = [
    # The config has no label names; the model card names 0 "human", 1 "ai".
    (
        "multilingual_xlmr",
        "bibbbu/multilingual-ai-human-detector_xlm-roberta-base",
        "b4aa2e06d8428b4d27f2ef625267044d74b2eeb8",
        "LABEL_0",
    ),
    ("indobert", "PlatinaSB/indobert-ai-text-classification", "d296ab63ce4a7603a53061e1bbea3b03391c9c9d", "Human"),
    (
        "german_academic_mmbert",
        "aliissa824/german-academic-ai-text-mmbert-small",
        "2084d948a3ad79cfdcc4c29508274e7196582483",
        "human",
    ),
]
# Tokens read per post: the shortest fine-tuning length of the three (multilingual_xlmr), far above a tweet's length.
MAX_TOKENS = 256
# Post types that carry their own content: reposts copy the parent, replies go to classify_replies.
OWN_CONTENT_POST_TYPES = ["original", "quote"]
# Posts scored per forward pass.
BATCH_SIZE = 32


@cache
def text_model(model_id: str, revision: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
    return tokenizer, model.to(DEVICE).eval()


def text_ai_probabilities(texts: list[str], model_id: str, revision: str, human_label: str) -> list[float]:
    tokenizer, model = text_model(model_id, revision)
    human_index = model.config.label2id[human_label]
    probabilities = []
    for start in range(0, len(texts), BATCH_SIZE):
        inputs = tokenizer(
            texts[start : start + BATCH_SIZE], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            probabilities += (1 - model(**inputs).logits.softmax(-1)[:, human_index]).tolist()
    return probabilities


def run(posts: pd.DataFrame) -> pd.DataFrame:
    texts = posts[(posts["media_type"] == "text") & posts["post_type"].isin(OWN_CONTENT_POST_TYPES)]
    probabilities = pd.DataFrame(
        {
            short_name: text_ai_probabilities(texts["text"].to_list(), model_id, revision, human_label)
            for short_name, model_id, revision, human_label in TEXT_MODELS
        },
        dtype="float64",
    )
    table = probabilities.add_prefix("text_ai_probability_")
    table.insert(0, "post_id", texts["post_id"].to_list())
    table["media_signal"] = [quorum(row) for row in probabilities.to_numpy().tolist()]
    return table.sort_values("post_id", ignore_index=True)
