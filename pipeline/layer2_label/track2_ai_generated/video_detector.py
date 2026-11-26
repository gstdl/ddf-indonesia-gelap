"""Box: Video: video / audio deepfake models. Takes the "Video" arrow of "Content type?".

Output: one row per video post (media_type == "video") that is an original or a quote
(reposts copy their parent; replies are judged by classify_replies). Columns:
    post_id
    frame_ai_probability_<image short name>   one per image_detector.IMAGE_MODELS model: its mean AI probability
                                              over the video's keyframes (layer 1 keyframe_paths)
    audio_ai_probability_<audio short name>   one per AUDIO_MODELS model: its mean deepfake-speech probability over
                                              equal windows of the audio track (layer 1 audio_path, 16 kHz mono wav);
                                              NaN when the video has no audio track
    frame_signal   quorum() of the frame probabilities: "synthetic" | "real" | "disagree"
    audio_signal   quorum() of the audio probabilities; "disagree" when there is no audio track (NaN does not vote)
    media_signal   video_signal(frame_signal, audio_signal, has_audio)
Audio models (all three are self-supervised speech encoders, a CNN waveform encoder plus a Transformer, pretrained
with different objectives and fine-tuned on different deepfake data; no other architecture family passed the probe):
    wav2vec2_xlsr   wav2vec2 XLS-R large (contrastive pretraining on 53 languages) fine-tuned on ASVspoof2019 LA;
                    card reports EER 4.0% on its evaluation subset; no paper; English studio TTS/VC attacks only.
    hubert_itw      HuBERT base (masked cluster prediction, LibriSpeech 960 h) fine-tuned on In-the-Wild
                    (Müller et al. 2022, arXiv:2203.16263: deepfakes of English-speaking public figures from social
                    media); card reports EER 1.4%; no model paper; English only.
    wavlm_asvspoof5 WavLM base (denoising masked prediction, LibriSpeech) fine-tuned on ASVspoof5 extended with other
                    deepfake datasets (Combei et al. 2024, arXiv:2408.07414); config has no label names, index 0 is
                    the spoof class (macOS TTS -> 1.00 on LABEL_0, LibriSpeech -> 1.00 on LABEL_1); English only.
"""

from functools import cache

import numpy as np
import pandas as pd
import soundfile
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from pipeline.hf import DEVICE
from pipeline.layer2_label.track2_ai_generated.image_detector import IMAGE_MODELS, image_ai_probabilities
from pipeline.layer2_label.track2_ai_generated.quorum import quorum
from pipeline.step import DATA, progress

# Speech deepfake detectors: (short name, Hugging Face id, revision, id2label name of the fake class).
AUDIO_MODELS = [
    (
        "wav2vec2_xlsr",
        "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
        "f7050b586236dc910d1157f430def2d0647b02b4",
        "fake",
    ),
    ("hubert_itw", "abhishtagatya/hubert-base-960h-itw-deepfake", "965d3063b43e05849f677740ecda3d4cd6b18131", "spoof"),
    ("wavlm_asvspoof5", "DavidCombei/wavLM-base-Deepfake_V2", "03396be9160ddb0dd38f1718428b595707577641", "LABEL_0"),
]
# Post types that carry their own media: reposts copy the parent, replies go to classify_replies.
OWN_CONTENT_POST_TYPES = ["original", "quote"]
# Audio is scored in windows of at most this many seconds (the classifiers were trained on short clips).
AUDIO_WINDOW_SECONDS = 10
# The classifiers expect 16 kHz mono audio.
AUDIO_SAMPLE_RATE = 16000


@cache
def audio_model(model_id: str, revision: str):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
    model = AutoModelForAudioClassification.from_pretrained(model_id, revision=revision)
    return feature_extractor, model.to(DEVICE).eval()


def audio_ai_probabilities(path: str) -> dict[str, float]:
    """Mean fake-speech probability over equal windows of one wav file (path relative to data/) per model short_name."""
    audio, sample_rate = soundfile.read(DATA / path, dtype="float32")
    assert sample_rate == AUDIO_SAMPLE_RATE, f"{path} is {sample_rate} Hz, expected {AUDIO_SAMPLE_RATE} Hz"
    windows = np.array_split(audio, int(np.ceil(len(audio) / (AUDIO_WINDOW_SECONDS * AUDIO_SAMPLE_RATE))))
    probabilities = {}
    for short_name, model_id, revision, ai_label in AUDIO_MODELS:
        feature_extractor, model = audio_model(model_id, revision)
        fake = model.config.label2id[ai_label]
        window_probabilities = []
        for window in windows:
            inputs = feature_extractor(window, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                window_probabilities.append(model(**inputs).logits.softmax(-1)[0, fake].item())
        probabilities[short_name] = float(np.mean(window_probabilities))
    return probabilities


def frame_ai_probabilities(paths: list[str]) -> dict[str, float]:
    """Mean AI probability over a video's keyframes (paths relative to data/) per image model short_name."""
    per_frame = [image_ai_probabilities(path) for path in paths]
    return {short_name: float(np.mean([frame[short_name] for frame in per_frame])) for short_name, *_ in IMAGE_MODELS}


def video_signal(frame_signal: str, audio_signal: str, has_audio: bool) -> str:
    """Synthetic if either track is (deepfake audio over real footage is still manipulated); real if the frames are
    real and the audio is real or absent; else disagree."""
    if "synthetic" in (frame_signal, audio_signal):
        return "synthetic"
    if frame_signal == "real" and (audio_signal == "real" or not has_audio):
        return "real"
    return "disagree"


def run(posts: pd.DataFrame) -> pd.DataFrame:
    videos = posts[(posts["media_type"] == "video") & posts["post_type"].isin(OWN_CONTENT_POST_TYPES)]
    audio_short_names = [short_name for short_name, *_ in AUDIO_MODELS]
    frames = pd.DataFrame(
        [frame_ai_probabilities(paths) for paths in progress(videos["keyframe_paths"].to_list(), "video frame detectors")],
        columns=[short_name for short_name, *_ in IMAGE_MODELS],
        dtype="float64",
    )
    audio = pd.DataFrame(
        [
            audio_ai_probabilities(path) if pd.notna(path) else dict.fromkeys(audio_short_names, np.nan)
            for path in progress(videos["audio_path"].to_list(), "video audio detectors")
        ],
        columns=audio_short_names,
        dtype="float64",
    )
    table = pd.concat(
        [frames.add_prefix("frame_ai_probability_"), audio.add_prefix("audio_ai_probability_")], axis=1
    )
    table.insert(0, "post_id", videos["post_id"].to_list())
    table["frame_signal"] = [quorum(row) for row in frames.to_numpy().tolist()]
    table["audio_signal"] = [quorum(row) for row in audio.to_numpy().tolist()]
    table["media_signal"] = [
        video_signal(frame, sound, has_audio)
        for frame, sound, has_audio in zip(table["frame_signal"], table["audio_signal"], videos["audio_path"].notna())
    ]
    return table.sort_values("post_id", ignore_index=True)
