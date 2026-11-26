"""Box: Video: keyframes, pHash, OCR, STT, caption. Takes the "Video" arrow of "Media type?".

Output: video posts (media_type == "video") plus
    keyframe_paths     list of keyframes data/raw/keyframes/<post_id>_<nn>.jpg, relative to data/
    phash              perceptual hash of the middle keyframe
    media_ocr          unique non-empty Florence-2 <OCR> lines of all keyframes, in order; NaN when empty
    media_caption      Florence-2 <CAPTION> of the middle keyframe, NaN when empty
    audio_path         audio track data/raw/audio/<post_id>.wav (16 kHz mono), relative to data/
    media_transcript   Whisper speech-to-text of the audio track (language detected), NaN when empty
    media_text         non-empty OCR, caption and transcript joined with "\n", NaN when all are empty
Keyframes and audio files are reused when a re-run finds them.

Thumbnails: the real Brandwatch export gives only a still image (.jpg) for X videos, no video file.
Such a post's only keyframe is the thumbnail itself, and audio_path / media_transcript are NaN
(no audio exists), so video_detector judges it on the frame models alone.
"""

import os
import subprocess
from functools import cache

import cv2
import imagehash
import pandas as pd
import soundfile
from PIL import Image
from transformers import pipeline

from pipeline.hf import DEVICE
from pipeline.layer1_prepare.image import describe, join_texts
from pipeline.step import DATA, progress

KEYFRAME_DIR = DATA / "raw" / "keyframes"
AUDIO_DIR = DATA / "raw" / "audio"
# Media files with these suffixes are video thumbnails (still images), not videos.
THUMBNAIL_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def is_thumbnail(media_path: str) -> bool:
    return os.path.splitext(media_path)[1].lower() in THUMBNAIL_SUFFIXES


# Seconds between two keyframes.
KEYFRAME_INTERVAL_SECONDS = 2
# Most keyframes taken from one video.
MAX_KEYFRAMES = 10
# Whisper expects 16 kHz mono audio.
AUDIO_SAMPLE_RATE = 16000

# Multilingual speech-to-text (incl. Indonesian).
WHISPER_MODEL = "openai/whisper-small"
WHISPER_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
# Whisper reads audio in 30-second windows.
WHISPER_CHUNK_SECONDS = 30


@cache
def whisper():
    return pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        revision=WHISPER_REVISION,
        device=DEVICE,
        chunk_length_s=WHISPER_CHUNK_SECONDS,
    )


def extract_keyframes(media_path: str, post_id: str) -> list[str]:
    """One frame every KEYFRAME_INTERVAL_SECONDS, at most MAX_KEYFRAMES; saved once, reused on a re-run."""
    video = cv2.VideoCapture(str(DATA / media_path))
    duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
    seconds = list(range(0, int(duration) + 1, KEYFRAME_INTERVAL_SECONDS))
    seconds = [second for second in seconds if second < duration][:MAX_KEYFRAMES]
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, second in enumerate(seconds):
        path = KEYFRAME_DIR / f"{post_id}_{number:02d}.jpg"
        if not path.exists():
            video.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            found, frame = video.read()
            assert found, f"no frame at {second}s in {media_path}"
            tmp_path = path.with_name(path.stem + ".tmp.jpg")
            cv2.imwrite(str(tmp_path), frame)
            os.replace(tmp_path, path)
        paths.append(str(path.relative_to(DATA)))
    video.release()
    return paths


def extract_audio(media_path: str, post_id: str) -> str:
    """16 kHz mono wav of the video's audio track; saved once, reused on a re-run."""
    path = AUDIO_DIR / f"{post_id}.wav"
    if not path.exists():
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.stem + ".tmp.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(DATA / media_path), "-vn", "-ac", "1",
             "-ar", str(AUDIO_SAMPLE_RATE), str(tmp_path)],
            check=True,
        )
        os.replace(tmp_path, path)
    return str(path.relative_to(DATA))


def transcribe(audio_path: str) -> str:
    audio, sample_rate = soundfile.read(DATA / audio_path, dtype="float32")
    result = whisper()({"raw": audio, "sampling_rate": sample_rate}, generate_kwargs={"task": "transcribe"})
    return result["text"].strip()


def keyframe_ocr(keyframe_paths: list[str]) -> str | None:
    """Unique non-empty OCR lines over all keyframes, in order of appearance."""
    lines = []
    for path in keyframe_paths:
        for line in describe(Image.open(DATA / path), "<OCR>").splitlines():
            if line.strip() and line.strip() not in lines:
                lines.append(line.strip())
    return "\n".join(lines) if lines else None


def run(posts: pd.DataFrame) -> pd.DataFrame:
    videos = posts[posts["media_type"] == "video"]
    keyframes = [
        [path] if is_thumbnail(path) else extract_keyframes(path, post_id)
        for path, post_id in zip(videos["media_path"], videos["post_id"])
    ]
    middle = [Image.open(DATA / paths[len(paths) // 2]) for paths in keyframes]
    ocr = [keyframe_ocr(paths) for paths in progress(keyframes, "video OCR")]
    caption = [describe(image, "<CAPTION>") or None for image in progress(middle, "video caption")]
    audio = [
        None if is_thumbnail(path) else extract_audio(path, post_id)
        for path, post_id in zip(videos["media_path"], videos["post_id"])
    ]
    transcript = [transcribe(path) or None if path is not None else None for path in progress(audio, "video transcript")]
    return videos.assign(
        keyframe_paths=keyframes,
        phash=[str(imagehash.phash(image)) for image in middle],
        media_ocr=ocr,
        media_caption=caption,
        audio_path=audio,
        media_transcript=transcript,
        media_text=[join_texts(*texts) for texts in zip(ocr, caption, transcript)],
    )
