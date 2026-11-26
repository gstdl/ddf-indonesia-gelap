"""Entrypoint: builds the #IndonesiaGelap pipeline from the Miro board.

One box on the board = one file in pipeline/ = one saved table in data/.
`step(box, *inputs)` runs `box` on the saved tables of `inputs` (the arrows into the box).
Comment out a step to skip it; later steps reuse its last saved table.
Every box run is logged to the console and appended to data/pipeline.log.

Run: uv run python main.py path/to/brandwatch_export.csv
"""

import argparse
import importlib.util
import logging
from pathlib import Path

from pipeline.layer1_prepare import (
    cluster,
    document,
    download_media,
    export,
    image,
    link,
    normalize,
    text_only,
    video,
    x_only,
)
from pipeline.layer2_label.track1_hoax import cekfakta, claimed_false, google_fact_check, komdigi, similar_to_known_hoax
from pipeline.layer2_label.track2_ai_generated import (
    classify_replies,
    collect_replies,
    image_detector,
    signals_agree,
    text_detector,
    video_detector,
)
from pipeline.layer2_label.track3_account import check_verified
from pipeline.layer2_label.track4_buzzer import interaction_network, score_accounts
from pipeline.layer2_label.track5_narrative import narrative_coding
from pipeline.layer2_label.track6_traction import traction_signals
from pipeline.layer3_analyze import merge, network_analysis
from pipeline.layer3_analyze.analysis import rq1a_prevalence, rq1b_spreaders, rq1c_narratives
from pipeline.layer4_publish import anonymize
from pipeline import step as step_module
from pipeline.step import DATA, step

LOG_FILE = DATA / "pipeline.log"


# Comment a source out to drop it from both hoax checks.
FACT_CHECK_SOURCES = [
    komdigi,
    # cekfakta,  # blocked: cekfakta.com times out from this machine
    google_fact_check,
]


def main(brandwatch_export: Path) -> None:
    # Layer 1: collect and prepare posts
    step(export, path=brandwatch_export)
    step(x_only, export)  # not on the board: drop news, Facebook, TikTok, ... rows
    step(download_media, x_only)
    step(normalize, download_media)
    step(text_only, normalize)  # Media type? Text
    step(image, normalize)  # Media type? Image
    step(video, normalize)  # Media type? Video
    step(document, text_only, image, video)
    step(cluster, document)
    step(link, cluster)

    # Layer 2, track 1: hoax label
    for source in FACT_CHECK_SOURCES:
        step(source)
    step(claimed_false, link, *FACT_CHECK_SOURCES)
    step(similar_to_known_hoax, link, claimed_false, *FACT_CHECK_SOURCES)

    # Layer 2, track 2: AI-generated / deepfake label
    step(text_detector, link)  # Content type? Text
    step(image_detector, link)  # Content type? Image
    step(video_detector, link)  # Content type? Video
    step(collect_replies, link)
    step(classify_replies, collect_replies)
    step(signals_agree, link, text_detector, image_detector, video_detector, classify_replies)

    # Layer 2, track 3: account label
    step(check_verified, link)

    # Layer 2, track 4: buzzer label
    step(interaction_network, link)
    step(score_accounts, link)

    # Layer 2, track 5: narrative label
    step(narrative_coding, link)

    # Layer 2, track 6: traction and monetization
    step(traction_signals, link)

    # Layer 3: merge and analyze networks
    step(
        merge,
        link,
        similar_to_known_hoax,  # track 1
        signals_agree,  # track 2
        check_verified,  # track 3
        score_accounts,  # track 4
        narrative_coding,  # track 5
        traction_signals,  # track 6
    )
    step(network_analysis, merge, interaction_network)
    step(rq1a_prevalence, merge)
    step(rq1b_spreaders, merge, network_analysis)
    step(rq1c_narratives, merge)

    # Layer 4: ethics review, then publish
    step(anonymize, rq1a_prevalence, name="rq1a_prevalence")
    step(anonymize, rq1b_spreaders, name="rq1b_spreaders")
    step(anonymize, rq1c_narratives, name="rq1c_narratives")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the #IndonesiaGelap pipeline.")
    parser.add_argument("brandwatch_export", type=Path, help="Brandwatch mentions export (CSV)")
    parser.add_argument(
        "--overwrite", nargs="+", default=[], metavar="BOX",
        help='rerun these boxes even when up to date, e.g. layer1_prepare/image (boxes downstream rerun too)',
    )
    args = parser.parse_args()
    unknown = [box for box in args.overwrite if importlib.util.find_spec("pipeline." + box.replace("/", ".")) is None]
    if unknown:
        parser.error(f"unknown boxes for --overwrite: {', '.join(unknown)}")
    step_module.OVERWRITE.update(args.overwrite)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
    )
    try:
        main(args.brandwatch_export.resolve())
    except Exception:
        logging.exception("pipeline failed")
        raise
