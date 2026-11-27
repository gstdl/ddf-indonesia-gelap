"""Boost August-related claims without removing unrelated or undated reviews."""

import numpy as np

from scripts.hillclimb.august_retrieval import (
    focus_indices,
    fuse,
    load_retrieval,
    parse_args,
    ranked,
    report_ranks,
    score_documents,
)


def main():
    args = parse_args(__doc__)
    claims, sources, documents, targets, archive_count, reviewed = load_retrieval(args.data_root, args.labels)
    fusion = fuse(*score_documents(documents, claims))
    focused = np.isin(np.arange(len(claims)), focus_indices(claims, sources))
    baseline = ranked(fusion)
    review_start = len(targets)
    for boost in (1.0, 1.10, 1.25, 1.50):
        order = baseline if boost == 1.0 else ranked(fusion * np.where(focused, boost, 1.0))
        report_ranks(f"{(boost - 1):+.0%}", order, targets, archive_count)
        changed = np.count_nonzero(order[review_start:, 0] != baseline[review_start:, 0])
        print(f"  reviewed X top result changed: {changed}/{len(reviewed)}")
        for post, row in zip(reviewed, order[review_start:], strict=True):
            print(f"  reviewed X row {post}: {row[:3].tolist()}")


if __name__ == "__main__":
    main()
