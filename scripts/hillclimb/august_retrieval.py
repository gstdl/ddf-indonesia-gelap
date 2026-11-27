"""Compare full and August-focused claim shortlists without accepting any matches."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.hf import embed
from pipeline.layer2_label.track1_hoax.claimed_false import false_claims
from scripts.hillclimb.assertion import load_claim_sources
from scripts.hillclimb.labeled_posts import load_august_posts

EVENT_TERMS = re.compile(
    r"\b(?:DPR|DPRD|MPR|demonstrasi|demo|pendemo|pengunjuk rasa|unjuk rasa|Affan|Sahroni|"
    r"Brimob|ojol|Pejompongan|Kwitang|penjarahan|dijarah|perampasan aset)\b",
    re.IGNORECASE,
)


def focus_indices(claims, sources):
    indices = set()
    for source in sources:
        days = source.published_at.dt.tz_convert("Asia/Jakarta").dt.strftime("%Y-%m-%d")
        eligible = days.between("2025-08-01", "2025-09-30") & source.claim_text.str.contains(
            EVENT_TERMS, na=False,
        )
        keys = source.loc[eligible, ["claim_text", "source_url"]].drop_duplicates()
        indices.update(claims.reset_index().merge(keys, on=["claim_text", "source_url"])["index"])
    return np.array(sorted(indices), dtype=int)


def ranked(scores):
    return np.argsort(-scores, axis=1, kind="stable")


def fuse(semantic, keyword):
    semantic_rank = np.argsort(ranked(semantic), axis=1) + 1
    keyword_rank = np.argsort(ranked(keyword), axis=1) + 1
    # Zero lexical matches contribute no vote, irrespective of their tie order.
    return 1 / (60 + semantic_rank) + (keyword > 0) / (60 + keyword_rank)


def load_retrieval(data_dir, labels_path):
    sources = load_claim_sources(data_dir)
    claims = false_claims(sources)
    posts = load_august_posts(data_dir)
    labels = json.loads(labels_path.read_text())
    archived = labels["archived_positives"]
    reviewed = labels["x_review"]
    controlled = json.loads(Path(__file__).with_name("claim_cases.json").read_text())
    targets = [[index] for index, _ in archived] + [
        claims.index[claims.claim_text.eq(claim)].tolist() for claim, *_ in controlled
    ]
    if not all(indices and all(0 <= index < len(claims) for index in indices) for indices in targets):
        raise ValueError("The labeled targets must exist in the original sorted claim cache")
    documents = [quote for _, quote in archived] + [paraphrase for _, paraphrase, *_ in controlled]
    documents.extend(posts.loc[reviewed, "document"])
    return claims, sources, documents, targets, len(archived), reviewed


def score_documents(documents, claims):
    claim_texts = claims.claim_text.tolist()
    semantic = embed(documents) @ embed(claim_texts).T
    # Fit on the entire cache so focusing changes eligibility, not term weights.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    terms = vectorizer.fit_transform(claim_texts)
    keyword = (vectorizer.transform(documents) @ terms.T).toarray()
    return semantic, keyword


def report_ranks(label, order, targets, archive_count):
    ranks = []
    for row, target in zip(order[:len(targets)], targets, strict=True):
        matches = np.flatnonzero(np.isin(row, target))
        ranks.append(int(matches[0]) + 1 if matches.size else None)
    archived = ranks[:archive_count]
    controlled = ranks[archive_count:]
    counts = [[sum(rank is not None and rank <= k for rank in group) for k in (1, 3, 10)]
              for group in (archived, controlled)]
    print(f"{label}: archived ranks {archived}; controlled ranks {controlled}")
    print(f"  recall counts @1,3,10 (archived, controlled): {counts}")


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, required=True, help="Original pipeline data root")
    parser.add_argument("--labels", type=Path, required=True, help="Local real-claims-labels.json")
    return parser.parse_args()


def main():
    args = parse_args(__doc__)
    claims, sources, documents, targets, archive_count, reviewed = load_retrieval(args.data_root, args.labels)
    semantic, keyword = score_documents(documents, claims)
    focus = focus_indices(claims, sources)
    for label, indices in (("full", np.arange(len(claims))), ("focused", focus)):
        print(f"{label} corpus rows: {len(indices)}")
        if not indices.size:
            raise ValueError("No claims match the August focus rule")
        sem, lex = semantic[:, indices], keyword[:, indices]
        for method, scores in (("semantic", sem), ("keyword", lex), ("hybrid", fuse(sem, lex))):
            order = indices[ranked(scores)]
            if method == "keyword":
                # A zero score means no shared words or bigrams, not a candidate.
                order = [row[:count] for row, count in zip(order, np.count_nonzero(lex, axis=1), strict=True)]
            report_ranks(f"{label} {method}", order, targets, archive_count)
            for post, row in zip(reviewed, order[len(targets):], strict=True):
                print(f"  reviewed X row {post}: {row[:3].tolist()}")


if __name__ == "__main__":
    main()
