"""Sweep shortlist sizes and cosine floors with the frozen assertion verifier."""

from pipeline.layer2_label.track1_hoax.claimed_false import false_claims
from pipeline.layer2_label.track1_hoax.similar_to_known_hoax import KNOWN_HOAX_SIMILARITY

from .assertion import (
    ENTAILMENT_CUTOFF,
    load_claim_sources,
    load_trial,
    rank_trial,
    selected_counts,
    trial_parser,
    verify_ranked,
)

KS = (1, 3, 5, 10)
FLOORS = (KNOWN_HOAX_SIMILARITY, 0.60)


def main() -> None:
    args = trial_parser(__doc__).parse_args()
    claims = false_claims(load_claim_sources(args.data_root))
    local = load_trial(args.data_root, args.trial)
    ranked = rank_trial(local, claims, k=max(KS))
    pairs, _, elapsed_seconds = verify_ranked(ranked, min(FLOORS))
    print(f"Claims: {len(claims)}; documents by kind: {local.kind.value_counts(sort=False).to_dict()}")
    print(f"NLI plus tokenization seconds: {elapsed_seconds:.2f}; pairs: {pairs}")
    for k in KS:
        shortlist = ranked.loc[ranked["rank"].le(k)]
        intended = shortlist.loc[shortlist.kind.eq("assertion") & shortlist.correct]
        print(f"k={k}: no-floor intended assertion retrieval={intended.post.nunique()}")
        for floor in FLOORS:
            eligible = shortlist.loc[shortlist.cosine.ge(floor)]
            accepted = eligible.loc[eligible.entailment.ge(ENTAILMENT_CUTOFF)].drop_duplicates("post")
            retrieved = intended.loc[intended.cosine.ge(floor), "post"].nunique()
            print(f"  floor={floor:.2f}; pairs={len(eligible)}; intended assertions retrieved={retrieved}",
                  selected_counts(accepted))
    assertions = ranked.loc[ranked.kind.eq("assertion")]
    for target, rows in assertions.groupby("target", sort=False):
        matches = rows.loc[rows.correct, ["rank", "cosine", "entailment"]]
        print(f"Intended ranks for {target}: {matches.round(4).to_dict('records')}")
    ranked.to_json(args.output, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
