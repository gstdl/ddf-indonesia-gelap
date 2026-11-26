"""Box: Network analysis: communities, cascades, all labels.

Output: one row per post. Columns: post_id, community_id (Louvain community of the post's author, named by its
smallest lowercase handle), cascade_id (root post of its parent_post_id chain within the dataset), cascade_size
(posts in the cascade).
"""

import networkx as nx
import pandas as pd

SEED = 42  # Louvain seed


def communities(authors: pd.Series, edges: pd.DataFrame) -> dict[str, str]:
    graph = nx.Graph()
    graph.add_nodes_from(sorted(authors))
    for edge in edges.itertuples(index=False):
        if graph.has_edge(edge.source_author, edge.target_author):
            graph[edge.source_author][edge.target_author]["weight"] += edge.weight
        else:
            graph.add_edge(edge.source_author, edge.target_author, weight=edge.weight)
    community_of = {}
    for community in nx.community.louvain_communities(graph, weight="weight", seed=SEED):
        for handle in community:
            community_of[handle] = min(community)
    return community_of


def cascade_root(post_id: str, parent_of: dict[str, str]) -> str:
    while pd.notna(parent_of.get(post_id)) and parent_of[post_id] in parent_of:
        post_id = parent_of[post_id]
    return post_id


def run(merged: pd.DataFrame, interaction_network: pd.DataFrame) -> pd.DataFrame:
    authors = merged["author"].str.lower()
    community_of = communities(authors.drop_duplicates(), interaction_network)
    parent_of = dict(zip(merged["post_id"], merged["parent_post_id"]))
    network = merged[["post_id"]].copy()
    network["community_id"] = authors.map(community_of)
    network["cascade_id"] = network["post_id"].map(lambda post_id: cascade_root(post_id, parent_of))
    network["cascade_size"] = network.groupby("cascade_id")["post_id"].transform("size")
    return network.sort_values("post_id", ignore_index=True)
