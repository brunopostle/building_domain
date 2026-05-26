#!/usr/bin/env python3
"""
Alexander Pattern Language connectivity analysis.

Builds a graph from the 253 APL patterns using both the canonical
higher/lower pattern relationships (from apl_patterns.json) and any
additional relationships recorded in the database (related_pattern_ids).

Outputs:
- Hub patterns (highest degree)
- Bridge patterns (highest betweenness centrality)
- Community clusters
- Cross-section connectors
- Isolated / weakly-connected patterns
"""

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).parent.parent
APL_JSON = PROJECT_ROOT / "data" / "apl_patterns.json"
DB_PATH = PROJECT_ROOT / "bsos.db"

SECTIONS = {
    "Town and Country": range(1, 95),
    "Buildings": range(95, 205),
    "Construction": range(205, 254),
}


def section_of(number: int) -> str:
    for name, r in SECTIONS.items():
        if number in r:
            return name
    return "Unknown"


def slugify(name: str) -> str:
    return name.lower().replace(" ", "-").replace("'", "").replace(",", "").replace(".", "")


def build_graph() -> tuple[nx.Graph, dict, dict]:
    """Return (graph, num_to_id, id_to_data) from APL JSON + DB relationships."""
    apl = json.loads(APL_JSON.read_text())

    # Build slug → number lookup for resolving DB references
    slug_to_num: dict[str, int] = {}
    num_to_data: dict[int, dict] = {}
    id_to_num: dict[str, int] = {}  # DB UUID → pattern number

    for p in apl:
        num_to_data[p["number"]] = p
        slug_to_num[p["id"]] = p["number"]
        slug_to_num[slugify(p["name"])] = p["number"]

    G = nx.Graph()
    for p in apl:
        G.add_node(p["number"], name=p["name"], section=section_of(p["number"]), slug=p["id"])

    # Edges from canonical higher/lower relationships
    apl_edges = 0
    for p in apl:
        for lower in p.get("lower_patterns", []):
            target_num = slug_to_num.get(lower["id"]) or slug_to_num.get(slugify(lower["name"]))
            if target_num and not G.has_edge(p["number"], target_num):
                G.add_edge(p["number"], target_num, source="apl")
                apl_edges += 1

    # Edges from DB patterns.related_pattern_ids
    try:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        rows = con.execute("SELECT id, name, related_pattern_ids FROM patterns WHERE status='accepted'").fetchall()
        # Build UUID → number map
        for db_id, db_name, _ in rows:
            slug = slugify(db_name)
            if slug in slug_to_num:
                id_to_num[db_id] = slug_to_num[slug]

        db_edges = 0
        for db_id, db_name, rel_json in rows:
            src_num = id_to_num.get(db_id)
            if src_num is None:
                continue
            refs = json.loads(rel_json or "[]")
            for ref in refs:
                target_num = id_to_num.get(ref) or slug_to_num.get(ref) or slug_to_num.get(slugify(ref))
                if target_num and target_num != src_num and not G.has_edge(src_num, target_num):
                    G.add_edge(src_num, target_num, source="db")
                    db_edges += 1
        con.close()
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
              f"({apl_edges} from APL hierarchy, {db_edges} from DB relationships)")
    except Exception as e:
        print(f"Warning: could not load DB relationships: {e}")
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (APL hierarchy only)")

    return G, num_to_data


def fmt(number: int, data: dict) -> str:
    p = data[number]
    return f"  #{number:3d}  {p['name']}"


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def analyse(G: nx.Graph, num_to_data: dict) -> None:
    print_section("ALEXANDER PATTERN LANGUAGE — CONNECTIVITY ANALYSIS")
    print(f"\n253 patterns across 3 sections:")
    for name, r in SECTIONS.items():
        print(f"  {name}: patterns {r.start}–{r.stop - 1}")

    # ── Degree distribution ──────────────────────────────────────────────────
    print_section("TOP 20 HUBS  (highest degree — most connections)")
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    for num, deg in degrees[:20]:
        section = num_to_data[num]["section"].split(":")[0].strip()
        print(f"  #{num:3d}  {deg:2d} connections  {num_to_data[num]['name']}  [{section}]")

    print_section("BOTTOM 10 — least connected patterns")
    for num, deg in degrees[-10:]:
        section = num_to_data[num]["section"].split(":")[0].strip()
        print(f"  #{num:3d}  {deg:2d} connections  {num_to_data[num]['name']}  [{section}]")

    # ── Betweenness centrality ───────────────────────────────────────────────
    print_section("TOP 20 BRIDGES  (highest betweenness centrality)")
    bc = nx.betweenness_centrality(G, normalized=True)
    top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:20]
    for num, score in top_bc:
        section = num_to_data[num]["section"].split(":")[0].strip()
        print(f"  #{num:3d}  {score:.4f}  {num_to_data[num]['name']}  [{section}]")

    # ── Community detection ──────────────────────────────────────────────────
    print_section("COMMUNITY CLUSTERS  (Louvain method)")
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(G, seed=42)
    except ImportError:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))

    communities_sorted = sorted(communities, key=len, reverse=True)
    print(f"\n{len(communities_sorted)} communities found:\n")
    for i, community in enumerate(communities_sorted):
        members = sorted(community)
        sections_in = Counter(section_of(n) for n in members)
        dominant = sections_in.most_common(1)[0][0]
        # Top 3 hubs within community
        hubs = sorted(members, key=lambda n: G.degree(n), reverse=True)[:3]
        hub_names = ", ".join(f"#{n} {num_to_data[n]['name']}" for n in hubs)
        print(f"  Cluster {i+1:2d} ({len(members):3d} patterns, dominant: {dominant})")
        print(f"          Key nodes: {hub_names}")
        print(f"          Patterns:  {', '.join(str(n) for n in members)}")
        print()

    # ── Cross-section bridges ────────────────────────────────────────────────
    print_section("CROSS-SECTION BRIDGES  (edges connecting Town↔Buildings, Buildings↔Construction)")
    cross = []
    for u, v in G.edges():
        su, sv = section_of(u), section_of(v)
        if su != sv:
            cross.append((u, v, su, sv))

    cross_counts: Counter = Counter()
    for u, v, su, sv in cross:
        pair = tuple(sorted([su, sv]))
        cross_counts[pair] += 1

    print(f"\n  Cross-section edges: {len(cross)}")
    for pair, count in cross_counts.most_common():
        print(f"    {pair[0]} ↔ {pair[1]}: {count} edges")

    print(f"\n  Patterns with the most cross-section connections:")
    cross_degree: Counter = Counter()
    for u, v, su, sv in cross:
        cross_degree[u] += 1
        cross_degree[v] += 1
    for num, cnt in cross_degree.most_common(15):
        su = section_of(num)
        print(f"  #{num:3d}  {cnt} cross-section  {num_to_data[num]['name']}  [{su}]")

    # ── Connectivity summary ─────────────────────────────────────────────────
    print_section("CONNECTIVITY SUMMARY")
    components = list(nx.connected_components(G))
    print(f"\n  Connected components: {len(components)}")
    if len(components) > 1:
        for i, comp in enumerate(sorted(components, key=len, reverse=True)):
            names = ", ".join(f"#{n} {num_to_data[n]['name']}" for n in sorted(comp)[:3])
            print(f"    Component {i+1}: {len(comp)} patterns — {names}{'…' if len(comp) > 3 else ''}")

    diameter = nx.diameter(G) if nx.is_connected(G) else "n/a (disconnected)"
    avg_path = nx.average_shortest_path_length(G) if nx.is_connected(G) else "n/a"
    avg_clustering = nx.average_clustering(G)
    print(f"\n  Graph diameter:           {diameter}")
    if isinstance(avg_path, float):
        print(f"  Average shortest path:    {avg_path:.2f}")
    else:
        print(f"  Average shortest path:    {avg_path}")
    print(f"  Average clustering coeff: {avg_clustering:.4f}")
    print(f"  Graph density:            {nx.density(G):.4f}")

    # Degree histogram
    deg_vals = [d for _, d in G.degree()]
    print(f"\n  Degree stats: min={min(deg_vals)}, max={max(deg_vals)}, "
          f"mean={sum(deg_vals)/len(deg_vals):.1f}, median={sorted(deg_vals)[len(deg_vals)//2]}")

    bins = [0, 5, 10, 15, 20, 25, 30, 50]
    print("\n  Degree distribution:")
    for lo, hi in zip(bins, bins[1:]):
        count = sum(1 for d in deg_vals if lo <= d < hi)
        bar = "█" * count
        print(f"    {lo:2d}-{hi-1:2d}: {bar} ({count})")
    hi_count = sum(1 for d in deg_vals if d >= bins[-1])
    print(f"    {bins[-1]}+  : {'█' * hi_count} ({hi_count})")


if __name__ == "__main__":
    G, num_to_data = build_graph()
    analyse(G, num_to_data)
