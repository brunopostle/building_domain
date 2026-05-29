"""bsos visualize command — render knowledge graph to interactive HTML or static PNG."""
import json
import math
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer()

# Entity-type colour palette (HTML/hex)
_TYPE_COLORS = {
    "component":  "#4a90d9",   # blue
    "activity":   "#f5a623",   # amber
    "space":      "#7ed321",   # green
    "material":   "#9b59b6",   # purple
    "system":     "#e74c3c",   # red
    "ifc_class":  "#95a5a6",   # grey
}
_DEFAULT_COLOR = "#cccccc"

# Predicate families → edge colour
_PRED_COLORS = {
    "requires":     "#f5a623",
    "depends_on":   "#f5a623",
    "contains":     "#7ed321",
    "supports":     "#4a90d9",
    "connects_to":  "#9b59b6",
    "protects_from":"#e74c3c",
}
_DEFAULT_EDGE_COLOR = "#555555"


def _load_graph(graph_path: Optional[str], db: Optional[str]):
    """Return a NetworkX graph, loading from pkl or building from db."""
    if graph_path:
        import joblib
        payload = joblib.load(graph_path)
        return payload["graph"]

    from bsos.cli.db_context import resolve_db_path
    from bsos.persistence.database import create_db_engine
    from bsos.graph import build_full_graph
    from sqlmodel import Session

    db_path = resolve_db_path(db)
    engine = create_db_engine(db_path)
    with Session(engine) as session:
        return build_full_graph(session)


def _entity_subgraph(g, max_nodes: int):
    """Return entity nodes + inter-entity edges, filtered to top max_nodes by degree."""
    import networkx as nx
    entities = {
        n for n, d in g.nodes(data=True)
        if d.get("node_type") == "entity"
    }
    entity_view = g.subgraph(entities)
    degrees = sorted(entity_view.degree(), key=lambda x: -x[1])
    top_ids = {n for n, _ in degrees[:max_nodes]}
    return entity_view.subgraph(top_ids).copy()


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BSOS Building Domain Knowledge Graph</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
  #network {{ width: 100vw; height: 100vh; }}
  #overlay {{
    position: absolute; top: 16px; left: 16px;
    background: rgba(22,27,34,0.92); border: 1px solid #30363d;
    border-radius: 8px; padding: 16px; min-width: 220px;
    pointer-events: none;
  }}
  #overlay h2 {{ font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #58a6ff; }}
  #stats {{ font-size: 12px; color: #8b949e; margin-bottom: 12px; line-height: 1.6; }}
  #legend {{ font-size: 12px; }}
  .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }}
  #search-box {{
    position: absolute; top: 16px; right: 16px;
    background: rgba(22,27,34,0.92); border: 1px solid #30363d;
    border-radius: 8px; padding: 12px;
  }}
  #search-box input {{
    background: #161b22; border: 1px solid #30363d; border-radius: 4px;
    color: #e6edf3; padding: 6px 10px; font-size: 12px; width: 200px;
    outline: none;
  }}
  #search-box input:focus {{ border-color: #58a6ff; }}
  #search-result {{ font-size: 11px; color: #8b949e; margin-top: 6px; }}
  #tooltip {{
    position: absolute; background: rgba(22,27,34,0.95); border: 1px solid #30363d;
    border-radius: 6px; padding: 10px 12px; font-size: 12px; pointer-events: none;
    display: none; max-width: 280px; line-height: 1.6;
  }}
  #tooltip .t-name {{ font-weight: 600; color: #e6edf3; margin-bottom: 4px; }}
  #tooltip .t-meta {{ color: #8b949e; }}
</style>
</head>
<body>
<div id="network"></div>

<div id="overlay">
  <h2>BSOS Knowledge Graph</h2>
  <div id="stats">
    {stat_entities} entities shown<br>
    {stat_edges} connections<br>
    {stat_total} total entities in base
  </div>
  <div id="legend">
    {legend_html}
  </div>
</div>

<div id="search-box">
  <input type="text" id="search-input" placeholder="Search entities…" />
  <div id="search-result"></div>
</div>

<div id="tooltip">
  <div class="t-name" id="t-name"></div>
  <div class="t-meta" id="t-meta"></div>
</div>

<script>
var nodes = new vis.DataSet({graph_nodes_json});
var edges = new vis.DataSet({graph_edges_json});

var container = document.getElementById('network');
var data = {{ nodes: nodes, edges: edges }};
var options = {{
  physics: {{
    enabled: true,
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{
      gravitationalConstant: -26,
      centralGravity: 0.004,
      springLength: 120,
      springConstant: 0.06,
      damping: 0.5,
      avoidOverlap: 0.1
    }},
    maxVelocity: 60,
    stabilization: {{ iterations: 200, fit: true, updateInterval: 25 }}
  }},
  nodes: {{
    shape: 'dot',
    font: {{ size: 0 }},
    borderWidth: 0
  }},
  edges: {{
    smooth: {{ type: 'continuous', roundness: 0.2 }},
    arrows: {{ to: {{ enabled: false }} }},
    selectionWidth: 0
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 100,
    hideEdgesOnDrag: true
  }},
  background: {{ color: '#0d1117' }}
}};

var network = new vis.Network(container, data, options);

// Tooltip
var tooltip = document.getElementById('tooltip');
var tName = document.getElementById('t-name');
var tMeta = document.getElementById('t-meta');

network.on('hoverNode', function(params) {{
  var node = nodes.get(params.node);
  tName.textContent = node.label;
  tMeta.innerHTML = node.entity_type + ' &bull; ' + node.degree + ' connections';
  tooltip.style.display = 'block';
}});
network.on('blurNode', function() {{ tooltip.style.display = 'none'; }});
document.addEventListener('mousemove', function(e) {{
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top  = (e.clientY + 14) + 'px';
}});

// Search
document.getElementById('search-input').addEventListener('input', function() {{
  var q = this.value.trim().toLowerCase();
  var res = document.getElementById('search-result');
  if (!q) {{ res.textContent = ''; network.unselectAll(); return; }}
  var matches = nodes.get({{ filter: function(n) {{ return n.label.toLowerCase().includes(q); }} }});
  if (matches.length === 0) {{
    res.textContent = 'No results';
    network.unselectAll();
  }} else {{
    res.textContent = matches.length + ' match' + (matches.length > 1 ? 'es' : '');
    network.selectNodes(matches.map(function(n) {{ return n.id; }}));
    if (matches.length === 1) {{ network.focus(matches[0].id, {{ scale: 1.5, animation: true }}); }}
  }}
}});

network.on('stabilizationProgress', function(p) {{
  var pct = Math.round(p.iterations / p.total * 100);
  document.getElementById('stats').querySelector('div') &&
    (document.getElementById('stats').textContent = 'Layouting… ' + pct + '%');
}});
</script>
</body>
</html>
"""


def _build_html(g, max_nodes: int, total_entities: int) -> str:
    sub = _entity_subgraph(g, max_nodes)

    graph_nodes = []
    for nid, data in sub.nodes(data=True):
        deg = sub.degree(nid)
        etype = data.get("entity_type", "component")
        color = _TYPE_COLORS.get(etype, _DEFAULT_COLOR)
        size = 4 + math.sqrt(deg) * 1.8
        graph_nodes.append({
            "id": nid,
            "label": data.get("name", nid),
            "entity_type": etype,
            "degree": deg,
            "color": {"background": color, "border": color, "highlight": {"background": "#ffffff", "border": "#ffffff"}},
            "size": round(size, 1),
        })

    graph_edges = []
    for i, (u, v, edata) in enumerate(sub.edges(data=True)):
        pred = edata.get("edge_type", "")
        color = _PRED_COLORS.get(pred, _DEFAULT_EDGE_COLOR)
        graph_edges.append({
            "id": i,
            "from": u,
            "to": v,
            "color": {"color": color + "55", "hover": color + "aa"},
            "width": 0.5,
            "title": pred,
        })

    # Legend HTML
    type_counts = {}
    for _, d in sub.nodes(data=True):
        t = d.get("entity_type", "component")
        type_counts[t] = type_counts.get(t, 0) + 1

    legend_parts = []
    for etype, color in _TYPE_COLORS.items():
        count = type_counts.get(etype, 0)
        if count == 0:
            continue
        legend_parts.append(
            f'<div class="legend-item">'
            f'<div class="legend-dot" style="background:{color}"></div>'
            f'{etype} ({count})'
            f'</div>'
        )

    return _HTML_TEMPLATE.format(
        stat_entities=sub.number_of_nodes(),
        stat_edges=sub.number_of_edges(),
        stat_total=f"{total_entities:,}",
        legend_html="\n    ".join(legend_parts),
        graph_nodes_json=json.dumps(graph_nodes),
        graph_edges_json=json.dumps(graph_edges),
    )


# ---------------------------------------------------------------------------
# PNG output
# ---------------------------------------------------------------------------

def _build_png(g, max_nodes: int, total_entities: int, output_path: str) -> None:
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    sub = _entity_subgraph(g, max_nodes)
    all_entities = {n for n, d in g.nodes(data=True) if d.get("node_type") == "entity"}

    typer.echo(f"Computing layout for {sub.number_of_nodes()} nodes…")
    # Use spectral layout on subgraph for structured positioning
    try:
        pos = nx.spectral_layout(sub, weight=None)
    except Exception:
        pos = nx.random_layout(sub, seed=42)

    # Pad position dict for any node in all_entities but not in sub
    # (we don't draw those, but keep the data clean)
    fig, ax = plt.subplots(figsize=(18, 14), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges first (faint)
    for u, v, edata in sub.edges(data=True):
        pred = edata.get("edge_type", "")
        color = _PRED_COLORS.get(pred, "#333333")
        xu, yu = pos[u]
        xv, yv = pos[v]
        ax.plot([xu, xv], [yu, yv], color=color, alpha=0.08, linewidth=0.3, zorder=1)

    # Draw nodes grouped by type for legend
    degrees = dict(sub.degree())
    max_deg = max(degrees.values()) if degrees else 1
    patches = []
    for etype, color in _TYPE_COLORS.items():
        nids = [n for n, d in sub.nodes(data=True) if d.get("entity_type") == etype]
        if not nids:
            continue
        xs = [pos[n][0] for n in nids]
        ys = [pos[n][1] for n in nids]
        sizes = [3 + (degrees.get(n, 0) / max_deg) * 80 for n in nids]
        ax.scatter(xs, ys, s=sizes, c=color, alpha=0.75, linewidths=0, zorder=2)
        patches.append(mpatches.Patch(color=color, label=etype))

    # Title and legend
    ax.set_title(
        f"BSOS Building Domain Knowledge Graph\n"
        f"{total_entities:,} entities · {g.number_of_edges():,} connections "
        f"(showing top {sub.number_of_nodes():,} by connectivity)",
        color="#e6edf3", fontsize=13, pad=16,
    )
    legend = ax.legend(
        handles=patches, loc="lower left",
        framealpha=0.6, facecolor="#161b22", edgecolor="#30363d",
        labelcolor="#e6edf3", fontsize=9, markerscale=1.2,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    typer.echo(f"Saved PNG: {output_path}")


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def visualize(
    graph: Optional[str] = typer.Option(
        None, "--graph", "-g",
        help="Path to bsos_graph.pkl (default: auto-detect or build from --db)",
    ),
    fmt: str = typer.Option("html", "--format", "-f", help="Output format: html or png"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path"),
    max_nodes: int = typer.Option(
        1000, "--max-nodes",
        help="Max entity nodes to include (highest-degree first)",
    ),
    db: Optional[str] = typer.Option(None, "--db"),
) -> None:
    """Render the knowledge graph as an interactive HTML page or static PNG."""
    fmt = fmt.lower()
    if fmt not in ("html", "png"):
        typer.echo(f"Unknown format '{fmt}'. Use html or png.", err=True)
        raise typer.Exit(1)

    # Locate graph file
    graph_path = graph
    if graph_path is None and Path("bsos_graph.pkl").exists():
        graph_path = "bsos_graph.pkl"

    typer.echo("Loading graph…")
    g = _load_graph(graph_path, db)

    total_entities = sum(1 for _, d in g.nodes(data=True) if d.get("node_type") == "entity")
    typer.echo(f"Graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges "
               f"({total_entities:,} entities)")

    default_output = f"bsos_graph.{fmt}"
    out = output or default_output

    if fmt == "html":
        typer.echo(f"Building interactive HTML (top {max_nodes} entities)…")
        html = _build_html(g, max_nodes, total_entities)
        Path(out).write_text(html, encoding="utf-8")
        size_kb = Path(out).stat().st_size // 1024
        typer.echo(f"Saved HTML: {out} ({size_kb} KB)")
        typer.echo("Open in a browser — requires internet for vis.js CDN.")
    else:
        _build_png(g, max_nodes, total_entities, out)
