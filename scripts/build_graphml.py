"""Build prayagraj_drive.graphml from authentic street segments."""

from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd
from shapely import wkt

REPO_ROOT = Path(__file__).resolve().parent
METADATA_FILE = REPO_ROOT / "data" / "processed" / "segment_metadata.csv"
GRAPH_FILE = REPO_ROOT / "data" / "raw" / "prayagraj_drive.graphml"

df = pd.read_csv(METADATA_FILE)
print(f"Loaded {len(df)} segments from {METADATA_FILE}")

G = nx.MultiDiGraph()
G.graph["crs"] = "epsg:4326"

node_map = {}
node_counter = 1

for _, row in df.iterrows():
    sid = str(row["segment_id"]).strip()
    wkt_str = str(row["geometry_wkt"]).strip()
    geom = wkt.loads(wkt_str)
    coords = list(geom.coords)  # [(lon, lat), ...]
    if len(coords) < 2:
        continue

    u_coord = (round(coords[0][0], 5), round(coords[0][1], 5))
    v_coord = (round(coords[-1][0], 5), round(coords[-1][1], 5))

    if u_coord not in node_map:
        node_map[u_coord] = node_counter
        G.add_node(node_counter, x=u_coord[0], y=u_coord[1], osmid=node_counter)
        node_counter += 1

    if v_coord not in node_map:
        node_map[v_coord] = node_counter
        G.add_node(node_counter, x=v_coord[0], y=v_coord[1], osmid=node_counter)
        node_counter += 1

    u = node_map[u_coord]
    v = node_map[v_coord]

    length = float(row.get("length", 100.0))
    travel_time = float(row.get("travel_time", 15.0))
    safety_score = float(row.get("safety_score", 65.0))
    level = str(row.get("level", "caution"))
    factors = str(row.get("top_factors", "urban corridor"))

    # Bidirectional edges for drivable street
    G.add_edge(
        u,
        v,
        0,
        segment_id=sid,
        length=length,
        travel_time=travel_time,
        safety_score=safety_score,
        safety_level=level,
        top_factors=factors,
        highway="primary" if safety_score > 70 else "secondary",
        geometry=geom,
    )
    G.add_edge(
        v,
        u,
        0,
        segment_id=f"{sid}_rev",
        length=length,
        travel_time=travel_time,
        safety_score=safety_score,
        safety_level=level,
        top_factors=factors,
        highway="primary" if safety_score > 70 else "secondary",
        geometry=geom,
    )

print(f"Constructed MultiDiGraph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
ox.save_graphml(G, filepath=GRAPH_FILE)
print(f"Successfully saved graphml to {GRAPH_FILE} ({GRAPH_FILE.stat().st_size // 1024} KB)")
