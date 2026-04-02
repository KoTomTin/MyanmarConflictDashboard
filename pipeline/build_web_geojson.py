"""
Build a web-optimized Myanmar township boundary GeoJSON for the dashboard.

This keeps the source boundary file intact and writes a lighter version for
runtime use. The output:
  - preserves topology
  - keeps only the properties the app actually uses
  - simplifies geometry enough to materially reduce Dash callback payload size
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import shape, mapping


ROOT = Path(__file__).resolve().parents[1]
SHAPES = ROOT / "data" / "shapes"
SOURCE = SHAPES / "boundaries.geojson"
OUTPUT = SHAPES / "boundaries_web.geojson"

KEEP_PROPERTIES = {"TS_PCODE", "TS", "ST"}
SIMPLIFY_TOLERANCE = 0.005


def main():
    obj = json.loads(SOURCE.read_text(encoding="utf-8"))

    out = {"type": "FeatureCollection", "features": []}
    for feature in obj.get("features", []):
        geom = shape(feature["geometry"]).simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
        out["features"].append({
            "type": "Feature",
            "properties": {
                key: value
                for key, value in feature.get("properties", {}).items()
                if key in KEEP_PROPERTIES
            },
            "geometry": mapping(geom),
        })

    OUTPUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")

    src_size = SOURCE.stat().st_size / (1024 * 1024)
    out_size = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"Source: {SOURCE.name} {src_size:.3f} MB")
    print(f"Output: {OUTPUT.name} {out_size:.3f} MB")
    print(f"Features: {len(out['features'])}")


if __name__ == "__main__":
    main()
