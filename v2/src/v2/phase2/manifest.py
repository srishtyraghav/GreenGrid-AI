"""V2 Phase-2 audit 5/5: manifest of audit artifacts.

Writes data/v2/phase2/manifest/manifest.json (project-relative file list with
sha256 + sizes, generated-at UTC, tool versions) and SHA256SUMS.txt over every
file under data/v2/phase2/.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT_ROOT, dump_json, rel, sha256_file  # noqa: E402

MANIFEST_DIR = OUT_ROOT / "manifest"


def tool_versions() -> dict:
    import geopandas
    import numpy
    import pandas
    import rasterio
    import shapely

    return {
        "python": platform.python_version(),
        "rasterio": rasterio.__version__,
        "geopandas": geopandas.__version__,
        "shapely": shapely.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }


def main() -> int:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in OUT_ROOT.rglob("*")
        if p.is_file() and MANIFEST_DIR not in p.parents
    )
    entries = []
    for p in files:
        entries.append({
            "path": rel(p),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool_versions": tool_versions(),
        "root": "data/v2/phase2",
        "file_count": len(entries),
        "total_bytes": sum(e["size_bytes"] for e in entries),
        "files": entries,
    }
    dump_json(manifest, MANIFEST_DIR / "manifest.json")
    with open(MANIFEST_DIR / "SHA256SUMS.txt", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {e['path']}\n")
    print(f"[manifest] {len(entries)} files -> {MANIFEST_DIR}/manifest.json, SHA256SUMS.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
