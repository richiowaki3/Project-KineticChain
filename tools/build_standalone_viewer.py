# -*- coding: utf-8 -*-
"""
Builds an all-in-one standalone HTML viewer with embedded JSON data.
Allows opening the visualizer directly from local file:// in any browser without CORS issues.
"""

import json
from pathlib import Path


def build_standalone_viewer(json_path: Path, template_html_path: Path, output_html_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(template_html_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Inject embedded data call before end of script
    json_str = json.dumps(data, ensure_ascii=False)
    injection = f"""
      // Embedded dataset auto-load
      window.addEventListener('DOMContentLoaded', () => {{
        const embeddedData = {json_str};
        window.loadVisualizerData(embeddedData);
      }});
    """

    injected_html = template.replace(
      "window.loadVisualizerData = loadJsonData;",
      f"window.loadVisualizerData = loadJsonData;\n{injection}"
    )

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(injected_html)

    print(f"Generated standalone viewer: {output_html_path} ({output_html_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    json_p = root / "visualizer" / "data" / "MRka5p5qTxw_vis.json"
    template_p = root / "visualizer" / "index.html"
    out_p = root / "visualizer" / "standalone_MRka5p5qTxw.html"
    build_standalone_viewer(json_p, template_p, out_p)
