#!/usr/bin/env python3
"""
generate_gallery.py
===================
Copies pipeline output PNGs to docs/ folder and generates an HTML gallery
for GitHub Pages. Run after each pipeline update, then git add/commit/push.

Usage:
    python generate_gallery.py

Then:
    cd /scratch/easmit31/cell_cell
    git add docs/
    git commit -m "Update results gallery"
    git push
"""

import os
import glob
import shutil
from pathlib import Path
from datetime import datetime

# ── PATHS ─────────────────────────────────────────────────────────────────────

REPO_DIR    = Path("/scratch/easmit31/cell_cell")
RESULTS_DIR = REPO_DIR / "results/within_region_analysis_corrected"
DOCS_DIR    = REPO_DIR / "docs"
IMG_DIR     = DOCS_DIR / "img"

DOCS_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

# ── SECTIONS: (title, glob pattern, output subfolder) ─────────────────────────

SECTIONS = [
    (
        "Cell Type Enrichment Heatmaps",
        "celltype_heatmaps",
        sorted(glob.glob(str(RESULTS_DIR / "hypergeometric_celltype/heatmap_*.png")))
    ),
    (
        "Functional Category Enrichment Heatmaps",
        "category_enrichment",
        sorted(glob.glob(str(RESULTS_DIR / "hypergeometric_all_regions/category_heatmap_*.png")))
    ),
    (
        "LR Pair Heatmaps by Category",
        "lr_heatmaps",
        sorted(glob.glob(str(RESULTS_DIR / "hypergeometric_all_regions/heatmaps_broad_clustered/*.png")))
    ),
    (
        "Chord Plots by Category",
        "chord_plots",
        sorted(glob.glob(str(RESULTS_DIR / "regression_results/chord_plots_*/grid_*.png")))
    ),
]

# ── COPY IMAGES ───────────────────────────────────────────────────────────────

print("Copying images...")
all_sections_html = []

for title, subfolder, files in SECTIONS:
    if not files:
        print(f"  No files found for: {title}")
        continue

    out_subdir = IMG_DIR / subfolder
    out_subdir.mkdir(exist_ok=True)

    section_images = []
    for fpath in files:
        fname = os.path.basename(fpath)
        dest  = out_subdir / fname
        shutil.copy2(fpath, dest)
        rel   = f"img/{subfolder}/{fname}"
        label = fname.replace('.png','').replace('_',' ')
        section_images.append((rel, label))

    print(f"  {title}: {len(section_images)} images")
    all_sections_html.append((title, subfolder, section_images))

# ── GENERATE HTML ─────────────────────────────────────────────────────────────

print("Generating index.html...")

nav_links = "\n".join(
    f'<a href="#{sf}">{title}</a>'
    for title, sf, imgs in all_sections_html
)

sections_html = ""
for title, sf, images in all_sections_html:
    thumbs = ""
    for rel, label in images:
        thumbs += f"""
        <div class="thumb">
            <a href="{rel}" target="_blank">
                <img src="{rel}" alt="{label}" loading="lazy">
            </a>
            <p>{label}</p>
        </div>"""

    sections_html += f"""
    <section id="{sf}">
        <h2>{title}</h2>
        <div class="grid">{thumbs}
        </div>
    </section>
    <hr>"""

now = datetime.now().strftime("%Y-%m-%d %H:%M")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CCC Pipeline Results</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f5; color: #222; }}
  header {{ background: #1a1a2e; color: white; padding: 1.5rem 2rem;
            position: sticky; top: 0; z-index: 100; }}
  header h1 {{ font-size: 1.4rem; font-weight: 500; }}
  header p  {{ font-size: 0.85rem; color: #aaa; margin-top: 0.25rem; }}
  nav {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }}
  nav a {{ color: #7eb8f7; text-decoration: none; font-size: 0.85rem;
           padding: 0.2rem 0.6rem; border: 1px solid #7eb8f7;
           border-radius: 4px; }}
  nav a:hover {{ background: #7eb8f7; color: #1a1a2e; }}
  main {{ max-width: 1600px; margin: 0 auto; padding: 2rem; }}
  section {{ margin-bottom: 2rem; }}
  h2 {{ font-size: 1.2rem; font-weight: 500; margin-bottom: 1rem;
        padding-bottom: 0.5rem; border-bottom: 2px solid #1a1a2e; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
  .thumb {{ background: white; border-radius: 8px; padding: 0.75rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1); width: 280px; }}
  .thumb img {{ width: 100%; height: 200px; object-fit: contain;
                border-radius: 4px; display: block; }}
  .thumb p {{ font-size: 0.72rem; color: #555; margin-top: 0.5rem;
              word-break: break-word; line-height: 1.4; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 1.5rem 0; }}
  footer {{ text-align: center; padding: 2rem; color: #888; font-size: 0.8rem; }}
</style>
</head>
<body>
<header>
  <h1>CCC Pipeline Results — Brain Aging</h1>
  <p>Last updated: {now} &nbsp;|&nbsp; 11 regions · 21 functional categories · 2,909 LR pairs</p>
  <nav>{nav_links}</nav>
</header>
<main>
{sections_html}
</main>
<footer>
  Generated by generate_gallery.py &nbsp;|&nbsp; easmit31 / cell-communication
</footer>
</body>
</html>"""

(DOCS_DIR / "index.html").write_text(html)
print(f"✓ Saved: {DOCS_DIR / 'index.html'}")
print(f"\nNext steps:")
print(f"  cd {REPO_DIR}")
print(f"  git add docs/")
print(f"  git commit -m 'Update results gallery'")
print(f"  git push")
print(f"\nThen enable GitHub Pages in repo Settings → Pages → Source: docs/")
