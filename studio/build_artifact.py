"""Build one self-contained Studio page for publishing as an Artifact.

    python3 studio/build_artifact.py

Inlines styles.css and app.js and embeds data/evidence.json, writing
studio/dist/wirelessjev-studio.html. The publish skeleton supplies doctype,
head and body, so the output starts at <title>.
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
html = (HERE / "index.html").read_text()
css = (HERE / "styles.css").read_text()
js = (HERE / "app.js").read_text()
evidence = HERE / "data" / "evidence.json"
if evidence.exists():
    bundle = evidence.read_text().replace("</", "<\\/")
else:                       # the page then fetches data/evidence.json when opened next to it
    bundle = None
    print(f"note: {evidence} not found; run studio/build_bundle.py first to embed the data")

body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
body = body.replace('<script type="module" src="app.js"></script>', "")
page = (
    '<meta charset="utf-8">\n'
    "<title>WirelessJEV Studio</title>\n"
    '<meta name="claude:viewport-fit-sides" content="cover">\n'
    f"<style>\n{css}\n</style>\n"
    f"{body.strip()}\n"
    + (f'<script type="application/json" id="bundle">{bundle}</script>\n' if bundle else "")
    + f'<script type="module">\n{js}\n</script>\n'
)
out = HERE / "dist" / "wirelessjev-studio.html"
out.parent.mkdir(exist_ok=True)
out.write_text(page)
print(out, f"{len(page) / 1e6:.2f} MB")
