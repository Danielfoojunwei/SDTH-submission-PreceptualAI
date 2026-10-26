"""Build one self-contained Studio page for publishing as an Artifact.

    python3 studio/build_artifact.py

Inlines styles.css and app.js and embeds data/bundle.json, writing
studio/dist/wirelessjev-studio.html. The publish skeleton supplies doctype,
head and body, so the output starts at <title>.
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
html = (HERE / "index.html").read_text()
css = (HERE / "styles.css").read_text()
js = (HERE / "app.js").read_text()
bundle = (HERE / "data" / "bundle.json").read_text().replace("</", "<\\/")

body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
body = body.replace('<script type="module" src="app.js"></script>', "")
page = (
    "<title>WirelessJEV Studio</title>\n"
    '<meta name="claude:viewport-fit-sides" content="cover">\n'
    f"<style>\n{css}\n</style>\n"
    f"{body.strip()}\n"
    f'<script type="application/json" id="bundle">{bundle}</script>\n'
    f'<script type="module">\n{js}\n</script>\n'
)
out = HERE / "dist" / "wirelessjev-studio.html"
out.parent.mkdir(exist_ok=True)
out.write_text(page)
print(out, f"{len(page) / 1e6:.2f} MB")
