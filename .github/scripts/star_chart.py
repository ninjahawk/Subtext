"""Generate a star-history SVG chart for a repo (light + dark themes).

Fetches stargazer timestamps from the GitHub API (requires a token with
Metadata read access — since June 2026 the stargazers endpoint is limited
to repo admins/collaborators) and renders a cumulative line chart as
static SVG, dependency-free.

Env: GITHUB_TOKEN, REPO ("owner/name"), OUT_DIR (default "media").
"""

import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timezone

REPO = os.environ["REPO"]
TOKEN = os.environ["GITHUB_TOKEN"]
OUT_DIR = os.environ.get("OUT_DIR", "media")

API = f"https://api.github.com/repos/{REPO}/stargazers"


def fetch_star_dates():
    dates, page = [], 1
    while page <= 400:  # 40k stars; the API caps the listing there anyway
        req = urllib.request.Request(
            f"{API}?per_page=100&page={page}",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Accept": "application/vnd.github.star+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(req) as r:
            batch = json.load(r)
        dates += [
            datetime.fromisoformat(s["starred_at"].replace("Z", "+00:00"))
            for s in batch
        ]
        if len(batch) < 100:
            break
        page += 1
    return sorted(dates)


def nice_ceil(v):
    if v <= 10:
        return 10
    mag = 10 ** int(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * mag:
            return int(m * mag)
    return int(10 * mag)


def render(dates, theme):
    w, h = 800, 533
    pad_l, pad_r, pad_t, pad_b = 70, 40, 70, 60
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b

    now = datetime.now(timezone.utc)
    t0, t1 = dates[0].timestamp(), now.timestamp()
    if t1 - t0 < 3600:
        t1 = t0 + 3600
    y_max = nice_ceil(len(dates))

    def x(t):
        return pad_l + (t - t0) / (t1 - t0) * pw

    def y(n):
        return pad_t + ph - n / y_max * ph

    dark = theme == "dark"
    bg = "#0d1117" if dark else "#ffffff"
    fg = "#c9d1d9" if dark else "#24292f"
    grid = "#30363d" if dark else "#d0d7de"
    line = "#e8a13c"

    pts = [(x(d.timestamp()), y(i + 1)) for i, d in enumerate(dates)]
    pts.append((x(t1), y(len(dates))))
    path = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)

    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
        f'<rect width="{w}" height="{h}" fill="{bg}" rx="8"/>',
        f'<text x="{w / 2}" y="38" text-anchor="middle" fill="{fg}" '
        f'font-size="22" font-weight="600">Star History — {REPO}</text>',
    ]
    for i in range(6):  # horizontal grid + y labels
        n = y_max * i / 5
        gy = y(n)
        s.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w - pad_r}" y2="{gy:.1f}" '
            f'stroke="{grid}" stroke-width="1"/>'
        )
        s.append(
            f'<text x="{pad_l - 10}" y="{gy + 4:.1f}" text-anchor="end" '
            f'fill="{fg}" font-size="13">{int(n)}</text>'
        )
    for i in range(5):  # x labels
        t = t0 + (t1 - t0) * i / 4
        label = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%b %d")
        s.append(
            f'<text x="{x(t):.1f}" y="{h - pad_b + 24}" text-anchor="middle" '
            f'fill="{fg}" font-size="13">{label}</text>'
        )
    s.append(
        f'<path d="{path}" fill="none" stroke="{line}" stroke-width="2.5" '
        f'stroke-linejoin="round"/>'
    )
    if len(pts) <= 60:
        for px, py in pts[:-1]:
            s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{line}"/>')
    lx, ly = pts[-1]
    s.append(
        f'<text x="{min(lx, w - pad_r) - 6:.1f}" y="{ly - 10:.1f}" text-anchor="end" '
        f'fill="{fg}" font-size="14" font-weight="600">{len(dates)} stars</text>'
    )
    s.append(
        f'<text x="{w - pad_r}" y="{h - 16}" text-anchor="end" fill="{grid}" '
        f'font-size="11">updated {now.strftime("%Y-%m-%d %H:%M UTC")}</text>'
    )
    s.append("</svg>")
    return "\n".join(s)


def main():
    dates = fetch_star_dates()
    if not dates:
        print("no stargazers found", file=sys.stderr)
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    for theme, name in (("light", "star-history.svg"), ("dark", "star-history-dark.svg")):
        out = os.path.join(OUT_DIR, name)
        with open(out, "w", encoding="utf-8") as f:
            f.write(render(dates, theme))
        print(f"wrote {out} ({len(dates)} stars)")


if __name__ == "__main__":
    main()
