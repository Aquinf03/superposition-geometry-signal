# spg docs (GitHub Pages)

Public docs for the Superposition Geometry SDK and CLI.

```
docs/
  documentation/   # product docs
  assets/          # shared chrome (CSS, logo, favicon)
  index.html       # redirects → documentation/
```

**Edit under `documentation/` directly.** SDK-first: `pip install -e .` then `from spg import Tracker, Diff, plot, GeoControl` or `spg …`.

## GitHub Pages

Workflow: [`.github/workflows/deploy-docs.yml`](../.github/workflows/deploy-docs.yml)

In the repo: **Settings → Pages → Source → GitHub Actions** (not “Deploy from a branch”). The workflow already sets `permissions: pages: write` and `id-token: write`, which `actions/deploy-pages` needs.

Live URL (once enabled): `https://aquinf03.github.io/superposition-geometry-signal/`

- Docs home: [`documentation/`](documentation/)
- Paper draft: [`../paper/`](../paper/)
- Thesis: [`../THESIS.md`](../THESIS.md)
