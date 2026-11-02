# Strong Again — landing page

Static landing page (plain HTML, no build step), deployed on Netlify.

- `index.html` — the page (dark-mode toggle is a small inline script)
- `assets/` — preview video and poster image
- `favicon.svg` — logo mark
- `netlify.toml` — publishes the repo root; long-cache headers for `/assets/*`

Run locally: `python3 -m http.server` and open http://localhost:8000.

Note: buttons that point to the app ("Start free trial", "Sign in", feature cards)
currently scroll to the closing call-to-action (`#get-started`) until the app URL exists.
