# Benchmark images

Drop any AI-generated images here (png/jpg/webp) and run `python -m pipeline.benchmark`
from the repo root. The harness is category-agnostic — it computes no-reference
quality metrics, so no labels/ground-truth are needed and any category works
(portraits, posters, product, fashion, vehicles, architecture, interiors, logos,
anime, landscapes, illustrations, book covers, marketing creatives, …).

Committed here:
- `poster.png`, `logo.png`, `landscape.png` — synthetic robustness fixtures
  (verify the pipeline runs + degrades gracefully on non-book content).
- `hero.png` — a small real graphic.

Not committed (large): drop your own real AI images (e.g. a book cover, a
portrait, a product shot). `bookcover.png` (the original benchmark sample) is
gitignored to keep the repo lean; copy `uploads/test.png` here to include it.
