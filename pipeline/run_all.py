"""
Production entry point — run the full extraction on ANY image in one process
(models loaded once). Generalized; no image-specific logic.

  python -m pipeline.run_all --image path/to/image.png --out pipeline/_out/layers
  # then deploy:  cp <out>/* editable-editor/public/layers/
"""

import argparse
import json
import os

from . import config as C
from . import engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=C.IMAGE_PATH)
    ap.add_argument("--out", default=os.path.join(C.ROOT, "pipeline", "_out", "layers"))
    args = ap.parse_args()

    engine.load_all()
    summary = engine.process_image(args.image, args.out)
    print(json.dumps(summary, indent=1))
    print(f"\nlayers + metadata -> {args.out}")
    print(f"deploy with:  cp {args.out}/* editable-editor/public/layers/")


if __name__ == "__main__":
    main()
