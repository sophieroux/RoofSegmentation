"""Write the five masks from final.pt. Same entry point, with training skipped."""

from __future__ import annotations

import sys

from roofseg.execute_experiment import main

if __name__ == "__main__":
    if "--predict-only" not in sys.argv:
        sys.argv[1:1] = ["--predict-only", "outputs/checkpoints/final.pt"]
    main()
