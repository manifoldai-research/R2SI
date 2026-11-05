"""Trial-bound launcher for the updated Harness."""
from pathlib import Path
import sys
checkout = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(checkout.parent), str(checkout)]
from XPolicyLab.policy.WorldScape_R2SI.harness.server import main
if __name__ == "__main__":
    main()
