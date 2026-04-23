from __future__ import annotations

import sys

from train_casme3_flow import main


if __name__ == "__main__":
    sys.argv = ["train_casme3_flow.py", "--config", "configs/casme3_flow_debug.toml"]
    main()
