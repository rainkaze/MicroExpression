from __future__ import annotations

import sys

from train_casme3_apex import main


if __name__ == "__main__":
    sys.argv = ["train_casme3_apex.py", "--config", "configs/casme3_apex_debug.toml"]
    main()
