"""Pytest configuration.

Force the test suite onto CPU (fp32) regardless of host hardware. The unit tests validate
logic, not GPU performance, and the device-aware precision policy means a visible GPU would
make models bf16/cuda while test inputs are built on CPU — causing device/dtype mismatches.
Hiding CUDA here keeps tests deterministic and green on both laptops and Colab GPU boxes.

Must run before torch first queries CUDA, so it lives at import time in conftest.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
