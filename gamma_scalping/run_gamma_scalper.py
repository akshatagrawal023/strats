import os
import sys
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gamma_scalping.gamma_scalper import GammaScalper
try:
    from config_live import Config
except Exception:
    # Ensure the Live Simulation directory is importable
    live_sim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Live Simulation')
    sys.path.append(os.path.abspath(live_sim_dir))
    from config_live import Config


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    scalper = GammaScalper(Config)
    # Example: NIFTY weekly syntax must match your option_symbols formatting for expiry
    base = getattr(Config, 'GAMMA_BASE', 'NIFTY')
    expiry = getattr(Config, 'GAMMA_EXPIRY', '25AUG')  # e.g., '25AUG'
    lots = getattr(Config, 'GAMMA_LOTS', 1)

    scalper.manage(base=base, expiry=expiry, lots=lots)


if __name__ == "__main__":
    main()

