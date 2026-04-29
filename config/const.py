from dotenv import load_dotenv
import os

load_dotenv()

MAX_EPISODE_STEPS = int(os.getenv("MAX_EPISODE_STEPS"))
MAX_ACT_TIMEOUT = int(os.getenv("MAX_ACT_TIMEOUT"))
MAX_SHIP_SPEED = float(os.getenv("MAX_SHIP_SPEED"))
SUM_RADIUS = float(os.getenv("SUM_RADIUS"))
BOARD_SIZE = float(os.getenv("BOARD_SIZE"))
COMET_SPEED = float(os.getenv("COMET_SPEED"))