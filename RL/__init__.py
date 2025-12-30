import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

from .dreamerv3.agent import Agent
from .tdmpc2.agent import TDMPC2Agent
