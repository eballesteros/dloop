import importlib.metadata

# Import key classes
from .events import Event, LoopEvents
from .loop import Loop
from .types import LoopState

# figure out version dynamically
__version__ = importlib.metadata.version("dloop")

# Define public API
__all__ = ["Event", "LoopEvents", "Loop", "LoopState"]
