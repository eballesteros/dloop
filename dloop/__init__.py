__version__ = "0.1.0"

# Import key classes
from .events import Event, LoopEvents
from .loop import Loop

# Import internal classes (not exposed in __all__)
from .loop import LoopState
