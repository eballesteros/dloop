from dataclasses import dataclass


from typing import Any, Callable, Dict

# Type definition for the loop state
@dataclass
class LoopState:
    epoch: int
    global_step: int
    epoch_step: int
    epoch_end: bool
    training_end: bool

# Type definition for condition functions
# A function that takes a LoopState and returns a boolean
ConditionFunction = Callable[[LoopState], bool]