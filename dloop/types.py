from typing import Any, Callable, Dict, Optional, TypeVar, Union

# Type definition for the loop state
LoopState = Dict[str, Any]

# Type definition for condition functions
# A function that takes a LoopState and returns a boolean
ConditionFunction = Callable[[LoopState], bool]