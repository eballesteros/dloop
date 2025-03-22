from enum import Enum, auto, unique
from functools import partial

from .types import LoopState


@unique
class LoopEvents(Enum):
    """
    Standard events for the training loop.

    This enum defines built-in events that can be used with Event handlers.

    Note: To create custom events, users should define their own separate Enum classes
    rather than attempting to extend this one. Due to limitations in Python's Enum
    implementation, inheritance-based extension is not supported.

    Example:
        ```python
        class MyCustomEvents(Enum):
            VALIDATION = auto()
            CHECKPOINT = auto()
        ```
    """

    EXCEPTION = auto()  # Triggered when any exception occurs
    EPOCH_END = auto()  # Triggered at the end of each epoch
    TRAINING_END = auto()  # Triggered at the end of training


def _every_n_steps(loop_state: LoopState, n_steps: int) -> bool:
    return (loop_state.epoch_step + 1) % n_steps == 0


def _at_step(loop_state: LoopState, step: int) -> bool:
    return loop_state.global_step == step


class Event:
    def __init__(self, condition_function=None, every_n_steps=None, at_step=None):
        """
        Initialize an event with a triggering condition.

        Args:
            condition_func (callable, optional): Custom function that determines when event triggers
            every_n_steps (int, optional): Trigger every N steps
            at_step (int, optional): Trigger at a specific step (once)
        """
        self._condition_functions = []

        if condition_function is not None:
            self._condition_functions.append(condition_function)

        if every_n_steps is not None:
            self._condition_functions.append(partial(_every_n_steps, n_steps=every_n_steps))

        if at_step is not None:
            self._condition_functions.append(partial(_at_step, step=at_step))

    def should_trigger(self, loop_state) -> bool:
        """
        Determine if the event should trigger based on current loop state.

        Args:
            loop_state: Current LoopState instance

        Returns:
            bool: True if the event should trigger, False otherwise
        """
        return any(cf(loop_state) for cf in self._condition_functions)
