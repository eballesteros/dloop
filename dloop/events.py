import time
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
    def __init__(
        self,
        condition_function=None,
        every_n_steps=None,
        at_step=None,
        every_n_seconds=None,
        at_time=None,
    ):
        """
        Initialize an event with a triggering condition.

        Args:
            condition_function (callable, optional): Custom function that determines when
                event triggers
            every_n_steps (int, optional): Trigger every N steps
            at_step (int, optional): Trigger at a specific step (once)
            every_n_seconds (float, optional): Trigger every N seconds of training
            at_time (float, optional): Trigger once when training reaches this time in seconds
        """
        self._condition_functions = []
        self._time_conditions = {}

        # Track time-based event state
        self._start_time = time.time()
        self._last_triggered_time = self._start_time
        self._at_time_triggered = False

        if condition_function is not None:
            self._condition_functions.append(condition_function)

        if every_n_steps is not None:
            self._condition_functions.append(partial(_every_n_steps, n_steps=every_n_steps))

        if at_step is not None:
            self._condition_functions.append(partial(_at_step, step=at_step))

        if every_n_seconds is not None:
            self._time_conditions["every_n_seconds"] = every_n_seconds

        if at_time is not None:
            self._time_conditions["at_time"] = at_time

    def should_trigger(self, loop_state) -> bool:
        """
        Determine if the event should trigger based on current loop state.

        Args:
            loop_state: Current LoopState instance

        Returns:
            bool: True if the event should trigger, False otherwise
        """
        # Check regular conditions
        if any(cf(loop_state) for cf in self._condition_functions):
            return True

        # Check time-based conditions
        current_time = time.time()

        # Check every_n_seconds condition
        if "every_n_seconds" in self._time_conditions:
            interval = self._time_conditions["every_n_seconds"]
            if current_time - self._last_triggered_time >= interval:
                self._last_triggered_time = current_time
                return True

        # Check at_time condition (triggers once)
        if "at_time" in self._time_conditions and not self._at_time_triggered:
            target_time = self._time_conditions["at_time"]
            if current_time - self._start_time >= target_time:
                self._at_time_triggered = True
                return True

        return False
