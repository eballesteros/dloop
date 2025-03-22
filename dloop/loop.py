import collections.abc
from collections.abc import Iterable
from typing import Any, Optional

from .events import Event
from .iter_logic import NoLenIterationStrategy, get_iter_dl_with_events


class Loop:
    """Main loop class that manages the training loop and events."""

    def __init__(
        self,
        dataloader: Iterable,
        events: Optional[dict[Any, Event]] = None,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        state_file: Optional[str] = None,
        dataloader_len: Optional[int] = None,
        no_len_iteration_strategy: NoLenIterationStrategy = "pairwise",
    ):
        """
        Initialize the loop.

        Args:
            dataloader: DataLoader providing batches
            events: Dictionary mapping event keys to Event instances
            max_epochs: Maximum number of epochs
            max_steps: Maximum number of steps
            state_file: Path to save/load loop state
            dataloader_len: length of the dataloader. If not provided, will try to be inferred
                with len(dataloader)
            no_len_iteration_strategy: Iteration strategy if the length of the dataloader is not
                provided and cannot be inferred

        Raises:
            ValueError: If no stopping condition (max_epochs or max_steps) is provided
        """
        self.dataloader = dataloader
        self.events = events or {}
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.state_file = state_file

        # try to infer if not provided
        dl_len = dataloader_len or (
            len(self.dataloader) if isinstance(self.dataloader, collections.abc.Sized) else None
        )

        # Ensure at least one stopping condition is provided
        if self.max_epochs is None and self.max_steps is None:
            raise ValueError(
                "At least one stopping condition (max_epochs or max_steps) must be provided"
            )

        # Will hold the dataloader iterator
        self._iterator = get_iter_dl_with_events(
            self.dataloader,
            dl_len=dl_len,
            max_epochs=max_epochs,
            max_steps=max_steps,
            no_len_iteration_strategy=no_len_iteration_strategy,
            events=events,
        )

    def __enter__(self):
        """
        Context manager enter method.

        Returns:
            self: The Loop instance
        """
        # Will later handle state loading
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        Returns:
            bool: True to suppress exceptions, False otherwise
        """
        # Will later handle exception catching and state saving
        return False  # Don't suppress exceptions for now

    def __iter__(self):
        yield from self._iterator
