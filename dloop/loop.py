import time
import json
from typing import Dict, Any, Optional, Iterable, Union, Iterator, ClassVar

class LoopEvents:
    """Standard events for the training loop."""
    pass

class LoopState:
    def __init__(self, 
                 current_epoch: int = 0,
                 global_step: int = 0,
                 last_epoch_change_step: int = 0,
                 start_time: Optional[float] = None):
        """
        Initialize loop state tracking.
        
        Args:
            current_epoch: Current epoch number (default: 0)
            global_step: Current global step (default: 0)
            last_epoch_change_step: Step at which the last epoch change occurred (default: 0)
            start_time: Time when training started in seconds since epoch (default: current time)
        """
        self.current_epoch = current_epoch
        self.global_step = global_step
        self._last_epoch_change_step = last_epoch_change_step
        self.start_time = time.time() if start_time is None else start_time
        
    @property
    def local_epoch_step(self):
        """Steps in current epoch."""
        return self.global_step - self._last_epoch_change_step
        
    @property
    def elapsed_time(self):
        """Time elapsed since loop start in seconds."""
        return time.time() - self.start_time
    
    def increment_epoch(self):
        """
        Increment the epoch counter and update epoch change tracking.
        """
        self.current_epoch += 1
        self._last_epoch_change_step = self.global_step
    
    def increment_step(self):
        """
        Increment the global step counter.
        """
        self.global_step += 1
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary for serialization.
        
        Returns:
            Dictionary containing all state information
        """
        return {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "last_epoch_change_step": self._last_epoch_change_step,
            "start_time": self.start_time,
            "serialized_at": time.time()
        }
        
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> 'LoopState':
        """
        Create state from dictionary when resuming.
        
        Args:
            state_dict: Dictionary containing state information
            
        Returns:
            Reconstructed LoopState object
        """
        return cls(
            current_epoch=state_dict["current_epoch"],
            global_step=state_dict["global_step"],
            last_epoch_change_step=state_dict["last_epoch_change_step"],
            start_time=state_dict["start_time"]
        )



class Loop:
    """Main loop class that manages the training loop and events."""
    
    def __init__(self, 
                 dataloader: Iterable, 
                 events: Optional[Dict[Any, Any]] = None, 
                 max_epochs: Optional[int] = None, 
                 max_steps: Optional[int] = None,
                 state_file: Optional[str] = None):
        """
        Initialize the loop.
        
        Args:
            dataloader: DataLoader providing batches
            events: Dictionary mapping event keys to Event instances
            max_epochs: Maximum number of epochs
            max_steps: Maximum number of steps
            state_file: Path to save/load loop state
            
        Raises:
            ValueError: If no stopping condition (max_epochs or max_steps) is provided
        """
        self.dataloader = dataloader
        self.events = events or {}
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.state_file = state_file
        
        # Ensure at least one stopping condition is provided
        if self.max_epochs is None and self.max_steps is None:
            raise ValueError("At least one stopping condition (max_epochs or max_steps) must be provided")
        
        # Initialize state
        self.state = LoopState()
        
        # Will hold the dataloader iterator
        self._iterator = None
        
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
        """Iterator interface.
        
        Returns:
            self: The Loop instance
        """
        # Initialize the iterator from dataloader
        self._iterator = iter(self.dataloader)
        return self
        
    def __next__(self):
        """Get next batch.
        
        Returns:
            The next batch from the dataloader
            
        Raises:
            StopIteration: When iteration should end
        """
        # Check termination conditions
        if self.max_epochs is not None and self.state.current_epoch >= self.max_epochs:
            raise StopIteration("Reached maximum number of epochs")
            
        if self.max_steps is not None and self.state.global_step >= self.max_steps:
            raise StopIteration("Reached maximum number of steps")
        
        try:
            # Try to get next batch from current epoch
            batch = next(self._iterator)
            
            # Update step counter
            self.state.increment_step()
            
            return batch
            
        except StopIteration:
            # End of epoch reached, increment epoch counter
            self.state.increment_epoch()
            
            # Check if we've reached max_epochs after incrementing
            if self.max_epochs is not None and self.state.current_epoch >= self.max_epochs:
                raise StopIteration("Reached maximum number of epochs")
            
            # Start a new iterator for the next epoch and continue
            self._iterator = iter(self.dataloader)
            
            # Get the first batch of the new epoch
            batch = next(self._iterator)
            
            # Update step counter for this new batch too
            self.state.increment_step()
            
            return batch