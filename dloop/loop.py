import time
import json
from typing import Dict, Any, Optional, ClassVar

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
    pass