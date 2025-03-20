class LoopEvents:
    """Standard events for the training loop."""
    pass

class LoopState:
    def __init__(self):
        """Initialize loop state tracking."""
        self.current_epoch = 0
        self.global_step = 0
        self._last_epoch_change_step = 0
        
    @property
    def local_epoch_step(self):
        """Steps in current epoch."""
        return self.global_step - self._last_epoch_change_step
    
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


class Loop:
    """Main loop class that manages the training loop and events."""
    pass