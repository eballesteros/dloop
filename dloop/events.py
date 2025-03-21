from enum import Enum, auto, unique

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
    TRAINING_END = auto()  # Triggered at the end training


class Event:
    def __init__(self, condition_func=None, every_n_steps=None, at_step=None):
        """
        Initialize an event with a triggering condition.
        
        Args:
            condition_func (callable, optional): Custom function that determines when event triggers
            every_n_steps (int, optional): Trigger every N steps
            at_step (int, optional): Trigger at a specific step (once)
        """
        self.condition_func = condition_func
        self.every_n_steps = every_n_steps
        self.at_step = at_step
    
    def should_trigger(self, loop_state):
        """
        Determine if the event should trigger based on current loop state.
        
        Args:
            loop_state: Current LoopState instance
            
        Returns:
            bool: True if the event should trigger, False otherwise
        """
        # Get the current step from loop state
        current_step = loop_state.epoch_step
        
        # Check every_n_steps condition (0-indexed, triggers at steps 3, 7, 11, etc. for every_n_steps=4)
        if self.every_n_steps is not None:
            if (current_step + 1) % self.every_n_steps == 0:
                return True
        
        # Check at_step condition (0-indexed)
        if self.at_step is not None:
            if current_step == self.at_step:
                return True
        
        # Check custom condition function if provided
        if self.condition_func is not None:
            return self.condition_func(loop_state)
                
        return False