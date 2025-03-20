import pytest
from enum import Enum, auto
from dloop.loop import LoopState
from dloop.events import Event, LoopEvents

def test_event_init():
    """Test that Event can be instantiated with and without a condition function."""
    # Test without condition function
    event = Event()
    assert event.condition_func is None
    assert event.every_n_steps is None
    assert event.at_step is None
    
    # Test with condition function
    def condition_func(state):
        return True
    
    event = Event(condition_func=condition_func)
    assert event.condition_func is condition_func
    
    # Test with step parameters
    event = Event(every_n_steps=5, at_step=10)
    assert event.every_n_steps == 5
    assert event.at_step == 10
    
def test_every_n_steps():
    """Test that every_n_steps triggers at the right steps."""
    event = Event(every_n_steps=4)
    
    # Should not trigger at step 0
    assert event.should_trigger(LoopState(global_step=0)) is False
    
    # Should trigger at steps 3, 7, 11, etc. (0-indexed)
    assert event.should_trigger(LoopState(global_step=3)) is True
    assert event.should_trigger(LoopState(global_step=7)) is True
    assert event.should_trigger(LoopState(global_step=11)) is True
    
    # Should not trigger at other steps
    assert event.should_trigger(LoopState(global_step=1)) is False
    assert event.should_trigger(LoopState(global_step=2)) is False
    assert event.should_trigger(LoopState(global_step=4)) is False
    assert event.should_trigger(LoopState(global_step=5)) is False
    assert event.should_trigger(LoopState(global_step=6)) is False
    
def test_at_step():
    """Test that at_step triggers only at the specified step."""
    event = Event(at_step=7)
    
    # Should not trigger at step 0
    assert event.should_trigger(LoopState(global_step=0)) is False
    
    # Should trigger only at step 7
    assert event.should_trigger(LoopState(global_step=7)) is True
    
    # Should not trigger at other steps
    assert event.should_trigger(LoopState(global_step=1)) is False
    assert event.should_trigger(LoopState(global_step=6)) is False
    assert event.should_trigger(LoopState(global_step=8)) is False
    
def test_multiple_conditions():
    """Test that multiple conditions work together."""
    # Event with both every_n_steps and at_step
    event = Event(every_n_steps=4, at_step=5)
    
    # Should trigger at steps 3, 5, 7, 11 (steps 3 and 7 from every_n_steps=4, step 5 from at_step)
    assert event.should_trigger(LoopState(global_step=3)) is True
    assert event.should_trigger(LoopState(global_step=5)) is True
    assert event.should_trigger(LoopState(global_step=7)) is True
    assert event.should_trigger(LoopState(global_step=11)) is True
    
    # Should not trigger at other steps
    assert event.should_trigger(LoopState(global_step=0)) is False
    assert event.should_trigger(LoopState(global_step=1)) is False
    assert event.should_trigger(LoopState(global_step=2)) is False
    assert event.should_trigger(LoopState(global_step=4)) is False
    assert event.should_trigger(LoopState(global_step=6)) is False
    
def test_condition_func():
    """Test that custom condition function works."""
    # Event with condition_func that triggers on even steps
    def even_steps(state):
        return state.global_step % 2 == 0
    
    event = Event(condition_func=even_steps)
    
    # Should trigger at even steps (except 0)
    assert event.should_trigger(LoopState(global_step=2)) is True
    assert event.should_trigger(LoopState(global_step=4)) is True
    assert event.should_trigger(LoopState(global_step=6)) is True
    
    # Should not trigger at odd steps
    assert event.should_trigger(LoopState(global_step=1)) is False
    assert event.should_trigger(LoopState(global_step=3)) is False
    assert event.should_trigger(LoopState(global_step=5)) is False
    
def test_loop_events_enum():
    """Test that LoopEvents enum can be accessed."""
    # Check that the built-in events exist
    assert hasattr(LoopEvents, "EXCEPTION")
    assert hasattr(LoopEvents, "EPOCH_END")
    
    # Check values are different
    assert LoopEvents.EXCEPTION != LoopEvents.EPOCH_END
    
    # Check enum works as expected
    assert isinstance(LoopEvents.EXCEPTION, LoopEvents)
    assert isinstance(LoopEvents.EPOCH_END, LoopEvents)

def test_custom_events():
    """Test creating custom event enums separate from LoopEvents."""
    # Create a custom events enum
    class TrainingEvents(Enum):
        LOGGING = auto()
        OPTIMIZER_STEP = auto()
    
    # Check that the custom events exist
    assert hasattr(TrainingEvents, "LOGGING")
    assert hasattr(TrainingEvents, "OPTIMIZER_STEP")
    
    # Check that built-in events still exist separately
    assert hasattr(LoopEvents, "EXCEPTION")
    assert hasattr(LoopEvents, "EPOCH_END")
    
    # Verify each enum maintains its own namespace
    assert not hasattr(TrainingEvents, "EXCEPTION")
    assert not hasattr(LoopEvents, "LOGGING")
    
    # Check that custom event values don't conflict with built-in ones
    assert TrainingEvents.LOGGING not in list(LoopEvents)
    assert LoopEvents.EXCEPTION not in list(TrainingEvents)

def test_loop_events_as_dict_keys():
    """Test using events as dictionary keys."""
    # Create a dictionary with events as keys
    handlers = {
        LoopEvents.EXCEPTION: "exception_handler",
        LoopEvents.EPOCH_END: "epoch_end_handler"
    }
    
    # Test dictionary access with events
    assert handlers[LoopEvents.EXCEPTION] == "exception_handler"
    assert handlers[LoopEvents.EPOCH_END] == "epoch_end_handler"
    
    # Test with custom events in same dictionary
    class TrainingEvents(Enum):
        VALIDATION = auto()
        LOGGING = auto()
    
    # Update dictionary with custom event
    handlers[TrainingEvents.VALIDATION] = "validation_handler"
    handlers[TrainingEvents.LOGGING] = "logging_handler"
    
    # Check event handlers can be accessed with their respective enum values
    assert handlers[TrainingEvents.VALIDATION] == "validation_handler"
    assert handlers[TrainingEvents.LOGGING] == "logging_handler"
    assert handlers[LoopEvents.EXCEPTION] == "exception_handler"