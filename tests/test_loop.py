import pytest
import time
import json
from collections import deque
from dloop.loop import LoopState, Loop
from dloop.events import Event, LoopEvents

def test_loop_state_init():
    """Test LoopState initialization with default values."""
    state = LoopState()
    
    # Check initial values
    assert state.current_epoch == 0
    assert state.global_step == 0
    assert state.epoch_step == 0
    
    # Verify start_time is a recent timestamp
    current_time = time.time()
    assert state.start_time <= current_time
    assert state.start_time > current_time - 10  # Within last 10 seconds

def test_loop_state_epoch_step():
    """Test epoch_step calculation."""
    state = LoopState()
    
    # Should start at 0
    assert state.epoch_step == 0
    assert state.global_step == 0
    
    # Increment steps and check epoch_step
    state.increment_step()
    assert state.global_step == 1
    assert state.epoch_step == 1
    
    state.increment_step()
    assert state.global_step == 2
    assert state.epoch_step == 2
    
    # Increment epoch and check epoch_step resets
    state.increment_epoch()
    assert state.current_epoch == 1
    assert state.global_step == 2
    assert state.epoch_step == 0
    
    # Add more steps in the new epoch
    state.increment_step()
    assert state.global_step == 3
    assert state.epoch_step == 1

def test_loop_state_epoch_transitions():
    """Test epoch transitions and step counting."""
    state = LoopState()
    
    # Simulate first epoch
    for _ in range(5):
        state.increment_step()
    
    assert state.current_epoch == 0
    assert state.global_step == 5
    assert state.epoch_step == 5
    
    # Transition to next epoch
    state.increment_epoch()
    
    assert state.current_epoch == 1
    assert state.global_step == 5
    assert state.epoch_step == 0
    
    # Simulate second epoch
    for _ in range(3):
        state.increment_step()
    
    assert state.current_epoch == 1
    assert state.global_step == 8
    assert state.epoch_step == 3

def test_elapsed_time():
    """Test elapsed_time calculation."""
    state = LoopState()
    
    # Elapsed time should start near zero
    initial_elapsed = state.elapsed_time
    assert initial_elapsed >= 0
    assert initial_elapsed < 0.1  # Should be very small initially
    
    # Sleep briefly and check that elapsed time increases
    time.sleep(0.1)
    assert state.elapsed_time > initial_elapsed
    
    # Sleep again and verify elapsed time continues to increase
    previous_elapsed = state.elapsed_time
    time.sleep(0.1)
    assert state.elapsed_time > previous_elapsed

def test_loop_state_to_dict():
    """Test converting state to dictionary."""
    # Initialize with custom values
    state = LoopState(
        current_epoch=3,
        global_step=42,
        last_epoch_change_step=30,
        start_time=1000.0  # Fixed timestamp for testing
    )
    
    # Convert to dictionary
    state_dict = state.to_dict()
    
    # Check that all expected keys are present
    assert "current_epoch" in state_dict
    assert "global_step" in state_dict
    assert "last_epoch_change_step" in state_dict
    assert "start_time" in state_dict
    assert "serialized_at" in state_dict
    
    # Check values
    assert state_dict["current_epoch"] == 3
    assert state_dict["global_step"] == 42
    assert state_dict["last_epoch_change_step"] == 30
    assert state_dict["start_time"] == 1000.0
    
    # serialized_at should be current time
    assert state_dict["serialized_at"] >= time.time() - 1
    assert state_dict["serialized_at"] <= time.time() + 1

def test_loop_state_from_dict():
    """Test recreating state from dictionary."""
    # Create a state dictionary
    state_dict = {
        "current_epoch": 5,
        "global_step": 100,
        "last_epoch_change_step": 80,
        "start_time": 2000.0,
        "serialized_at": time.time()  # Not used by from_dict
    }
    
    # Create state from dictionary
    state = LoopState.from_dict(state_dict)
    
    # Check values
    assert state.current_epoch == 5
    assert state.global_step == 100
    assert state._last_epoch_change_step == 80
    assert state.start_time == 2000.0
    assert state.epoch_step == 20  # 100 - 80

def test_loop_state_round_trip():
    """Test round-trip state → dict → state."""
    # Create original state with non-default values and run a few iterations
    original = LoopState()
    original.increment_step()
    original.increment_step()
    original.increment_epoch()
    original.increment_step()
    
    # Convert to dict
    state_dict = original.to_dict()
    
    # Convert back to state
    reconstructed = LoopState.from_dict(state_dict)
    
    # Check all values match
    assert reconstructed.current_epoch == original.current_epoch
    assert reconstructed.global_step == original.global_step
    assert reconstructed._last_epoch_change_step == original._last_epoch_change_step
    assert reconstructed.start_time == original.start_time
    assert reconstructed.epoch_step == original.epoch_step

def test_json_serialization():
    """Test JSON serialization of state."""
    # Create state with non-default values
    state = LoopState(
        current_epoch=7,
        global_step=150,
        last_epoch_change_step=120
    )
    
    # Convert to dictionary
    state_dict = state.to_dict()
    
    # Serialize to JSON
    json_str = json.dumps(state_dict)
    
    # Deserialize from JSON
    loaded_dict = json.loads(json_str)
    
    # Recreate state
    loaded_state = LoopState.from_dict(loaded_dict)
    
    # Check values match
    assert loaded_state.current_epoch == 7
    assert loaded_state.global_step == 150
    assert loaded_state._last_epoch_change_step == 120
    assert loaded_state.epoch_step == 30

# Simple mock dataloader for testing
class MockDataLoader:
    def __init__(self, batches):
        self.batches = batches
        
    def __iter__(self):
        return iter(self.batches)

# Tests for Loop class
def test_loop_init_basic():
    """Test Loop initialization with basic parameters."""
    # Create a simple dataloader
    dataloader = MockDataLoader([1, 2, 3])
    
    # Create a loop with minimal parameters (including max_epochs as stopping condition)
    loop = Loop(dataloader, max_epochs=1)
    
    # Check dataloader is stored
    assert loop.dataloader is dataloader
    
    # Check defaults and specified values
    assert loop.events == {}
    assert loop.max_epochs == 1
    assert loop.max_steps is None
    assert loop.state_file is None
    
    # Check state is initialized
    assert isinstance(loop._state, LoopState)
    assert loop._state.current_epoch == 0
    assert loop._state.global_step == 0

def test_loop_init_with_params():
    """Test Loop initialization with all parameters."""
    # Create a dataloader
    dataloader = MockDataLoader([1, 2, 3])
    
    # Create events dict
    events = {"event1": "handler1", "event2": "handler2"}
    
    # Create loop with all parameters
    loop = Loop(
        dataloader=dataloader,
        events=events,
        max_epochs=5,
        max_steps=100,
        state_file="/tmp/state.json"
    )
    
    # Check all parameters are stored
    assert loop.dataloader is dataloader
    assert loop.events == events
    assert loop.max_epochs == 5
    assert loop.max_steps == 100
    assert loop.state_file == "/tmp/state.json"
    
    # Check state is initialized
    assert isinstance(loop._state, LoopState)

def test_loop_with_events():
    """Test Loop with event dictionary."""
    # Create a dataloader
    dataloader = MockDataLoader([1, 2, 3])
    
    # Create events with different types of keys
    from enum import Enum, auto
    
    class TestEvents(Enum):
        EVENT1 = auto()
        EVENT2 = auto()
    
    # Create a dictionary with different types of event keys
    events = {
        "string_event": "handler1",
        TestEvents.EVENT1: "handler2",
        123: "handler3"
    }
    
    # Create loop with events and a stopping condition
    loop = Loop(dataloader=dataloader, events=events, max_steps=10)
    
    # Check events are stored correctly
    assert loop.events == events
    assert loop.events["string_event"] == "handler1"
    assert loop.events[TestEvents.EVENT1] == "handler2"
    assert loop.events[123] == "handler3"

def test_loop_with_deque():
    """Test Loop with a deque as dataloader."""
    # Create a deque as dataloader
    dataloader = deque([1, 2, 3])
    
    # Create loop with a stopping condition
    loop = Loop(dataloader, max_epochs=1)
    
    # Check dataloader is stored
    assert loop.dataloader is dataloader

def test_loop_context_manager():
    """Test Loop as a context manager."""
    # Create a dataloader
    dataloader = MockDataLoader([1, 2, 3])
    
    # Use Loop as a context manager with a stopping condition
    with Loop(dataloader, max_steps=10) as loop:
        # Check that __enter__ returns the Loop instance
        assert isinstance(loop, Loop)
        assert loop.dataloader is dataloader
        
        # Do something with the loop
        assert loop._state.current_epoch == 0
        
    # Context is exited here, __exit__ is called

# ExitTracker class is no longer needed as we use subclassing instead

def test_loop_exit_called():
    """Test that __exit__ is called when exiting the context."""
    # Create a dataloader
    dataloader = MockDataLoader([1, 2, 3])
    
    # Create a loop with a custom __exit__ method for tracking
    class TrackingLoop(Loop):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.exit_called = False
            self.exc_info = None
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exit_called = True
            self.exc_info = (exc_type, exc_val, exc_tb)
            return super().__exit__(exc_type, exc_val, exc_tb)
    
    # Create a tracking loop with a stopping condition
    loop = TrackingLoop(dataloader, max_epochs=1)
    
    # Use the loop as a context manager
    with loop:
        pass
    
    # Check that __exit__ was called
    assert loop.exit_called
    assert loop.exc_info == (None, None, None)

def test_loop_exit_with_exception():
    """Test __exit__ behavior with exceptions."""
    # Create a dataloader
    dataloader = MockDataLoader([1, 2, 3])
    
    # Create a loop with a custom __exit__ method for tracking
    class TrackingLoop(Loop):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.exit_called = False
            self.exc_info = None
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exit_called = True
            self.exc_info = (exc_type, exc_val, exc_tb)
            return super().__exit__(exc_type, exc_val, exc_tb)
    
    # Create a tracking loop with a stopping condition
    loop = TrackingLoop(dataloader, max_steps=5)
    
    # Use the loop as a context manager with an exception
    try:
        with loop:
            raise ValueError("Test exception")
    except ValueError:
        pass  # Expected exception
    
    # Check that __exit__ was called with exception info
    assert loop.exit_called
    assert loop.exc_info[0] is ValueError
    assert str(loop.exc_info[1]) == "Test exception"
    assert loop.exc_info[2] is not None

def test_loop_iteration():
    """Test that Loop can be iterated over."""
    # Create a dataloader with known batches
    batches = [10, 20, 30]
    dataloader = MockDataLoader(batches)
    
    # Create a loop
    loop = Loop(dataloader, max_epochs=1)
    
    # Iterate over the loop
    collected_batches = []
    for batch, events in loop:
        collected_batches.append(batch)
    
    # Check that we got the expected batches
    assert collected_batches == batches

def test_loop_next():
    """Test that __next__ returns batches from the dataloader."""
    # Create a dataloader with known batches
    batches = [100, 200, 300, 400]
    dataloader = MockDataLoader(batches)
    
    # Create a loop with a stopping condition
    loop = Loop(dataloader, max_epochs=1)
    
    # Initialize the iterator
    iter(loop)
    
    # Get batches one by one
    batch, events = next(loop)
    assert batch == 100
    assert isinstance(events, set)
    
    batch, events = next(loop)
    assert batch == 200
    assert isinstance(events, set)
    
    batch, events = next(loop)
    assert batch == 300
    assert isinstance(events, set)
    
    batch, events = next(loop)
    assert batch == 400
    assert isinstance(events, set)
    
    # Should raise StopIteration when exhausted
    with pytest.raises(StopIteration):
        next(loop)

def test_state_updates_during_iteration():
    """Test state updates during iteration."""
    # Create a dataloader
    batches = [10, 20, 30]
    dataloader = MockDataLoader(batches)
    
    # Create a loop with a stopping condition
    loop = Loop(dataloader, max_epochs=1)
    
    # Initial state
    assert loop._state.current_epoch == 0
    assert loop._state.global_step == 0
    assert loop._state.epoch_step == 0
    
    # First batch
    iter(loop)  # Initialize iterator
    batch, events = next(loop)
    assert batch == 10
    assert loop._state.global_step == 1
    assert loop._state.epoch_step == 1
    
    # Second batch
    batch, events = next(loop)
    assert batch == 20
    assert loop._state.global_step == 2
    assert loop._state.epoch_step == 2
    
    # Third batch
    batch, events = next(loop)
    assert batch == 30
    assert loop._state.global_step == 3
    assert loop._state.epoch_step == 3

def test_epoch_transitions():
    """Test epoch transitions during iteration."""
    # Create a dataloader with a single batch
    dataloader = MockDataLoader([1])
    
    # Create a loop with a stopping condition, allowing multiple epochs
    loop = Loop(dataloader, max_epochs=3)
    
    # Initial state
    assert loop._state.current_epoch == 0
    
    # First epoch
    iter(loop)
    batch, events = next(loop)
    assert batch == 1
    assert loop._state.current_epoch == 0
    assert loop._state.global_step == 1
    
    # Transition to second epoch
    batch, events = next(loop)
    assert batch == 1  # Same batch, new epoch
    assert loop._state.current_epoch == 1
    assert loop._state.global_step == 2
    assert loop._state.epoch_step == 1  # Reset for new epoch
    assert LoopEvents.EPOCH_END in events  # Should be present at epoch transition
    
    # Transition to third epoch
    batch, events = next(loop)
    assert batch == 1  # Same batch, new epoch
    assert loop._state.current_epoch == 2
    assert loop._state.global_step == 3
    assert loop._state.epoch_step == 1
    assert LoopEvents.EPOCH_END in events  # Should be present at epoch transition

def test_max_epochs_termination():
    """Test termination at max_epochs."""
    # Create a dataloader with a single batch
    dataloader = MockDataLoader([1])
    
    # Create a loop with max_epochs=2
    loop = Loop(dataloader, max_epochs=2)
    
    # Should get 2 batches (one per epoch) then stop
    collected_batches = []
    for batch, events in loop:
        collected_batches.append(batch)
        
    assert collected_batches == [1, 1]
    assert loop._state.current_epoch == 2
    
    # Check that we reached max_epochs
    with pytest.raises(StopIteration):
        next(iter(loop))

def test_max_steps_termination():
    """Test termination at max_steps."""
    # Create a dataloader with multiple batches
    dataloader = MockDataLoader([1, 2, 3, 4, 5])
    
    # Create a loop with max_steps=3
    loop = Loop(dataloader, max_steps=3)
    
    # Should get 3 batches then stop
    collected_batches = []
    for batch, events in loop:
        collected_batches.append(batch)
        
    assert collected_batches == [1, 2, 3]
    assert loop._state.global_step == 3
    
    # Check that we reached max_steps
    with pytest.raises(StopIteration):
        next(iter(loop))
        
def test_no_stopping_condition():
    """Test the new validation for required stopping condition."""
    # Create a dataloader
    dataloader = MockDataLoader([1, 2, 3])
    
    # Creating a loop without a stopping condition should raise ValueError
    with pytest.raises(ValueError, match="stopping condition"):
        Loop(dataloader)

def test_event_triggering():
    """Test that events trigger at the correct steps."""
    # Create a dataloader with several batches
    dataloader = MockDataLoader([1, 2, 3, 4, 5, 6, 7, 8])
    
    # Create events with different triggering conditions
    events = {
        'every_2_steps': Event(every_n_steps=2),  # Will trigger at steps 1, 3, 5, 7 (2nd, 4th, 6th, 8th iterations)
        'at_step_3': Event(at_step=3),  # Will trigger at step 3 (4th iteration)
        'custom_condition': Event(condition_func=lambda state: state.global_step == 5)  # Step 5
    }
    
    # Create loop with events
    loop = Loop(dataloader=dataloader, events=events, max_steps=8)
    
    # Iterate through the loop and check events
    iter(loop)
    
    # Step 1 - first iteration, step 0 in 0-indexed counting
    batch, triggered = next(loop)
    assert batch == 1
    assert len(triggered) == 0  # No events triggered at step 0
    
    # Step 2 - second iteration, step 1 in 0-indexed counting
    batch, triggered = next(loop)
    assert batch == 2
    assert len(triggered) == 1
    assert 'every_2_steps' in triggered  # Triggers on 1st step (0-indexed)
    
    # Step 3 - third iteration, step 2 in 0-indexed counting 
    batch, triggered = next(loop)
    assert batch == 3
    assert len(triggered) == 0
    
    # Step 4 - fourth iteration, step 3 in 0-indexed counting
    batch, triggered = next(loop)
    assert batch == 4
    assert len(triggered) == 2
    assert 'every_2_steps' in triggered  # Triggers on 3rd step (0-indexed)
    assert 'at_step_3' in triggered  # Triggers on step 3
    
    # Step 5 - fifth iteration, step 4 in 0-indexed counting
    batch, triggered = next(loop)
    assert batch == 5
    assert len(triggered) == 0
    
    # Step 6 - sixth iteration, step 5 in 0-indexed counting
    batch, triggered = next(loop)
    assert batch == 6
    assert len(triggered) == 2
    assert 'every_2_steps' in triggered  # Triggers on 5th step (0-indexed)
    assert 'custom_condition' in triggered  # Triggers when global_step is 5

def test_multiple_events_trigger():
    """Test that multiple events can trigger simultaneously."""
    # Create a dataloader with enough batches
    dataloader = MockDataLoader([1, 2, 3, 4, 5, 6])
    
    # Create events that will trigger at the same step
    # For step 3 (fourth iteration) - remember we're 0-indexed
    events = {
        'event_a': Event(every_n_steps=2),  # Triggers on steps 1, 3, 5 (0-indexed)
        'event_b': Event(at_step=3),  # Triggers on step 3 (0-indexed)
        'event_c': Event(condition_func=lambda state: state.global_step == 3)  # When global_step is 3
    }
    
    # Create loop with events
    loop = Loop(dataloader=dataloader, events=events, max_steps=6)
    
    # Iterate and get to step 4 where multiple events should trigger
    iter(loop)
    next(loop)  # Step 1
    next(loop)  # Step 2
    next(loop)  # Step 3
    
    # Step 4 - fourth iteration, step 3 in 0-indexed counting
    batch, triggered = next(loop)
    assert batch == 4
    assert len(triggered) == 3
    assert 'event_a' in triggered  # Triggers on step 3 (0-indexed)
    assert 'event_b' in triggered  # Triggers on step 3 (0-indexed)
    assert 'event_c' in triggered  # Triggers when global_step is 4

def test_epoch_end_event():
    """Test that LoopEvents.EPOCH_END triggers at epoch transitions."""
    # Create a dataloader with a few batches
    dataloader = MockDataLoader([1, 2])
    
    # Create a loop with multiple epochs
    loop = Loop(dataloader=dataloader, max_epochs=2)
    
    # First epoch
    iter(loop)
    batch, triggered = next(loop)
    assert batch == 1
    assert len(triggered) == 0  # No events yet
    
    batch, triggered = next(loop)
    assert batch == 2
    assert len(triggered) == 0  # No events yet
    
    # End of first epoch, beginning of second
    batch, triggered = next(loop)
    assert batch == 1  # Back to first batch of dataloader
    assert LoopEvents.EPOCH_END in triggered  # EPOCH_END should trigger
    
    # Second batch of second epoch
    batch, triggered = next(loop)
    assert batch == 2
    assert len(triggered) == 0  # No events for this batch