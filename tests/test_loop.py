import pytest
import time
import json
from collections import deque
from dloop.loop import LoopState, Loop

def test_loop_state_init():
    """Test LoopState initialization with default values."""
    state = LoopState()
    
    # Check initial values
    assert state.current_epoch == 0
    assert state.global_step == 0
    assert state.local_epoch_step == 0
    
    # Verify start_time is a recent timestamp
    current_time = time.time()
    assert state.start_time <= current_time
    assert state.start_time > current_time - 10  # Within last 10 seconds

def test_loop_state_local_epoch_step():
    """Test local_epoch_step calculation."""
    state = LoopState()
    
    # Should start at 0
    assert state.local_epoch_step == 0
    
    # Increment steps and check local_epoch_step
    state.increment_step()
    assert state.global_step == 1
    assert state.local_epoch_step == 1
    
    state.increment_step()
    assert state.global_step == 2
    assert state.local_epoch_step == 2
    
    # Increment epoch and check local_epoch_step resets
    state.increment_epoch()
    assert state.current_epoch == 1
    assert state.global_step == 2
    assert state.local_epoch_step == 0
    
    # Add more steps in the new epoch
    state.increment_step()
    assert state.global_step == 3
    assert state.local_epoch_step == 1

def test_loop_state_epoch_transitions():
    """Test epoch transitions and step counting."""
    state = LoopState()
    
    # Simulate first epoch
    for _ in range(5):
        state.increment_step()
    
    assert state.current_epoch == 0
    assert state.global_step == 5
    assert state.local_epoch_step == 5
    
    # Transition to next epoch
    state.increment_epoch()
    
    assert state.current_epoch == 1
    assert state.global_step == 5
    assert state.local_epoch_step == 0
    
    # Simulate second epoch
    for _ in range(3):
        state.increment_step()
    
    assert state.current_epoch == 1
    assert state.global_step == 8
    assert state.local_epoch_step == 3

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
    assert state.local_epoch_step == 20  # 100 - 80

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
    assert reconstructed.local_epoch_step == original.local_epoch_step

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
    assert loaded_state.local_epoch_step == 30

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
    
    # Create a loop with minimal parameters
    loop = Loop(dataloader)
    
    # Check dataloader is stored
    assert loop.dataloader is dataloader
    
    # Check defaults
    assert loop.events == {}
    assert loop.max_epochs is None
    assert loop.max_steps is None
    assert loop.state_file is None
    
    # Check state is initialized
    assert isinstance(loop.state, LoopState)
    assert loop.state.current_epoch == 0
    assert loop.state.global_step == 0

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
    assert isinstance(loop.state, LoopState)

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
    
    # Create loop with events
    loop = Loop(dataloader=dataloader, events=events)
    
    # Check events are stored correctly
    assert loop.events == events
    assert loop.events["string_event"] == "handler1"
    assert loop.events[TestEvents.EVENT1] == "handler2"
    assert loop.events[123] == "handler3"

def test_loop_with_deque():
    """Test Loop with a deque as dataloader."""
    # Create a deque as dataloader
    dataloader = deque([1, 2, 3])
    
    # Create loop
    loop = Loop(dataloader)
    
    # Check dataloader is stored
    assert loop.dataloader is dataloader