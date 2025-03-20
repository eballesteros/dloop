import pytest
from dloop.loop import LoopState

def test_loop_state_init():
    """Test LoopState initialization with default values."""
    state = LoopState()
    
    # Check initial values
    assert state.current_epoch == 0
    assert state.global_step == 0
    assert state.local_epoch_step == 0

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