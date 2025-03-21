from enum import Enum, auto, unique
from typing import Iterable
import pytest
import time
import json
from collections import deque
from dloop.loop import Loop
from dloop.types import LoopState
from dloop.events import Event, LoopEvents

class MockDataLoader:
    def __init__(self, l:Iterable) -> None:
        self.l = l

    def __iter__(self):
        return iter(self.l)
    
def test_mock_dl():
    """testing that my MockDataLoader dataloader does what I think it does
    """
    dl = MockDataLoader(list(range(4)))

    assert list(dl) == [0, 1, 2, 3]
    
    # check that size can't be inferred
    with pytest.raises(TypeError):
        len(dl) # type: ignore

@pytest.mark.parametrize("dl_len", [4, None])
def test_loop_basics(dl_len):
    dl = MockDataLoader(list(range(4)))

    # max epochs
    loop = Loop(dl, max_epochs=2, dataloader_len=dl_len)

    assert list(loop) == [
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END}),
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END, LoopEvents.TRAINING_END}),
    ]

    # max steps
    loop = Loop(dl, max_steps=6, dataloader_len=dl_len)
    
    assert list(loop) == [
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END}),
        (0, set()),
        (1, {LoopEvents.TRAINING_END})
    ]
    
    # max steps matches epoch end
    loop = Loop(dl, max_steps=8, dataloader_len=dl_len)

    assert list(loop) == [
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END}),
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END, LoopEvents.TRAINING_END}),
    ]

@pytest.mark.parametrize("dl_len", [4, None])
def test_loop_custom_events(dl_len):
    dl = MockDataLoader(list(range(4)))

    @unique
    class CustomEvents(Enum):
        Every2 = auto()
        CustomAt3 = auto()
        At6 = auto()

    custom_events = {
        CustomEvents.Every2: Event(every_n_steps=2),
        CustomEvents.At6: Event(at_step=6),
        CustomEvents.CustomAt3: Event(lambda loop_state: loop_state.global_step == 3),
    }


    # max epochs
    loop = Loop(dl, max_epochs=2, dataloader_len=dl_len, events=custom_events)

    assert list(loop) == [
        (0, set()),
        (1, {CustomEvents.Every2}),
        (2, set()),
        (3, {LoopEvents.EPOCH_END, CustomEvents.Every2, CustomEvents.CustomAt3}),
        (0, set()),
        (1, {CustomEvents.Every2}),
        (2, {CustomEvents.At6}),
        (3, {LoopEvents.EPOCH_END, LoopEvents.TRAINING_END, CustomEvents.Every2}),
    ]

    # max steps
    loop = Loop(dl, max_steps=6, dataloader_len=dl_len, events=custom_events)
    
    assert list(loop) == [
        (0, set()),
        (1, {CustomEvents.Every2}),
        (2, set()),
        (3, {LoopEvents.EPOCH_END, CustomEvents.Every2, CustomEvents.CustomAt3}),
        (0, set()),
        (1, {CustomEvents.Every2, LoopEvents.TRAINING_END}),
    ]
    
    # max steps matches epoch end
    loop = Loop(dl, max_steps=8, dataloader_len=dl_len, events=custom_events)

    assert list(loop) == [
        (0, set()),
        (1, {CustomEvents.Every2}),
        (2, set()),
        (3, {LoopEvents.EPOCH_END, CustomEvents.Every2, CustomEvents.CustomAt3}),
        (0, set()),
        (1, {CustomEvents.Every2}),
        (2, {CustomEvents.At6}),
        (3, {LoopEvents.EPOCH_END, LoopEvents.TRAINING_END, CustomEvents.Every2}),
    ]

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
        for i, (b, e) in enumerate(loop):
            if i == 3: break
        
    # Context is exited here, __exit__ is called

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

def test_loop_exit_called():
    """Test that __exit__ is called when exiting the context."""
    # Create a dataloader
    dataloader = MockDataLoader([1, 2, 3])
    
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
    assert loop.exc_info
    assert loop.exc_info[0] is ValueError
    assert str(loop.exc_info[1]) == "Test exception"
    assert loop.exc_info[2] is not None

        
def test_no_stopping_condition():
    """Test the new validation for required stopping condition."""
    # Creating a loop without a stopping condition should raise ValueError
    with pytest.raises(ValueError, match="stopping condition"):
        Loop([1, 2, 3])
