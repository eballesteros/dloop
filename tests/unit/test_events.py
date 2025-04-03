from enum import Enum, auto
from typing import Union
from unittest import mock

from dloop.events import Event, LoopEvents
from dloop.types import LoopState


def get_simple_state(steps: int):
    return LoopState(
        epoch=0, global_step=steps, epoch_step=steps, epoch_end=False, training_end=False
    )


def test_every_n_steps():
    """Test that every_n_steps triggers at the right steps."""
    event = Event(every_n_steps=4)

    # Should not trigger at step 0
    assert event.should_trigger(get_simple_state(steps=0)) is False

    # Should trigger at steps 3, 7, 11, etc. (0-indexed)
    assert event.should_trigger(get_simple_state(steps=3)) is True
    assert event.should_trigger(get_simple_state(steps=7)) is True
    assert event.should_trigger(get_simple_state(steps=11)) is True

    # Should not trigger at other steps
    assert event.should_trigger(get_simple_state(steps=1)) is False
    assert event.should_trigger(get_simple_state(steps=2)) is False
    assert event.should_trigger(get_simple_state(steps=4)) is False
    assert event.should_trigger(get_simple_state(steps=5)) is False
    assert event.should_trigger(get_simple_state(steps=6)) is False


def test_at_step():
    """Test that at_step triggers only at the specified step."""
    event = Event(at_step=7)

    # Should not trigger at step 0
    assert event.should_trigger(get_simple_state(steps=0)) is False

    # Should trigger only at step 7
    assert event.should_trigger(get_simple_state(steps=7)) is True

    # Should not trigger at other steps
    assert event.should_trigger(get_simple_state(steps=1)) is False
    assert event.should_trigger(get_simple_state(steps=6)) is False
    assert event.should_trigger(get_simple_state(steps=8)) is False


def test_multiple_conditions():
    """Test that multiple conditions work together."""
    # Event with both every_n_steps and at_step
    event = Event(every_n_steps=4, at_step=5)

    # Should trigger at steps 3, 5, 7, 11 (steps 3 and 7 from every_n_steps=4, step 5 from at_step)
    assert event.should_trigger(get_simple_state(steps=3)) is True
    assert event.should_trigger(get_simple_state(steps=5)) is True
    assert event.should_trigger(get_simple_state(steps=7)) is True
    assert event.should_trigger(get_simple_state(steps=11)) is True

    # Should not trigger at other steps
    assert event.should_trigger(get_simple_state(steps=0)) is False
    assert event.should_trigger(get_simple_state(steps=1)) is False
    assert event.should_trigger(get_simple_state(steps=2)) is False
    assert event.should_trigger(get_simple_state(steps=4)) is False
    assert event.should_trigger(get_simple_state(steps=6)) is False


def test_condition_func():
    """Test that custom condition function works."""

    # Event with condition_func that triggers on even steps
    def even_steps(state):
        return state.global_step % 2 == 0

    event = Event(condition_function=even_steps)

    # Should trigger at even steps (except 0)
    assert event.should_trigger(get_simple_state(steps=2)) is True
    assert event.should_trigger(get_simple_state(steps=4)) is True
    assert event.should_trigger(get_simple_state(steps=6)) is True

    # Should not trigger at odd steps
    assert event.should_trigger(get_simple_state(steps=1)) is False
    assert event.should_trigger(get_simple_state(steps=3)) is False
    assert event.should_trigger(get_simple_state(steps=5)) is False


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

    # Test with custom events in same dictionary
    class TrainingEvents(Enum):
        VALIDATION = auto()
        LOGGING = auto()

    # Create a dictionary with events as keys
    handlers: dict[Union[LoopEvents, TrainingEvents], str] = {
        LoopEvents.EXCEPTION: "exception_handler",
        LoopEvents.EPOCH_END: "epoch_end_handler",
    }

    # Test dictionary access with events
    assert handlers[LoopEvents.EXCEPTION] == "exception_handler"
    assert handlers[LoopEvents.EPOCH_END] == "epoch_end_handler"

    # Update dictionary with custom event
    handlers[TrainingEvents.VALIDATION] = "validation_handler"
    handlers[TrainingEvents.LOGGING] = "logging_handler"

    # Check event handlers can be accessed with their respective enum values
    assert handlers[TrainingEvents.VALIDATION] == "validation_handler"
    assert handlers[TrainingEvents.LOGGING] == "logging_handler"
    assert handlers[LoopEvents.EXCEPTION] == "exception_handler"


def test_every_n_seconds():
    """Test that every_n_seconds triggers at the right intervals."""
    # Create a state to pass (value doesn't matter for time-based events)
    state = get_simple_state(steps=0)

    # Mock time.time() to control the clock
    with mock.patch("time.time") as mock_time:
        # Start at time 100
        mock_time.return_value = 100

        # Create event that triggers every 5 seconds
        event = Event(every_n_seconds=5)

        # At time=100, shouldn't trigger yet (just initialized)
        assert event.should_trigger(state) is False

        # Move clock forward 3 seconds (not enough time)
        mock_time.return_value = 103
        assert event.should_trigger(state) is False

        # Move clock forward 3 more seconds (total 6s since start, enough time)
        mock_time.return_value = 106
        assert event.should_trigger(state) is True

        # Clock at 106, but we just triggered, so should be false again
        assert event.should_trigger(state) is False

        # Move clock forward another 5 seconds
        mock_time.return_value = 111
        assert event.should_trigger(state) is True

        # Clock at 111, just triggered again
        assert event.should_trigger(state) is False

        # Move clock forward 4 seconds (not enough)
        mock_time.return_value = 115
        assert event.should_trigger(state) is False

        # Move clock forward 1 more second (exactly 5s)
        mock_time.return_value = 116
        assert event.should_trigger(state) is True


def test_at_time():
    """Test that at_time triggers once at the specified time."""
    # Create a state to pass (value doesn't matter for time-based events)
    state = get_simple_state(steps=0)

    # Mock time.time() to control the clock
    with mock.patch("time.time") as mock_time:
        # Start at time 100
        mock_time.return_value = 100

        # Create event that triggers 10 seconds after start
        event = Event(at_time=10)

        # At time=100, shouldn't trigger yet
        assert event.should_trigger(state) is False

        # At time=109, still not enough time
        mock_time.return_value = 109
        assert event.should_trigger(state) is False

        # At time=110 (10s after start), should trigger
        mock_time.return_value = 110
        assert event.should_trigger(state) is True

        # Already triggered, should only trigger once
        mock_time.return_value = 120
        assert event.should_trigger(state) is False

        # Even much later, should still not trigger again
        mock_time.return_value = 200
        assert event.should_trigger(state) is False


def test_mixed_time_and_step_conditions():
    """Test events with both time and step conditions."""

    # Mock time.time() to control the clock
    with mock.patch("time.time") as mock_time:
        # Start at time 100
        mock_time.return_value = 100

        # Event with both step and time conditions
        event = Event(every_n_steps=4, every_n_seconds=10)

        # Should trigger due to steps, even though time condition isn't met
        assert event.should_trigger(get_simple_state(steps=3)) is True

        # Should not trigger due to steps
        assert event.should_trigger(get_simple_state(steps=1)) is False

        # Move time forward to trigger time condition
        mock_time.return_value = 111
        assert event.should_trigger(get_simple_state(steps=1)) is True

        # Time condition triggered, shouldn't trigger again until 10 more seconds
        mock_time.return_value = 115
        assert event.should_trigger(get_simple_state(steps=1)) is False

        # Step condition should still work independently
        assert event.should_trigger(get_simple_state(steps=3)) is True
