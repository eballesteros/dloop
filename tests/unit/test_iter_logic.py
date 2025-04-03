import time
from dataclasses import asdict
from enum import Enum, auto, unique

from dloop.events import Event, LoopEvents
from dloop.iter_logic import (
    get_iter_dl_with_events,
    iter_dl_known_length,
    iter_dl_unknown_length_with_pairwise_load,
)


def test_iter_dl_known_length_max_epochs():
    dl = list(range(4))
    it = iter_dl_known_length(dl, dl_len=len(dl), max_epochs=2)

    assert [(b, asdict(s)) for b, s in it] == [
        (
            0,
            {
                "epoch": 0,
                "global_step": 0,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 0,
                "global_step": 1,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            2,
            {
                "epoch": 0,
                "global_step": 2,
                "epoch_step": 2,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            3,
            {
                "epoch": 0,
                "global_step": 3,
                "epoch_step": 3,
                "epoch_end": True,
                "training_end": False,
            },
        ),
        (
            0,
            {
                "epoch": 1,
                "global_step": 4,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 1,
                "global_step": 5,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            2,
            {
                "epoch": 1,
                "global_step": 6,
                "epoch_step": 2,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            3,
            {
                "epoch": 1,
                "global_step": 7,
                "epoch_step": 3,
                "epoch_end": True,
                "training_end": True,
            },
        ),
    ]


def test_iter_dl_known_length_max_steps():
    dl = list(range(4))
    it = iter_dl_known_length(dl, dl_len=len(dl), max_steps=6)

    assert [(b, asdict(s)) for b, s in it] == [
        (
            0,
            {
                "epoch": 0,
                "global_step": 0,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 0,
                "global_step": 1,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            2,
            {
                "epoch": 0,
                "global_step": 2,
                "epoch_step": 2,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            3,
            {
                "epoch": 0,
                "global_step": 3,
                "epoch_step": 3,
                "epoch_end": True,
                "training_end": False,
            },
        ),
        (
            0,
            {
                "epoch": 1,
                "global_step": 4,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 1,
                "global_step": 5,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": True,
            },
        ),
    ]


def test_iter_dl_known_length_max_steps_epoch_end():
    """
    last step happens to match an end of epoch
    """
    dl = list(range(4))
    it = iter_dl_known_length(dl, dl_len=len(dl), max_steps=8)

    assert [(b, asdict(s)) for b, s in it] == [
        (
            0,
            {
                "epoch": 0,
                "global_step": 0,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 0,
                "global_step": 1,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            2,
            {
                "epoch": 0,
                "global_step": 2,
                "epoch_step": 2,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            3,
            {
                "epoch": 0,
                "global_step": 3,
                "epoch_step": 3,
                "epoch_end": True,
                "training_end": False,
            },
        ),
        (
            0,
            {
                "epoch": 1,
                "global_step": 4,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 1,
                "global_step": 5,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            2,
            {
                "epoch": 1,
                "global_step": 6,
                "epoch_step": 2,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            3,
            {
                "epoch": 1,
                "global_step": 7,
                "epoch_step": 3,
                "epoch_end": True,
                "training_end": True,
            },
        ),
    ]


def test_iter_dl_unknown_length_with_pairwise_load_max_epochs():
    dl = list(range(4))
    it = iter_dl_unknown_length_with_pairwise_load(dl, max_epochs=2)

    assert [(b, asdict(s)) for b, s in it] == [
        (
            0,
            {
                "epoch": 0,
                "global_step": 0,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 0,
                "global_step": 1,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            2,
            {
                "epoch": 0,
                "global_step": 2,
                "epoch_step": 2,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            3,
            {
                "epoch": 0,
                "global_step": 3,
                "epoch_step": 3,
                "epoch_end": True,
                "training_end": False,
            },
        ),
        (
            0,
            {
                "epoch": 1,
                "global_step": 4,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 1,
                "global_step": 5,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            2,
            {
                "epoch": 1,
                "global_step": 6,
                "epoch_step": 2,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            3,
            {
                "epoch": 1,
                "global_step": 7,
                "epoch_step": 3,
                "epoch_end": True,
                "training_end": True,
            },
        ),
    ]


def test_iter_dl_unknown_length_with_pairwise_load_max_steps():
    dl = list(range(4))
    it = iter_dl_unknown_length_with_pairwise_load(dl, max_steps=6)

    assert [(b, asdict(s)) for b, s in it] == [
        (
            0,
            {
                "epoch": 0,
                "global_step": 0,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 0,
                "global_step": 1,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            2,
            {
                "epoch": 0,
                "global_step": 2,
                "epoch_step": 2,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            3,
            {
                "epoch": 0,
                "global_step": 3,
                "epoch_step": 3,
                "epoch_end": True,
                "training_end": False,
            },
        ),
        (
            0,
            {
                "epoch": 1,
                "global_step": 4,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 1,
                "global_step": 5,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": True,
            },
        ),
    ]


def test_iter_dl_unknown_length_with_pairwise_load_max_steps_epoch_end():
    """
    last step happens to match an end of epoch
    """
    dl = list(range(4))
    it = iter_dl_unknown_length_with_pairwise_load(dl, max_steps=8)

    assert [(b, asdict(s)) for b, s in it] == [
        (
            0,
            {
                "epoch": 0,
                "global_step": 0,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 0,
                "global_step": 1,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            2,
            {
                "epoch": 0,
                "global_step": 2,
                "epoch_step": 2,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            3,
            {
                "epoch": 0,
                "global_step": 3,
                "epoch_step": 3,
                "epoch_end": True,
                "training_end": False,
            },
        ),
        (
            0,
            {
                "epoch": 1,
                "global_step": 4,
                "epoch_step": 0,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            1,
            {
                "epoch": 1,
                "global_step": 5,
                "epoch_step": 1,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            2,
            {
                "epoch": 1,
                "global_step": 6,
                "epoch_step": 2,
                "epoch_end": False,
                "training_end": False,
            },
        ),
        (
            3,
            {
                "epoch": 1,
                "global_step": 7,
                "epoch_step": 3,
                "epoch_end": True,
                "training_end": True,
            },
        ),
    ]


def test_get_iter_dl_with_events_known_len():
    dl = list(range(4))

    # max epochs
    it = get_iter_dl_with_events(dl, dl_len=len(dl), max_epochs=2)

    assert list(it) == [
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
    it = get_iter_dl_with_events(dl, dl_len=len(dl), max_steps=6)

    assert list(it) == [
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END}),
        (0, set()),
        (1, {LoopEvents.TRAINING_END}),
    ]

    # max steps matches epoch end
    it = get_iter_dl_with_events(dl, dl_len=len(dl), max_steps=8)

    assert list(it) == [
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END}),
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END, LoopEvents.TRAINING_END}),
    ]


def test_get_iter_dl_with_events_unknown_len():
    dl = list(range(4))

    # max epochs
    it = get_iter_dl_with_events(dl, max_epochs=2)

    assert list(it) == [
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
    it = get_iter_dl_with_events(dl, max_steps=6)

    assert list(it) == [
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END}),
        (0, set()),
        (1, {LoopEvents.TRAINING_END}),
    ]

    # max steps matches epoch end
    it = get_iter_dl_with_events(dl, max_steps=8)

    assert list(it) == [
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END}),
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END, LoopEvents.TRAINING_END}),
    ]


def test_iter_logic_with_custom_events():
    # Create test data
    dl = list(range(4))

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
    it = get_iter_dl_with_events(dl, max_epochs=2, events=custom_events)

    assert list(it) == [
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
    it = get_iter_dl_with_events(dl, max_steps=6, events=custom_events)

    assert list(it) == [
        (0, set()),
        (1, {CustomEvents.Every2}),
        (2, set()),
        (3, {LoopEvents.EPOCH_END, CustomEvents.Every2, CustomEvents.CustomAt3}),
        (0, set()),
        (1, {CustomEvents.Every2, LoopEvents.TRAINING_END}),
    ]

    # max steps matches epoch end
    it = get_iter_dl_with_events(dl, max_steps=8, events=custom_events)

    assert list(it) == [
        (0, set()),
        (1, {CustomEvents.Every2}),
        (2, set()),
        (3, {LoopEvents.EPOCH_END, CustomEvents.Every2, CustomEvents.CustomAt3}),
        (0, set()),
        (1, {CustomEvents.Every2}),
        (2, {CustomEvents.At6}),
        (3, {LoopEvents.EPOCH_END, LoopEvents.TRAINING_END, CustomEvents.Every2}),
    ]


def test_iter_dl_known_length_max_seconds():
    """Test that training stops after max_seconds with known length dataloader."""

    # Create a dataloader that includes a small sleep to make timing more predictable
    class SlowDataLoader:
        def __init__(self, data, sleep_time=0.05):
            self.data = data
            self.sleep_time = sleep_time

        def __iter__(self):
            for item in self.data:
                time.sleep(self.sleep_time)
                yield item

        def __len__(self):
            return len(self.data)

    # Create a dataloader with items 0 to 19
    dl = SlowDataLoader(range(20), sleep_time=0.05)

    # With a 0.05s delay per item and 0.2s max_seconds, we should get about 4 items
    it = iter_dl_known_length(dl, dl_len=len(dl), max_seconds=0.2)

    # Collect results
    results = []
    for batch, state in it:
        results.append((batch, asdict(state)))

    # We should have stopped due to time limit, not because we ran out of items
    assert len(results) < 20

    # Verify that the last item has training_end=True
    assert results[-1][1]["training_end"] is True

    # The rest should have training_end=False
    for _batch, state in results[:-1]:
        assert state["training_end"] is False


def test_iter_dl_unknown_length_max_seconds():
    """Test that training stops after max_seconds with unknown length dataloader."""

    # Create a dataloader that includes a small sleep to make timing more predictable
    class SlowIterableDataLoader:
        def __init__(self, data, sleep_time=0.05):
            self.data = data
            self.sleep_time = sleep_time

        def __iter__(self):
            for item in self.data:
                time.sleep(self.sleep_time)
                yield item

    # Create a dataloader with items 0 to 19 (no __len__ method)
    dl = SlowIterableDataLoader(range(20), sleep_time=0.05)

    # With a 0.05s delay per item and 0.2s max_seconds, we should get about 4 items
    it = iter_dl_unknown_length_with_pairwise_load(dl, max_seconds=0.2)

    # Collect results
    results = []
    for batch, state in it:
        results.append((batch, asdict(state)))

    # We should have stopped due to time limit, not because we ran out of items
    assert len(results) < 20

    # Verify that the last item has training_end=True
    assert results[-1][1]["training_end"] is True

    # The rest should have training_end=False
    for _batch, state in results[:-1]:
        assert state["training_end"] is False


def test_get_iter_dl_with_events_max_seconds():
    """Test that the events system works with max_seconds."""

    # Create a dataloader that includes a small sleep
    class SlowDataLoader:
        def __init__(self, data, sleep_time=0.05):
            self.data = data
            self.sleep_time = sleep_time

        def __iter__(self):
            for item in self.data:
                time.sleep(self.sleep_time)
                yield item

        def __len__(self):
            return len(self.data)

    # Create test events
    @unique
    class CustomEvents(Enum):
        Every2 = auto()

    custom_events = {
        CustomEvents.Every2: Event(every_n_steps=2),
    }

    # Create a dataloader with 20 items
    dl = SlowDataLoader(range(20), sleep_time=0.05)

    # With max_seconds=0.2, we should get about 4 items
    it = get_iter_dl_with_events(dl, dl_len=len(dl), max_seconds=0.2, events=custom_events)

    # Collect results (batch, events)
    results = list(it)

    # We should have stopped due to time limit
    assert len(results) < 20

    # The last item should have TRAINING_END event
    assert LoopEvents.TRAINING_END in results[-1][1]

    # Every other item should have the Every2 event on even steps
    for i, (_batch, events) in enumerate(results[:-1]):
        if i % 2 == 1:  # 0-indexed, so steps 1, 3, 5, etc.
            assert CustomEvents.Every2 in events
