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
