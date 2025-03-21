from dataclasses import asdict
from dloop.events import LoopEvents
from dloop.iter_logic import (
    get_iter_dl_with_events,
    iter_dl_known_length, 
    iter_dl_unknown_length_with_pairwise_load
)


def test_iter_dl_known_length_max_epochs():
    l = list(range(4))
    it = iter_dl_known_length(l, dl_len=len(l), max_epochs=2)

    assert [asdict(e) for e in it] == [
        {"batch": 0, "epoch": 0, "global_step": 0, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 0, "global_step": 1, "epoch_step": 1, "epoch_end": False, "training_end": False},
        {"batch": 2, "epoch": 0, "global_step": 2, "epoch_step": 2, "epoch_end": False, "training_end": False},
        {"batch": 3, "epoch": 0, "global_step": 3, "epoch_step": 3, "epoch_end": True, "training_end": False},
        {"batch": 0, "epoch": 1, "global_step": 4, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 1, "global_step": 5, "epoch_step": 1, "epoch_end": False, "training_end": False},
        {"batch": 2, "epoch": 1, "global_step": 6, "epoch_step": 2, "epoch_end": False, "training_end": False},
        {"batch": 3, "epoch": 1, "global_step": 7, "epoch_step": 3, "epoch_end": True, "training_end": True},
    ]

def test_iter_dl_known_length_max_steps():
    l = list(range(4))
    it = iter_dl_known_length(l, dl_len=len(l), max_steps=6)

    assert [asdict(e) for e in it] == [
        {"batch": 0, "epoch": 0, "global_step": 0, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 0, "global_step": 1, "epoch_step": 1, "epoch_end": False, "training_end": False},
        {"batch": 2, "epoch": 0, "global_step": 2, "epoch_step": 2, "epoch_end": False, "training_end": False},
        {"batch": 3, "epoch": 0, "global_step": 3, "epoch_step": 3, "epoch_end": True, "training_end": False},
        {"batch": 0, "epoch": 1, "global_step": 4, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 1, "global_step": 5, "epoch_step": 1, "epoch_end": False, "training_end": True},
    ]

def test_iter_dl_known_length_max_steps_epoch_end():
    """
    last step happens to match an end of epoch
    """
    l = list(range(4))
    it = iter_dl_known_length(l, dl_len=len(l), max_steps=8)

    assert [asdict(e) for e in it] == [
        {"batch": 0, "epoch": 0, "global_step": 0, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 0, "global_step": 1, "epoch_step": 1, "epoch_end": False, "training_end": False},
        {"batch": 2, "epoch": 0, "global_step": 2, "epoch_step": 2, "epoch_end": False, "training_end": False},
        {"batch": 3, "epoch": 0, "global_step": 3, "epoch_step": 3, "epoch_end": True, "training_end": False},
        {"batch": 0, "epoch": 1, "global_step": 4, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 1, "global_step": 5, "epoch_step": 1, "epoch_end": False, "training_end": False},
        {"batch": 2, "epoch": 1, "global_step": 6, "epoch_step": 2, "epoch_end": False, "training_end": False},
        {"batch": 3, "epoch": 1, "global_step": 7, "epoch_step": 3, "epoch_end": True, "training_end": True},
    ]

def test_iter_dl_unknown_length_with_pairwise_load_max_epochs():
    l = list(range(4))
    it = iter_dl_unknown_length_with_pairwise_load(l, max_epochs=2)

    assert [asdict(e) for e in it] == [
        {"batch": 0, "epoch": 0, "global_step": 0, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 0, "global_step": 1, "epoch_step": 1, "epoch_end": False, "training_end": False},
        {"batch": 2, "epoch": 0, "global_step": 2, "epoch_step": 2, "epoch_end": False, "training_end": False},
        {"batch": 3, "epoch": 0, "global_step": 3, "epoch_step": 3, "epoch_end": True, "training_end": False},
        {"batch": 0, "epoch": 1, "global_step": 4, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 1, "global_step": 5, "epoch_step": 1, "epoch_end": False, "training_end": False},
        {"batch": 2, "epoch": 1, "global_step": 6, "epoch_step": 2, "epoch_end": False, "training_end": False},
        {"batch": 3, "epoch": 1, "global_step": 7, "epoch_step": 3, "epoch_end": True, "training_end": True},
    ]

def test_iter_dl_unknown_length_with_pairwise_load_max_steps():
    l = list(range(4))
    it = iter_dl_unknown_length_with_pairwise_load(l, max_steps=6)

    assert [asdict(e) for e in it] == [
        {"batch": 0, "epoch": 0, "global_step": 0, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 0, "global_step": 1, "epoch_step": 1, "epoch_end": False, "training_end": False},
        {"batch": 2, "epoch": 0, "global_step": 2, "epoch_step": 2, "epoch_end": False, "training_end": False},
        {"batch": 3, "epoch": 0, "global_step": 3, "epoch_step": 3, "epoch_end": True, "training_end": False},
        {"batch": 0, "epoch": 1, "global_step": 4, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 1, "global_step": 5, "epoch_step": 1, "epoch_end": False, "training_end": True},
    ]

def test_iter_dl_unknown_length_with_pairwise_load_max_steps_epoch_end():
    """
    last step happens to match an end of epoch
    """
    l = list(range(4))
    it = iter_dl_unknown_length_with_pairwise_load(l, max_steps=8)

    assert [asdict(e) for e in it] == [
        {"batch": 0, "epoch": 0, "global_step": 0, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 0, "global_step": 1, "epoch_step": 1, "epoch_end": False, "training_end": False},
        {"batch": 2, "epoch": 0, "global_step": 2, "epoch_step": 2, "epoch_end": False, "training_end": False},
        {"batch": 3, "epoch": 0, "global_step": 3, "epoch_step": 3, "epoch_end": True, "training_end": False},
        {"batch": 0, "epoch": 1, "global_step": 4, "epoch_step": 0, "epoch_end": False, "training_end": False},
        {"batch": 1, "epoch": 1, "global_step": 5, "epoch_step": 1, "epoch_end": False, "training_end": False},
        {"batch": 2, "epoch": 1, "global_step": 6, "epoch_step": 2, "epoch_end": False, "training_end": False},
        {"batch": 3, "epoch": 1, "global_step": 7, "epoch_step": 3, "epoch_end": True, "training_end": True},
    ]

def test_get_iter_dl_with_events_known_len():
    l = list(range(4))

    # max epochs
    it = get_iter_dl_with_events(l, dl_len=len(l), max_epochs=2)

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
    it = get_iter_dl_with_events(l, dl_len=len(l), max_steps=6)

    assert list(it) == [
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END}),
        (0, set()),
        (1, {LoopEvents.TRAINING_END})
    ]
    
    # max steps matches epoch end
    it = get_iter_dl_with_events(l, dl_len=len(l), max_steps=8)

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
    l = list(range(4))

    # max epochs
    it = get_iter_dl_with_events(l, max_epochs=2)

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
    it = get_iter_dl_with_events(l, max_steps=6)

    assert list(it) == [
        (0, set()),
        (1, set()),
        (2, set()),
        (3, {LoopEvents.EPOCH_END}),
        (0, set()),
        (1, {LoopEvents.TRAINING_END})
    ]
    
    # max steps matches epoch end
    it = get_iter_dl_with_events(l, max_steps=8)

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