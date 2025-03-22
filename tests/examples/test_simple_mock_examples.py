from enum import Enum, auto, unique

from dloop import Event, Loop, LoopEvents


def test_logging_and_gradient_accumulation():
    dl = list(range(128))

    @unique
    class CustomEvents(Enum):
        GradientAccumulation = auto()
        Loggig = auto()

    events = {
        CustomEvents.GradientAccumulation: Event(every_n_steps=4),
        CustomEvents.Loggig: Event(every_n_steps=16),
    }

    n_forward = 0
    n_grad_accumulation_steps = 0
    n_loggs = 0
    n_valid = 0

    with Loop(dl, max_epochs=2, events=events) as train_loop:
        for _, batch_events in train_loop:
            n_forward += 1

            if CustomEvents.GradientAccumulation in batch_events:
                n_grad_accumulation_steps += 1

            if CustomEvents.Loggig in batch_events:
                n_loggs += 1

            if LoopEvents.EPOCH_END in batch_events:
                n_valid += 1

    assert n_forward == 2 * 128
    assert n_grad_accumulation_steps == 2 * 128 // 4
    assert n_loggs == 2 * 128 // 16
    assert n_valid == 2
