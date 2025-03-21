from dataclasses import dataclass
import math
from typing import Any, Generator, Iterable, Literal, Optional
from .events import LoopEvents

try:
    from itertools import pairwise
except ImportError: # python < 3.10
    from itertools import tee

    # from more-itertools
    def pairwise(iterable):
        """Returns an iterator of paired items, overlapping, from the original

        >>> take(4, pairwise(count()))
        [(0, 1), (1, 2), (2, 3), (3, 4)]
        """
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

@dataclass
class IterItem:
    batch: Any
    epoch: int
    global_step: int
    epoch_step: int
    epoch_end: bool
    training_end: bool

def _check_arguments(max_epochs: Optional[int]=None, max_steps: Optional[int]=None) -> None:
    # one needs to be none, the other not none
    if (max_epochs is None) == (max_steps is None):
        raise ValueError(f"One and only one of max_epochs and max_steps should be different from None.\nGot {max_epochs=} {max_steps=}")

def iter_dl_known_length(dl: Iterable, dl_len: int, max_epochs: Optional[int]=None, max_steps: Optional[int]=None) -> Generator[IterItem, None, None]:
    """
    """
    _check_arguments(max_epochs=max_epochs, max_steps=max_steps)
    n_epochs = max_epochs or math.ceil(max_steps / dl_len) # type: ignore

    global_step = 0
    last_epoch = False
    for epoch in range(n_epochs):
        if epoch == n_epochs - 1:
            last_epoch = True

        for epoch_step, batch in enumerate(dl):
            # both mean "after we yield this batch"
            max_steps_reached = max_steps is not None and global_step == max_steps - 1
            epoch_end = epoch_step == dl_len - 1

            yield IterItem(
                batch=batch,
                epoch=epoch,
                global_step=global_step,
                epoch_step=epoch_step,
                epoch_end=epoch_end,
                training_end=max_steps_reached or (last_epoch and epoch_end)
            )
            
            if max_steps_reached:
                return
            
            global_step += 1

def iter_dl_unknown_length_with_pairwise_load(dl: Iterable, max_epochs: Optional[int]=None, max_steps: Optional[int]=None) -> Generator[IterItem, None, None]:
    """
    within each epoch, iterates over pairwise(dl) to be able to tell when the epoch is done before yielding the last batch. 
    It's equivalent to efficiently peeking the next batch in the dl.
    """
    _check_arguments(max_epochs=max_epochs, max_steps=max_steps)

    global_step = 0
    epoch = 0
    
    # initialize an infinite loop, we'll use stop conditions to exit
    while True:
        last_epoch = (max_epochs is not None) and (epoch == max_epochs - 1)
        for epoch_step, (batch, next_batch) in enumerate(pairwise(dl)):
            # we always yield the first batch of the pair. 
            # pairwise handles batch = next_batch, next_batch = next(dl) for us
            
            # "after we yield this batch"
            max_steps_reached = max_steps is not None and global_step == max_steps - 1
            # at this point, the epoch hasn't ended, so the only way trining ended is if max_steps_reached
            training_end=max_steps_reached

            yield IterItem(
                batch=batch,
                epoch=epoch,
                global_step=global_step,
                epoch_step=epoch_step,
                epoch_end=False,
                # at this point, the epoch hasn't ended, so the only way trining ended is if max_steps_reached
                training_end=training_end
            )

            if training_end:
                return
            
            global_step += 1
        
        # If we exited the previous loop, it means next_batch = next(dl) failed because the dl was exausted
        # and therfore next_batch is the last batch of the epoch.
        
        # we're at the end of the epoch, so if it were the last epoch we'd be done traning
        max_steps_reached = max_steps is not None and global_step == max_steps - 1
        training_end=max_steps_reached or last_epoch
        yield IterItem(
            batch=next_batch,
            epoch=epoch,
            global_step=global_step,
            epoch_step=epoch_step + 1,
            epoch_end=True,
            training_end=training_end
        )

        if training_end:
            return

        global_step += 1
        epoch += 1


NoLenIterationStrategy = Literal['pairwise']

def get_iter_dl_with_events(
        dl: Iterable,
        dl_len: Optional[int]=None,
        max_epochs: Optional[int]=None,
        max_steps: Optional[int]=None,
        no_len_iteration_strategy: NoLenIterationStrategy = 'pairwise'
    ) -> Generator[tuple[Any, set[LoopEvents]], None, None]:
    """
    """
    kwargs = dict(max_epochs=max_epochs, max_steps=max_steps)
    if dl_len is not None:
        kwargs['dl_len'] = dl_len
        iter_f = iter_dl_known_length
    else:
        if no_len_iteration_strategy == 'pairwise':
            iter_f = iter_dl_unknown_length_with_pairwise_load

    for it_item in iter_f(dl, **kwargs): # type: ignore
        batch_events = set()
        if it_item.epoch_end:
            batch_events.add(LoopEvents.EPOCH_END)

        if it_item.training_end:
            batch_events.add(LoopEvents.TRAINING_END)

        yield it_item.batch, batch_events
