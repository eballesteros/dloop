# dloop

A lightweight Python library for simpler deep learning training loops.

## What is it?

`dloop` is a small library that helps clean up those messy training loops without imposing a whole framework on you. It won't allow you to train a neural network in 4 lines of code, nor will it make your model perform better nor converge faster, but if you enjoy rolling your own training loops it may handle some inconveniences for you.

This is how your training loop could look like with `dloop`:

```python
from dloop import Event, Loop, LoopEvents

# << your custom events, can be triggered every_n_steps,
# at_step, or through an arbitrary condition_function
events = {
    "OptimizerStep": Event(every_n_steps=16), 
    "Logging": Event(every_n_steps=100),
    "DecreaseLR": Event(at_step=10_000)
}


with Loop(dataloader, max_epochs=3, events=events) as train_loop: # << could also have used max_steps=50_000
    for batch, batch_events in train_loop: # << handles looping over you dataloader as many times as necessary, and stopping when appropriate
        # Forward pass
        loss = model(batch)
        loss.backward()
        
        if "OptimizerStep" in batch_events: # << periodically triggered every 16 batches (gradient accumulation)
            optimizer.step()
            optimizer.zero_grad()
            
        if "Logging" in batch_events:
            step = train_loop.global_step # << access to iteration state
            print(f"Step {step}: Loss {loss.item():.4f}")

        if "DecreaseLR" in batch_events:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1  # decrease learning rate by factor of 10
            
        if LoopEvents.EPOCH_END in batch_events: # << pre-defined events, like EPOCH_END or TRAINING_END
            validate(model, val_dataloader)
            checkpoint(model, optimizer)
```

The above example had PyTorch flavour, but it should work just as well with any other framework you may want to use to write your training loops.

Note the absence of hideous `if (step_i + 1) % 0 == LOGGING_STEPS` style checks, and infinitely nested loops. Also, switch between training for steps vs epochs only requires changing 1 param.

## Features

- Event-based system to replace conditional checks
- Clean loop abstraction with state tracking
- Framework-agnostic (works with PyTorch, JAX, TensorFlow, MLX, etc.)
- Minimal dependencies (just Python standard library)
- Works with any iterable data source

## Coming soon

- Exceptions as Events, no more humongous try-except blocks
- Time base iteration limits and event triggers

## Installation

```bash
pip install dloop
```

## License

MIT License