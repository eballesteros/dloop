# dloop

A lightweight Python library for simpler deep learning training loops.

## What is it?

`dloop` is a small library that helps clean up those messy training loops without imposing a whole framework on you. It won't allow you to train a neural network in 4 lines of code, nor will it auto-distribute your training run to thousands of GPUS, and it certainly wont make your model perform better nor converge faster... but if you enjoy rolling your own training loops it may handle some inconveniences for you.

Lest jump straight into an example. This is how your training loop could look like with `dloop`:

```python
from dloop import Event, Loop, LoopEvents

train_loop = Loop(
    dataloader, 
    max_epochs=3, # specify ONE stopping condition: epochs, steps, or time (in seconds)
    # max_steps=50_000, # alternative: stop after 50k steps
    # max_seconds=3600, # alternative: stop after 1 hour of training
    events={ # [1] 
        "ModelParametersUpdate": Event(every_n_steps=16), 
        "Logging": Event(every_n_steps=100),
        "DecreaseLR": Event(at_step=10_000),
        "HourlyCheckpoint": Event(every_n_seconds=3600),
        "TenMinuteWarning": Event(at_time=50 * 60)  # 50 minutes into training
    }
)

for batch, batch_events in train_loop: # [2]
    # <your-code> run forward and compute gradients </your-code>
    
    if "ModelParametersUpdate" in batch_events: # [3]
        # <your-code> update model parameter </your-code>
        
    if "Logging" in batch_events: # [4]
        # <your-code> log progress </your-code>

    if "DecreaseLR" in batch_events: # [5]
        # <your-code> decrease the learning rate </your-code>
        
    if "HourlyCheckpoint" in batch_events: # [6]
        # <your-code> save checkpoint every hour </your-code>
        
    if "TenMinuteWarning" in batch_events: # [7]
        # <your-code> log warning that training will end soon </your-code>
        
    if LoopEvents.EPOCH_END in batch_events: # [8]
        # <your-code> run validation? Checkpoint? </your-code>
```

- [1]: You define custom events by just giving them a name and specifying when they should be triggered. Currently, we support triggers `every_n_steps`, `at_step`, `every_n_seconds`, `at_time`, or through an arbitrary `condition_function(loop_state)`.
- [2]: `train_loop` will yield as many batches as you've specified, and then stop when appropriate, looping over the dataloader under the hood if necessary.
- [3-8]: `batch_events` will contain events that were triggered for this iteration, allowing to handle all iteration and error related logic with simple if statements:
    - `# 3` and `# 4` are examples of checks that will evaluate to true every n steps. For example, since `ModelParametersUpdate` triggers every 16 steps, condition `# 3` will evaluate to true in steps 15, 31, 47 etc, allowing you to implement gradient accumulation.
    - `# 5` shows how you can use the same kind of logic for one-time step-based events (for example decreasing your LR after 10k steps).
    - `# 6` demonstrates time-based recurring events that trigger on a regular time interval (e.g., saving checkpoints every hour).
    - `# 7` shows one-time time-based events that trigger at a specific time since training started.
    - Finally, `# 8` shows usage of other pre-defined events (like `LoopEvents.EPOCH_END` and `LoopEvents.TRAINING_END`) which are automatically added by dloop


Note how dloop is completely framework agnostic. All `<your-code>` blocks are completely up to you to write using your preferred framework.


## Features

- Clean loop abstraction with state tracking
- Multiple iteration limit options (by epochs, steps, or time)
- Event-based system to replace conditional checks
  - Step-based events: trigger on specific steps or every N steps
  - Time-based events: trigger at specific times or every N seconds
  - Custom condition events: trigger based on any logic
- Framework-agnostic (works with PyTorch, JAX, TensorFlow, MLX, etc.)
- Minimal dependencies (just Python standard library)
- Works with any iterable data source


## Installation

```bash
pip install dloop
```

## Development

### Release Process

This project uses a streamlined release process to publish to PyPI automatically when a release branch is merged to main.

#### Automated CI/CD Pipeline

When a branch following the pattern `release/vX.Y.Z` is merged into `main`, our CI/CD pipeline automatically:

1. Verifies that the version in the branch name matches the version in `pyproject.toml`
2. Runs linting and tests as quality gates
3. Builds the Python package
4. Creates a GitHub release with the version tag and package files
5. Publishes the package to PyPI

#### Making a New Release

To release a new version:

1. Start from the latest `main` and update the version:
    ```bash
    git checkout main
    git pull origin main

    # Bump the version
    poetry version patch  # or minor or major

    # Store the new version in a variable
    VERSION=$(poetry version --short)
    ```

2. Create a release branch with the correct version name:
    ```bash
    git checkout -b release/v$VERSION
    ```

3. Make any other release-specific changes.

4. Commit your changes:
    ```bash
    git add pyproject.toml
    git commit -m "Prepare release $VERSION"
    ```

5. Push the release branch and create a Pull Request:
    ```bash
    # Create a PR targeting main using the GitHub UI
    git push -u origin release/v$VERSION
    ```

6. After code review, merge the PR into `main`.

7. The CI/CD pipeline will automatically handle the rest:
   - Creating a git tag
   - Creating a GitHub release with release notes
   - Publishing to PyPI


## License

MIT License