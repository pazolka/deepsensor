<!-- Thank you for your contribution to DeepSensor! -->
<!-- Please fill out the details in the pull request description below as appropriate -->
<!-- Ensure that you abide by the code of conduct -->

## :pencil: Description
This pull request introduces support for optimizer weight decay in the PyTorch trainer and refactors tensor conversion checks to prevent premature conversion of wrapper objects.

### Changes:
1. **Weight Decay Support in `Trainer` (`deepsensor/train/train.py`)**:
   - Added a `weight_decay: float = 0.0` parameter to the `Trainer` constructor (defaulting to `0.0` to maintain backward compatibility).
   - Passed the `weight_decay` parameter to the PyTorch `optim.Adam` optimizer. This enables standard L2 weight regularization when training models.

2. **Robust Attribute Extraction in `_to_torch_float` (`deepsensor/model/gridded_tnp.py`)**:
   - Reordered checks in the `_to_torch_float` helper function.
   - Specifically, attribute extraction checks (checking for `.y` or `.data` attributes) are now performed *before* checking `torch.is_tensor(value)`. 
   - **Why**: This prevents wrapper/container objects (which may be recognized as tensors or subclass them, but still carry the target payload within a nested attribute) from bypassing the attribute-extraction steps.

## :white_check_mark: Checklist before requesting a review
(See the contributing guide for more details on these steps.)
- [ ] I have installed developer dependencies with `pip install .[dev]` and running `pre-commit install` (or alternatively, manually running `ruff format` before commiting)

If changing or adding source code:
- [ ] tests are included and are passing (run `pytest`).
- [ ] documentation is included or updated as relevant, including docstrings.

If changing or adding documentation:
- [ ] docs build successfully (`jupyter-book build docs --all`) and the changes look good from a manual inspection of the HTML in `docs/_build/html/`.
