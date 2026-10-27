"""Feature exchange between isolated AF3 preprocessing and inference environments."""

import json
import pathlib

import numpy as np


def write_intermediates(examples, seeds, directory: pathlib.Path) -> None:
    """Write the same per-seed format as the CPU-optimized xfold runner."""
    if len(examples) != len(seeds):
        raise ValueError("The number of examples must match the number of seeds.")
    directory.mkdir(parents=True, exist_ok=True)
    for index, (seed, example) in enumerate(zip(seeds, examples, strict=True)):
        path = directory / f"seed-{seed}_{index}.npy"
        with path.open("wb") as stream:
            np.save(stream, example, allow_pickle=True)
        summary = {
            "format_version": 1,
            "npy_file": path.name,
            "parameter_count": len(example),
            "parameters": [
                {
                    "name": name,
                    "type": type(value).__name__,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for name, value in sorted(example.items())
            ],
        }
        path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def load_intermediates(seeds, directory: pathlib.Path) -> list[dict[str, np.ndarray]]:
    """Load trusted local features; object arrays include output atom layouts."""
    examples = []
    for index, seed in enumerate(seeds):
        path = directory / f"seed-{seed}_{index}.npy"
        with path.open("rb") as stream:
            example = np.load(stream, allow_pickle=True).item()
        if not isinstance(example, dict) or not all(
            isinstance(key, str) and isinstance(value, np.ndarray)
            for key, value in example.items()
        ):
            raise ValueError(f"Invalid featurised model input in {path}")
        examples.append(example)
    return examples


def load_preprocessed_inputs(directory: pathlib.Path, input_class):
    """Load only completed jobs listed by the preprocessing invocation."""
    manifest = json.loads((directory / "preprocessing_manifest.json").read_text())
    if manifest["format_version"] != 1 or manifest["alphafold3_version"] != "3.0.4":
        raise ValueError("Unsupported preprocessing manifest.")
    jobs = manifest["jobs"]
    if not jobs or len(jobs) != len(set(jobs)):
        raise ValueError("Preprocessing manifest must contain unique jobs.")
    inputs = []
    for job in jobs:
        if not isinstance(job, str) or pathlib.Path(job).name != job or job in (".", ".."):
            raise ValueError(f"Invalid preprocessing job directory: {job!r}")
        path = directory / job / f"{job}_data.json"
        fold_input = input_class.from_json(path.read_text())
        if fold_input.sanitised_name() != job:
            raise ValueError(f"Preprocessed input name does not match directory {job!r}.")
        inputs.append(fold_input)
    return inputs
