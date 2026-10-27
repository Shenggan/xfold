"""Run sharded MSA/template search and featurisation in an isolated AF3 3.0.4 env."""

import argparse
import contextlib
import datetime
import fcntl
import importlib.metadata
import json
import os
import pathlib
import shutil
import time

# Capture the allocation before libraries initialize OpenMP.
_INITIAL_CPU_AFFINITY = os.sched_getaffinity(0)

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation, pipeline

from prepare_msa_databases import DATABASES, physical_cpu_count, prepare_database, search_settings
from xfold.preprocessing import write_intermediates


@contextlib.contextmanager
def _data_pipeline_affinity():
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, _INITIAL_CPU_AFFINITY)
    try:
        yield
    finally:
        os.sched_setaffinity(0, previous)


def make_pipeline_config(args) -> pipeline.DataPipelineConfig:
    """Reuse round-robin shards, with full-database statistics for E-values."""
    options = {
        f"{name}_binary_path": getattr(args, f"{name}_binary_path")
        for name in ("jackhmmer", "nhmmer", "hmmalign", "hmmsearch", "hmmbuild")
    }
    options.update(search_settings(
        cpu_budget=args.cpu_budget if args.cpu_budget is not None else physical_cpu_count(),
        num_shards=args.num_shards,
        jackhmmer_n_cpu=args.jackhmmer_n_cpu,
        nhmmer_n_cpu=args.nhmmer_n_cpu,
        jackhmmer_max_parallel_shards=args.jackhmmer_max_parallel_shards,
        nhmmer_max_parallel_shards=args.nhmmer_max_parallel_shards,
    ))
    for _, filename, _ in DATABASES:
        if not (args.db_dir / filename).is_file():
            raise FileNotFoundError(f"Missing source database: {args.db_dir / filename}")
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    with (args.shard_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, filename, is_rna in DATABASES:
            spec, sequences, residues = prepare_database(
                args.db_dir / filename, args.shard_dir, args.num_shards
            )
            options[f"{name}_database_path"] = spec
            options[f"{name}_z_value"] = residues / 1_000_000 if is_rna else sequences
    options.update(
        pdb_database_path=str(args.pdb_database_path or args.db_dir / "mmcif_files"),
        seqres_database_path=str(
            args.seqres_database_path or args.db_dir / "pdb_seqres_2022_09_28.fasta"
        ),
        max_template_date=datetime.date.fromisoformat(args.max_template_date),
    )
    return pipeline.DataPipelineConfig(**options)


def preprocess(fold_input, data_pipeline, output_dir, args):
    start = time.monotonic()
    if data_pipeline is not None:
        with _data_pipeline_affinity():
            fold_input = data_pipeline.process(fold_input)
    examples = featurisation.featurise_input(
        fold_input=fold_input,
        ccd=chemical_components.Ccd(user_ccd=fold_input.user_ccd),
        buckets=None,
        verbose=True,
        ref_max_modified_date=datetime.date.fromisoformat(args.max_template_date),
        conformer_max_iterations=args.conformer_max_iterations,
        resolve_msa_overlaps=args.resolve_msa_overlaps,
        fix_standalone_glycans=args.fix_standalone_glycans,
    )
    # to_json embeds resolved MSA/templates/CCD; its fields are v1-compatible.
    # Only the exchange JSON is downgraded, never an arbitrary user input.
    payload = json.loads(fold_input.to_json())
    payload["version"] = 1
    name = fold_input.sanitised_name()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_data.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_intermediates(examples, fold_input.rng_seeds, output_dir / "intermediates")
    print(f"Preprocessing {fold_input.name} took {time.monotonic() - start:.2f} seconds.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json_path", type=pathlib.Path)
    source.add_argument("--input_dir", type=pathlib.Path)
    parser.add_argument("--db_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, required=True)
    parser.add_argument("--shard_dir", type=pathlib.Path)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--cpu_budget", type=int)
    for name in ("jackhmmer", "nhmmer"):
        parser.add_argument(f"--{name}_n_cpu", type=int)
        parser.add_argument(f"--{name}_max_parallel_shards", type=int)
    for name in ("jackhmmer", "nhmmer", "hmmalign", "hmmsearch", "hmmbuild"):
        parser.add_argument(f"--{name}_binary_path", default=shutil.which(name))
    parser.add_argument("--pdb_database_path", type=pathlib.Path)
    parser.add_argument("--seqres_database_path", type=pathlib.Path)
    parser.add_argument("--max_template_date", default="2021-09-30")
    parser.add_argument("--conformer_max_iterations", type=int)
    parser.add_argument("--resolve_msa_overlaps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fix_standalone_glycans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_data_pipeline", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if importlib.metadata.version("alphafold3") != "3.0.4":
        parser.error("Use a separate AlphaFold 3.0.4 environment, not the 3.0.0 inference env.")
    if not 2 <= args.num_shards <= 99_999:
        parser.error("--num_shards must be between 2 and 99999.")
    args.db_dir = args.db_dir.resolve()
    args.shard_dir = (args.shard_dir or args.db_dir / "sharded").resolve()
    inputs = list(
        folding_input.load_fold_inputs_from_path(args.json_path)
        if args.json_path is not None
        else folding_input.load_fold_inputs_from_dir(args.input_dir)
    )
    names = [item.sanitised_name() for item in inputs]
    if not names or any(name in ("", ".", "..") for name in names) or len(set(names)) != len(names):
        raise ValueError("Inputs must have non-empty, unique sanitised job names.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "preprocessing_manifest.json"
    with _data_pipeline_affinity():
        data_pipeline = (
            pipeline.DataPipeline(make_pipeline_config(args))
            if args.run_data_pipeline else None
        )
    for fold_input, name in zip(inputs, names, strict=True):
        preprocess(
            fold_input, data_pipeline, args.output_dir / name, args
        )
    manifest_path.write_text(json.dumps({
        "format_version": 1,
        "alphafold3_version": "3.0.4",
        "jobs": names,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
