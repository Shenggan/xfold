"""Prepare reusable AF3 database shards and print launcher flags to stdout."""

import argparse
import contextlib
import fcntl
import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import TypedDict


DATABASES = (
    ("small_bfd", "bfd-first_non_consensus_sequences.fasta", False),
    ("mgnify", "mgy_clusters_2022_05.fa", False),
    ("uniprot_cluster_annot", "uniprot_all_2021_04.fa", False),
    ("uniref90", "uniref90_2022_05.fa", False),
    ("ntrna", "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta", True),
    ("rfam", "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta", True),
    ("rna_central", "rnacentral_active_seq_id_90_cov_80_linclust.fasta", True),
)


class SourceIdentity(TypedDict):
    path: str
    size: int
    mtime_ns: int


def _source_identity(source: pathlib.Path) -> SourceIdentity:
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def prepare_database(
    source: pathlib.Path, shard_root: pathlib.Path, num_shards: int
) -> tuple[str, int, int]:
    """Publish a complete round-robin split, preserving the original FASTA."""
    if not 2 <= num_shards <= 99_999:
        raise ValueError("The number of database shards must be between 2 and 99999.")
    identity = _source_identity(source)
    destination = shard_root / f"{source.name}.shards-{num_shards}"
    names = [
        f"{source.name}-{index:05d}-of-{num_shards:05d}"
        for index in range(num_shards)
    ]
    if destination.exists():
        metadata = json.loads((destination / "manifest.json").read_text())
        if (
            metadata["format_version"] != 1
            or metadata["source"] != identity
            or metadata["num_shards"] != num_shards
        ):
            raise ValueError(
                f"Stale shard cache: {destination}. Stop searches using this cache "
                "and move it aside, or choose a new AF3_SHARD_DIR."
            )
        for name in names:
            if (destination / name).stat().st_size != metadata["shard_sizes"][name]:
                raise ValueError(f"Incomplete or modified database shard: {destination / name}")
        sequences = metadata["sequence_count"]
        residues = metadata["residue_count"]
        if sequences < num_shards or residues <= 0:
            raise ValueError(f"Invalid sequence statistics in {destination / 'manifest.json'}")
        print(f"Reusing database shards: {destination}", file=sys.stderr)
    else:
        required = identity["size"] + max(identity["size"] // 20, 64 * 1024**2)
        if shutil.disk_usage(shard_root).free < required:
            raise OSError(
                f"Insufficient free space for {source}; need at least {required} bytes "
                f"in {shard_root}. Set AF3_SHARD_DIR to a larger filesystem."
            )
        print(
            f"Preparing {num_shards} shards for {source}; the first run scans and "
            "copies the full database. Original files are retained.",
            file=sys.stderr,
        )
        with tempfile.TemporaryDirectory(prefix=f".{source.name}-", dir=shard_root) as tmp:
            staging = pathlib.Path(tmp)
            sequences = 0
            residues = 0
            with contextlib.ExitStack() as stack:
                outputs = [
                    stack.enter_context((staging / name).open("wb", buffering=1024**2))
                    for name in names
                ]
                with source.open("rb", buffering=8 * 1024**2) as fasta:
                    for line in fasta:
                        if line.startswith(b">"):
                            sequences += 1
                        elif line.strip() and not line.startswith(b";"):
                            if sequences == 0:
                                raise ValueError(f"FASTA sequence before its header in {source}")
                            residues += len(b"".join(line.split()))
                        if sequences:
                            outputs[(sequences - 1) % num_shards].write(line)
                for output in outputs:
                    output.flush()
                    os.fsync(output.fileno())
            if sequences < num_shards or residues == 0:
                raise ValueError(f"Not enough FASTA records to shard {source}: {sequences}")
            if _source_identity(source) != identity:
                raise ValueError(f"Database changed while splitting: {source}")
            metadata = {
                "format_version": 1,
                "source": identity,
                "num_shards": num_shards,
                "sequence_count": sequences,
                "residue_count": residues,
                "shard_sizes": {name: (staging / name).stat().st_size for name in names},
            }
            (staging / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
            staging.rename(destination)
    return str(destination / source.name) + f"@{num_shards}", sequences, residues


def physical_cpu_count() -> int:
    """Count physical cores inside the launcher's CPU affinity allocation."""
    cpus = os.sched_getaffinity(0)
    cores = set()
    for cpu in cpus:
        topology = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        if not (topology / "physical_package_id").is_file() or not (topology / "core_id").is_file():
            print("CPU topology unavailable; using affinity CPU count for search budget.", file=sys.stderr)
            return len(cpus)
        cores.add(
            (
                (topology / "physical_package_id").read_text().strip(),
                (topology / "core_id").read_text().strip(),
            )
        )
    return len(cores)


def search_settings(
    cpu_budget: int,
    num_shards: int,
    jackhmmer_n_cpu: int | None = None,
    nhmmer_n_cpu: int | None = None,
    jackhmmer_max_parallel_shards: int | None = None,
    nhmmer_max_parallel_shards: int | None = None,
) -> dict[str, int]:
    if cpu_budget < 1:
        raise ValueError("Search CPU budget must be positive.")
    settings = {}
    for tool, databases, requested_cpus, requested_parallel in (
        ("jackhmmer", 4, jackhmmer_n_cpu, jackhmmer_max_parallel_shards),
        ("nhmmer", 3, nhmmer_n_cpu, nhmmer_max_parallel_shards),
    ):
        cpus = (
            requested_cpus
            if requested_cpus is not None
            else min(8, max(1, cpu_budget // databases - 1))
        )
        if cpus < 1 or (requested_parallel is not None and requested_parallel < 1):
            raise ValueError(f"{tool} CPU count and parallel shard limit must be positive.")
        # Reserve one coordinator per HMMER subprocess in addition to --cpu workers.
        parallel = min(
            num_shards,
            requested_parallel
            if requested_parallel is not None
            else max(1, cpu_budget // (databases * (cpus + 1))),
        )
        settings[f"{tool}_n_cpu"] = cpus
        settings[f"{tool}_max_parallel_shards"] = parallel
        demand = databases * parallel * (cpus + 1)
        print(
            f"{tool}: {databases} databases x {parallel} parallel shards x "
            f"({cpus} workers + 1 coordinator); CPU budget={cpu_budget}"
            + (" WARNING: requested concurrency exceeds budget." if demand > cpu_budget else ""),
            file=sys.stderr,
        )
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-dir", type=pathlib.Path, required=True)
    parser.add_argument("--shard-dir", type=pathlib.Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--cpu-budget", type=int)
    parser.add_argument("--jackhmmer-n-cpu", type=int)
    parser.add_argument("--nhmmer-n-cpu", type=int)
    parser.add_argument("--jackhmmer-max-parallel-shards", type=int)
    parser.add_argument("--nhmmer-max-parallel-shards", type=int)
    args = parser.parse_args()
    if not 2 <= args.num_shards <= 99_999:
        parser.error("--num-shards must be between 2 and 99999.")
    settings = search_settings(
        cpu_budget=args.cpu_budget if args.cpu_budget is not None else physical_cpu_count(),
        num_shards=args.num_shards,
        jackhmmer_n_cpu=args.jackhmmer_n_cpu,
        nhmmer_n_cpu=args.nhmmer_n_cpu,
        jackhmmer_max_parallel_shards=args.jackhmmer_max_parallel_shards,
        nhmmer_max_parallel_shards=args.nhmmer_max_parallel_shards,
    )
    for _, filename, _ in DATABASES:
        if not (args.db_dir / filename).is_file():
            raise FileNotFoundError(f"Missing source database: {args.db_dir / filename}")
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    flags = [f"--{name}={value}" for name, value in settings.items()]
    # Serialize preparation across simultaneous Step1 and end-to-end launches.
    with (args.shard_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, filename, is_rna in DATABASES:
            spec, sequences, residues = prepare_database(
                args.db_dir / filename, args.shard_dir, args.num_shards
            )
            z_value = residues / 1_000_000 if is_rna else sequences
            flags.extend((f"--{name}_database_path={spec}", f"--{name}_z_value={z_value}"))
    print("\n".join(flags))


if __name__ == "__main__":
    main()
