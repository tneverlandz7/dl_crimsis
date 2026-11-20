"""Environmental run logging. Every run (debug, exploratory, final, failed) appends one row to env_tracker.csv.

Usage in code:
    with EnvRun("A-inf-textcnn-s0-dbg-01", part="A", task="informative", run_type="debug",
                model="textcnn", seed=0, batch_size=64) as run:
        ...  # training
        run.update(examples_processed=n, epochs=e, val_macro_f1=f1)

Manual entry (runs not wrapped in EnvRun):
    python -m src.env_logger --experiment-id ... --part A --task informative --run-type debug \
        --model textcnn --runtime-s 95 --run-status completed
"""

import argparse
import csv
import logging
import os
import platform
import subprocess
import time
from pathlib import Path

import psutil
import torch

from src import config

FIELDS = [
    "experiment_id", "part", "task", "run_type", "model", "seed", "examples_processed", "epochs",
    "batch_size", "device", "runtime_s", "energy_kwh", "carbon_intensity_gco2e_per_kwh",
    "emissions_kgco2e", "val_macro_f1", "run_status", "cache_checkpoint_used",
]
RUN_TYPES = {"debug", "exploratory", "final", "failed"}
RUN_STATUSES = {"completed", "early_stopped", "failed", "interrupted"}


def _cpu_name() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def describe_device(device: torch.device) -> str:
    ram_gb = psutil.virtual_memory().total / 1e9
    cpu = f"{_cpu_name()} x{os.cpu_count()} threads"
    if device.type == "cuda":
        gpu = f"{torch.cuda.get_device_name(device)} x{torch.cuda.device_count()}"
        return f"GPU {gpu} | CPU {cpu} | {ram_gb:.0f} GB RAM"
    return f"{device.type.upper()} {cpu} | {ram_gb:.0f} GB RAM"


def estimate_energy_kwh(runtime_s: float, uses_gpu: bool) -> float:
    ram_w = config.RAM_POWER_W_PER_GB * psutil.virtual_memory().total / 1e9
    gpu_w = config.GPU_POWER_W * config.GPU_UTIL if uses_gpu else 0.0
    watts = config.CPU_POWER_W * config.CPU_UTIL + gpu_w + ram_w
    return runtime_s / 3600 * watts * config.PUE / 1000


def existing_ids(path: Path = config.ENV_TRACKER) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {row["experiment_id"] for row in csv.DictReader(f)}


def append_row(row: dict, path: Path = config.ENV_TRACKER) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def finalise_row(row: dict, runtime_s: float, energy_kwh: float) -> dict:
    intensity = config.CARBON_INTENSITY_G_PER_KWH
    row.update(
        runtime_s=f"{runtime_s:.1f}",
        energy_kwh=f"{energy_kwh:.6g}",
        carbon_intensity_gco2e_per_kwh=intensity,
        emissions_kgco2e=f"{energy_kwh * intensity / 1000:.6g}",
    )
    return row


class EnvRun:
    def __init__(self, experiment_id: str, part: str, task: str, run_type: str, model: str,
                 seed: int | None = None, batch_size: int | None = None, cache_checkpoint_used: str = "",
                 device: str | torch.device | None = None, log_path: Path = config.ENV_TRACKER,
                 use_codecarbon: bool = True):
        if run_type not in RUN_TYPES:
            raise ValueError(f"run_type must be one of {sorted(RUN_TYPES)}")
        self.log_path = Path(log_path)
        if experiment_id in existing_ids(self.log_path):
            raise ValueError(f"experiment_id '{experiment_id}' already logged; IDs must be unique")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_codecarbon = use_codecarbon
        self.row = {field: "" for field in FIELDS}
        self.row.update(
            experiment_id=experiment_id, part=part, task=task, run_type=run_type, model=model,
            seed="" if seed is None else seed, batch_size="" if batch_size is None else batch_size,
            device=describe_device(self.device), cache_checkpoint_used=cache_checkpoint_used,
        )
        self._tracker = None

    def update(self, **fields) -> None:
        unknown = set(fields) - set(FIELDS)
        if unknown:
            raise KeyError(f"Unknown tracker fields: {sorted(unknown)}")
        if "run_status" in fields and fields["run_status"] not in RUN_STATUSES:
            raise ValueError(f"run_status must be one of {sorted(RUN_STATUSES)}")
        self.row.update(fields)

    def __enter__(self) -> "EnvRun":
        if self.use_codecarbon:
            try:
                from codecarbon import OfflineEmissionsTracker
                logging.getLogger("codecarbon").setLevel(logging.ERROR)
                self._tracker = OfflineEmissionsTracker(
                    country_iso_code=config.CODECARBON_COUNTRY, save_to_file=False,
                    log_level="error", tracking_mode="process",
                )
                self._tracker.start()
            except Exception:
                self._tracker = None
        self._start = time.perf_counter()
        return self

    def _energy(self, runtime_s: float) -> float:
        if self._tracker is not None:
            try:
                self._tracker.stop()
                return float(self._tracker.final_emissions_data.energy_consumed)
            except Exception:
                pass
        return estimate_energy_kwh(runtime_s, self.device.type == "cuda")

    def __exit__(self, exc_type, exc, tb) -> bool:
        runtime_s = time.perf_counter() - self._start
        energy_kwh = self._energy(runtime_s)
        if exc_type is KeyboardInterrupt:
            self.row["run_status"] = "interrupted"
        elif exc_type is not None:
            self.row.update(run_status="failed", run_type="failed")
        elif not self.row["run_status"]:
            self.row["run_status"] = "completed"
        append_row(finalise_row(self.row, runtime_s, energy_kwh), self.log_path)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a manually measured run to env_tracker.csv")
    derived = {"carbon_intensity_gco2e_per_kwh", "emissions_kgco2e"}
    required = {"experiment_id", "part", "task", "run_type", "model", "runtime_s", "run_status"}
    for field in FIELDS:
        if field not in derived:
            parser.add_argument(f"--{field.replace('_', '-')}", required=field in required, default="")
    args = vars(parser.parse_args())

    if args["run_type"] not in RUN_TYPES or args["run_status"] not in RUN_STATUSES:
        parser.error(f"run_type in {sorted(RUN_TYPES)}; run_status in {sorted(RUN_STATUSES)}")
    if args["experiment_id"] in existing_ids():
        parser.error(f"experiment_id '{args['experiment_id']}' already logged")

    runtime_s = float(args.pop("runtime_s"))
    energy_arg = args.pop("energy_kwh")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    energy_kwh = float(energy_arg) if energy_arg else estimate_energy_kwh(runtime_s, device.type == "cuda")
    row = {field: "" for field in FIELDS}
    row.update({k: v for k, v in args.items() if v != ""})
    row["device"] = row["device"] or describe_device(device)
    append_row(finalise_row(row, runtime_s, energy_kwh))
    print(f"Logged {row['experiment_id']}: {row['energy_kwh']} kWh, {row['emissions_kgco2e']} kgCO2e")


if __name__ == "__main__":
    main()
