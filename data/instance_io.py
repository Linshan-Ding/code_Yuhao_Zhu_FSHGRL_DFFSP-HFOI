"""Instance specification and parser ."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class InstanceSpec:
    path: Path
    product_count: int
    stage_count: int
    machine_count: int
    machine_division: Tuple[int, ...]
    stage_machines: Tuple[Tuple[int, ...], ...]
    process_times: np.ndarray
    order_ids: Tuple[str, ...]
    product_types: np.ndarray
    arrival_times: np.ndarray
    due_dates: np.ndarray
    task_to_product_stage: Tuple[Tuple[int, int], ...]
    ope_ma_adj: torch.Tensor
    ope_pre_adj: torch.Tensor
    ope_sub_adj: torch.Tensor
    proc_rates: torch.Tensor

    @property
    def total_orders(self) -> int:
        return len(self.order_ids)

    @property
    def task_count(self) -> int:
        return len(self.task_to_product_stage)


def parse_instance_file(path: str | Path) -> InstanceSpec:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if len(lines) < 6:
        raise ValueError(f"Instance file has too few lines: {path}")

    product_count, stage_count, machine_count = map(int, lines[0].split())
    machine_division = tuple(map(int, lines[1].split()))
    if len(machine_division) != stage_count:
        raise ValueError(f"Machine division length does not match stage count in {path}")

    process_rows = []
    for row_index in range(product_count):
        values = list(map(float, lines[2 + row_index].split()))
        if len(values) != machine_count:
            raise ValueError(f"Process-time row {row_index} has wrong length in {path}")
        process_rows.append(values)
    process_times = np.asarray(process_rows, dtype=np.float32)

    offset = 2 + product_count
    order_ids = tuple(lines[offset].split())
    product_types = np.asarray(list(map(int, lines[offset + 1].split())), dtype=np.int64)
    arrival_times = np.asarray(list(map(float, lines[offset + 2].split())), dtype=np.float32)
    due_dates = np.asarray(list(map(float, lines[offset + 3].split())), dtype=np.float32)
    if not (len(order_ids) == len(product_types) == len(arrival_times) == len(due_dates)):
        raise ValueError(f"Order fields have inconsistent lengths in {path}")

    order = np.argsort(arrival_times, kind="stable")
    order_ids = tuple(order_ids[index] for index in order)
    product_types = product_types[order]
    arrival_times = arrival_times[order]
    due_dates = due_dates[order]

    stage_machines = []
    machine_offset = 0
    for count in machine_division:
        machines = tuple(range(machine_offset, machine_offset + count))
        stage_machines.append(machines)
        machine_offset += count
    stage_machines = tuple(stage_machines)

    task_to_product_stage = tuple(
        (product, stage)
        for product in range(product_count)
        for stage in range(stage_count)
    )
    task_count = len(task_to_product_stage)

    ope_ma_adj = torch.zeros((task_count, machine_count), dtype=torch.bool)
    proc_rates = torch.zeros((task_count, machine_count), dtype=torch.float32)
    for task_index, (product, stage) in enumerate(task_to_product_stage):
        for machine in stage_machines[stage]:
            process_time = float(process_times[product, machine])
            if process_time > 0:
                ope_ma_adj[task_index, machine] = True
                proc_rates[task_index, machine] = 1.0 / process_time

    ope_pre_adj = torch.zeros((task_count, task_count), dtype=torch.bool)
    ope_sub_adj = torch.zeros((task_count, task_count), dtype=torch.bool)
    for product in range(product_count):
        for stage in range(1, stage_count):
            current_index = product * stage_count + stage
            previous_index = product * stage_count + stage - 1
            ope_pre_adj[current_index, previous_index] = True
            ope_sub_adj[previous_index, current_index] = True

    return InstanceSpec(
        path=path,
        product_count=product_count,
        stage_count=stage_count,
        machine_count=machine_count,
        machine_division=machine_division,
        stage_machines=stage_machines,
        process_times=process_times,
        order_ids=order_ids,
        product_types=product_types,
        arrival_times=arrival_times,
        due_dates=due_dates,
        task_to_product_stage=task_to_product_stage,
        ope_ma_adj=ope_ma_adj,
        ope_pre_adj=ope_pre_adj,
        ope_sub_adj=ope_sub_adj,
        proc_rates=proc_rates,
    )
