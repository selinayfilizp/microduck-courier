"""Run an exported policy ONNX as a drop-in inference callable.

The export path (scripts/export.py) bakes the observation normalizer into the
graph, so the session consumes RAW actor observations and returns deterministic
mean actions. This makes the committed artifact itself evaluable: anyone who
clones the repo can reproduce the delivery numbers without the training
checkpoint, and CI does exactly that on every push.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class OnnxPolicy:
    """Callable matching the runner's inference-policy interface.

    Accepts the TensorDict observation batch produced by RslRlVecEnvWrapper,
    feeds the actor group through the ONNX session, and returns actions as a
    torch tensor on the requested device. Handles fixed-batch exports (the
    courier artifact is exported with batch 1) by running row-wise.
    """

    def __init__(self, path: str | Path, device: str = "cpu") -> None:
        import onnxruntime as ort

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {self.path}")
        self._session = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        batch = inp.shape[0]
        self._fixed_batch = batch if isinstance(batch, int) else None
        self._device = device

    def __call__(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            actor = obs
        elif "actor" in obs.keys():
            actor = obs["actor"]
        else:
            actor = obs["policy"]
        x = actor.detach().to("cpu", torch.float32).numpy()
        if self._fixed_batch is not None and x.shape[0] != self._fixed_batch:
            rows = [
                self._session.run(None, {self._input_name: x[i : i + 1]})[0]
                for i in range(x.shape[0])
            ]
            y = np.concatenate(rows, axis=0)
        else:
            y = self._session.run(None, {self._input_name: x})[0]
        return torch.from_numpy(y).to(self._device)
