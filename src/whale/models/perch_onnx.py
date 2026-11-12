"""Perch v2 as a frozen embedding model, run with ONNX Runtime (CPU or CUDA)."""

import os

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO_ID = "justinchuby/Perch-onnx"
EMBEDDING_DIM = 1536
SPATIAL_SHAPE = (16, 4, EMBEDDING_DIM)  # time frames x frequency bands x channels


def _allocated_cores() -> int:
    """Cores this process may run on (respects any CPU affinity limit)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _providers(device: str) -> list[str]:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if hasattr(ort, "preload_dlls"):  # onnxruntime-gpu[cuda,cudnn]: CUDA libs from pip
            ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(f"CUDA requested but onnxruntime offers only {ort.get_available_providers()}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")


class PerchEmbedder:
    """Maps waveforms [batch, 160000] (5 s at 32 kHz, float32) to embeddings
    [batch, 1536]. The mel front-end is inside the ONNX graph.

    The pooled ``embedding`` is the plain mean of ``spatial_embedding``
    [batch, 16, 4, 1536] over time and frequency (measured: identical)."""

    def __init__(self, filename: str = "perch_v2.onnx", threads: int | None = None, device: str = "cpu"):
        options = ort.SessionOptions()
        # Explicit count: left alone, ONNX Runtime sizes its pool by the whole
        # machine and oversubscribes a restricted CPU set.
        options.intra_op_num_threads = threads or _allocated_cores()
        options.inter_op_num_threads = 1
        model_path = hf_hub_download(REPO_ID, filename)
        self.session = ort.InferenceSession(model_path, options, providers=_providers(device))
        if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            # ONNX Runtime falls back to CPU silently when CUDA fails to initialise.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            raise RuntimeError(
                f"CUDA session fell back to {self.session.get_providers()} "
                f"(CUDA_VISIBLE_DEVICES={visible}; is a GPU visible?)"
            )
        self.input_name = self.session.get_inputs()[0].name

    def describe(self) -> dict:
        return {
            "providers": self.session.get_providers(),
            "inputs": [(i.name, i.shape, i.type) for i in self.session.get_inputs()],
            "outputs": [(o.name, o.shape, o.type) for o in self.session.get_outputs()],
        }

    def embed(self, waveforms: np.ndarray) -> np.ndarray:
        (embedding,) = self.session.run(["embedding"], {self.input_name: waveforms.astype(np.float32)})
        return embedding

    def embed_spatial(self, waveforms: np.ndarray) -> np.ndarray:
        """Spatial embeddings [batch, 16, 4, 1536] before pooling."""
        (spatial,) = self.session.run(["spatial_embedding"], {self.input_name: waveforms.astype(np.float32)})
        return spatial
