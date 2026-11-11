"""Perch v2 as a frozen embedding model, run with ONNX Runtime on CPU."""

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO_ID = "justinchuby/Perch-onnx"
EMBEDDING_DIM = 1536


class PerchEmbedder:
    """Maps waveforms [batch, 160000] (5 s at 32 kHz, float32) to embeddings
    [batch, 1536]. The mel front-end is inside the ONNX graph."""

    def __init__(self, filename: str = "perch_v2.onnx", threads: int | None = None):
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        model_path = hf_hub_download(REPO_ID, filename)
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def describe(self) -> dict:
        return {
            "inputs": [(i.name, i.shape, i.type) for i in self.session.get_inputs()],
            "outputs": [(o.name, o.shape, o.type) for o in self.session.get_outputs()],
        }

    def embed(self, waveforms: np.ndarray) -> np.ndarray:
        (embedding,) = self.session.run(["embedding"], {self.input_name: waveforms.astype(np.float32)})
        return embedding
