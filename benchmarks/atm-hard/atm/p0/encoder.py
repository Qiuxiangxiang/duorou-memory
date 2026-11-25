"""Local, batched MiniLM encoder for the duorou embedding callback."""
import hashlib
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class MiniLM:
    def __init__(self, directory):
        model = directory / "model.onnx"
        tokenizer = directory / "tokenizer.json"
        self.hashes = {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                       for item in (model, tokenizer)}
        expected = "4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474"
        if self.hashes[model.name] != expected:
            raise ValueError("Unexpected ONNX file SHA-256")
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.tokenizer.enable_truncation(max_length=256)
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 6
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_names = [entry.name for entry in self.session.get_inputs()]

    @lru_cache(maxsize=2048)
    def query(self, text):
        return self.encode([text])

    def __call__(self, texts):
        return self.query(texts[0]) if len(texts) == 1 else self.encode(texts)

    def encode(self, texts):
        tokens = self.tokenizer.encode_batch(texts)
        order = sorted(range(len(tokens)), key=lambda i: len(tokens[i].ids))
        output = np.empty((len(tokens), 384), dtype=np.float32)
        for start in range(0, len(order), 48):
            indices = order[start:start + 48]
            width = len(tokens[indices[-1]].ids)
            ids = np.full((len(indices), width), self.pad_id, dtype=np.int64)
            mask = np.zeros_like(ids)
            for row, idx in enumerate(indices):
                seq = tokens[idx].ids
                ids[row, :len(seq)] = seq
                mask[row, :len(seq)] = 1
            available = {"input_ids": ids, "attention_mask": mask,
                         "token_type_ids": np.zeros_like(ids)}
            hidden = self.session.run(None, {name: available[name] for name in self.input_names})[0]
            output[indices] = (hidden * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
            if len(order) > 1000 and start % 2400 == 0:
                print(f"encoded {min(start + len(indices), len(order))}/{len(order)}", flush=True)
        return output
