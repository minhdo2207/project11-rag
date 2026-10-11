"""Nạp HHEM-2.1-open (vectara/hallucination_evaluation_model) không qua trust_remote_code.

Mã đi kèm model lỗi với transformers 5.x (trọng số nằm ở meta tensor), nên ở đây dựng lại
đúng kiến trúc đó: flan-t5-base + đầu phân loại token, lấy logit ở vị trí đầu tiên.
"""

from __future__ import annotations

REPO = "vectara/hallucination_evaluation_model"
REVISION = "de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8"
FOUNDATION = "google/flan-t5-base"
PROMPT = (
    "<pad> Determine if the hypothesis is true given the premise?\n\n"
    "Premise: {premise}\n\nHypothesis: {hypothesis}"
)


class HHEM:
    def __init__(self, device: str = "cuda") -> None:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, T5ForTokenClassification

        state = load_file(hf_hub_download(REPO, "model.safetensors", revision=REVISION))
        state = {k.removeprefix("t5."): v for k, v in state.items()}
        # Embedding của encoder dùng chung trọng số với transformer.shared, file không lưu riêng.
        state.setdefault(
            "transformer.encoder.embed_tokens.weight", state["transformer.shared.weight"]
        )
        cfg = AutoConfig.from_pretrained(FOUNDATION)
        cfg.num_labels = state["classifier.weight"].shape[0]
        self.model = T5ForTokenClassification(cfg)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        self.model.to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(FOUNDATION)
        self.device = device

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        """pairs: list (premise, hypothesis). Trả về xác suất hypothesis được premise hỗ trợ."""
        import torch

        texts = [PROMPT.format(premise=p, hypothesis=h) for p, h in pairs]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # gom theo độ dài
        out = [0.0] * len(texts)
        with torch.no_grad():
            for s in range(0, len(order), batch_size):
                idx = order[s : s + batch_size]
                enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(
                    self.device
                )
                probs = torch.softmax(self.model(**enc).logits[:, 0, :], dim=-1)[:, 1]
                for i, p in zip(idx, probs.tolist(), strict=True):
                    out[i] = p
        return out
