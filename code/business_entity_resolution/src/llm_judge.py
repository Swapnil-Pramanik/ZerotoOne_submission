"""v7: a small instruction-tuned LLM as a judge for the hardest pairs only.

The judge sees the S1 business, up to three records already confidently matched to it
("siblings"), and the candidate record, and is asked whether the candidate belongs to the
same business. No text is generated: one forward pass per prompt, and the probability of
"Yes" versus "No" as the next token is the score. One model copy runs per GPU.

Models (open licences, <= 8B parameters): microsoft/Phi-3.5-mini-instruct (MIT, 3.8B) by
default, Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0, 1.5B) as a faster alternative.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

INSTRUCTION = ("You match business records from different databases. Records of the same business can have "
               "typos, abbreviations, reordered words, legal suffixes, a website instead of a name, another script, "
               "or partial addresses. Different businesses often have similar names. "
               "Answer only Yes or No.")


@dataclass
class JudgeConfig:
    model: str = "microsoft/Phi-3.5-mini-instruct"
    batch: int = 32
    max_len: int = 256
    max_siblings: int = 3


def prompt(business: str, siblings: list[str], candidate: str, cfg: JudgeConfig) -> str:
    sib = "\n".join(f"- {s}" for s in siblings[:cfg.max_siblings]) or "- (none known)"
    return (f"Business: {business}\nRecords known to belong to this business:\n{sib}\n"
            f"Candidate record: {candidate}\nDoes the candidate record belong to the same business?")


class Judge:
    def __init__(self, cfg: JudgeConfig = JudgeConfig()):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        n_gpu = torch.cuda.device_count()
        self.devices = [f"cuda:{i}" for i in range(n_gpu)] or ["cpu"]
        dtype = torch.float16 if n_gpu else torch.float32
        self.models = [AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(d).eval() for d in self.devices]
        self.yes = self._token_ids(["Yes", " Yes", "yes"])
        self.no = self._token_ids(["No", " No", "no"])

    def _token_ids(self, words):
        return sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in words})

    def _chat(self, text: str) -> str:
        msgs = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": text}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # templates without a system role
            msgs = [{"role": "user", "content": INSTRUCTION + "\n\n" + text}]
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def _score_batches(self, gpu: int, batches: list[list[str]]) -> list[np.ndarray]:
        model, dev = self.models[gpu], self.devices[gpu]
        out = []
        for texts in batches:
            x = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.cfg.max_len,
                         add_special_tokens=False).to(dev)
            logits = model(**x, logits_to_keep=1).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            yes = torch.logsumexp(lp[:, self.yes], dim=-1)
            no = torch.logsumexp(lp[:, self.no], dim=-1)
            out.append(torch.sigmoid(yes - no).cpu().numpy())
        return out

    def score(self, prompts: list[str], log=print) -> np.ndarray:
        """P(Yes) / (P(Yes) + P(No)) per prompt."""
        t0 = time.time()
        chats = [self._chat(p) for p in prompts]
        order = np.argsort([len(c) for c in chats], kind="stable")
        bs = self.cfg.batch
        batches = [[chats[i] for i in order[s:s + bs]] for s in range(0, len(order), bs)]
        k = len(self.models)
        shards = [batches[i::k] for i in range(k)]
        with ThreadPoolExecutor(k) as ex:
            results = list(ex.map(self._score_batches, range(k), shards))
        flat = [None] * len(batches)
        for i in range(k):
            flat[i::k] = results[i]
        scores = np.empty(len(prompts), np.float32)
        scores[order] = np.concatenate(flat) if flat else np.array([], np.float32)
        log(f"  llm judge: {len(prompts):,} prompts in {time.time() - t0:.0f}s "
            f"({len(prompts) / max(time.time() - t0, 1e-9):.0f}/s)")
        return scores
