#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


SUPPORTED_MODELS = {"complex", "rotate", "distmult", "transe"}
SUPPORTED_DUMP_MODES = {"head", "tail", "both"}
SUPPORTED_SPLITS = {"train", "valid", "test"}
SUPPORTED_OPTIMIZERS = {"adam", "adagrad", "sgd"}


@dataclass
class DatasetStats:
    num_entities: int
    num_relations: int
    train_size: int
    valid_size: int
    test_size: int


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_triples(path: Path) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def read_mapped_triples(
    path: Path,
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
) -> torch.LongTensor:
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            if h in entity_to_id and r in relation_to_id and t in entity_to_id:
                rows.append([entity_to_id[h], relation_to_id[r], entity_to_id[t]])
    if not rows:
        raise ValueError(f"No valid triples found in {path}")
    return torch.as_tensor(rows, dtype=torch.long)


def load_processed_dataset(
    processed_dir: Path,
) -> Tuple[
    Dict[str, int],
    Dict[str, int],
    torch.LongTensor,
    torch.LongTensor,
    torch.LongTensor,
    DatasetStats,
]:
    entity2id_path = processed_dir / "entity2id.json"
    relation2id_path = processed_dir / "relation2id.json"
    train_path = processed_dir / "train.tsv"
    valid_path = processed_dir / "valid.tsv"
    test_path = processed_dir / "test.tsv"

    for p in [entity2id_path, relation2id_path, train_path, valid_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required processed dataset file: {p}")

    entity_to_id = load_json(entity2id_path)
    relation_to_id = load_json(relation2id_path)

    train_mapped = read_mapped_triples(train_path, entity_to_id, relation_to_id)
    valid_mapped = read_mapped_triples(valid_path, entity_to_id, relation_to_id)
    test_mapped = read_mapped_triples(test_path, entity_to_id, relation_to_id)

    stats = DatasetStats(
        num_entities=len(entity_to_id),
        num_relations=len(relation_to_id),
        train_size=int(train_mapped.shape[0]),
        valid_size=int(valid_mapped.shape[0]),
        test_size=int(test_mapped.shape[0]),
    )
    return entity_to_id, relation_to_id, train_mapped, valid_mapped, test_mapped, stats


def _maybe_set_allocator_env(expandable_segments: bool) -> None:
    if expandable_segments:
        current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" not in current:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                (current + "," if current else "") + "expandable_segments:True"
            )


def _maybe_clear_cuda_cache(clear_cache: bool) -> None:
    if clear_cache and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if is_main_process():
                print("CUDA cache cleared")
        except Exception as e:
            if is_main_process():
                print(f"WARNING: CUDA cache clear failed: {e}")


class TripleDataset(Dataset):
    def __init__(self, mapped_triples: torch.LongTensor):
        self.mapped_triples = mapped_triples

    def __len__(self) -> int:
        return int(self.mapped_triples.shape[0])

    def __getitem__(self, idx: int) -> torch.LongTensor:
        return self.mapped_triples[idx]


class KGEModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        gamma: float = 12.0,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.gamma = gamma

        if self.model_name == "complex":
            self.entity_emb = nn.Embedding(num_entities, 2 * embedding_dim)
            self.relation_emb = nn.Embedding(num_relations, 2 * embedding_dim)
        elif self.model_name == "rotate":
            self.entity_emb = nn.Embedding(num_entities, 2 * embedding_dim)
            self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        else:
            self.entity_emb = nn.Embedding(num_entities, embedding_dim)
            self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.model_name == "transe":
            bound = 6.0 / math.sqrt(self.embedding_dim)
            nn.init.uniform_(self.entity_emb.weight, -bound, bound)
            nn.init.uniform_(self.relation_emb.weight, -bound, bound)
        elif self.model_name == "rotate":
            nn.init.uniform_(self.entity_emb.weight, -1.0, 1.0)
            nn.init.uniform_(self.relation_emb.weight, -math.pi, math.pi)
        else:
            nn.init.xavier_uniform_(self.entity_emb.weight)
            nn.init.xavier_uniform_(self.relation_emb.weight)

    def score_hrt(self, h_idx: torch.LongTensor, r_idx: torch.LongTensor, t_idx: torch.LongTensor) -> torch.Tensor:
        if self.model_name == "transe":
            h = self.entity_emb(h_idx)
            r = self.relation_emb(r_idx)
            t = self.entity_emb(t_idx)
            return -torch.linalg.norm(h + r - t, ord=1, dim=-1)

        if self.model_name == "distmult":
            h = self.entity_emb(h_idx)
            r = self.relation_emb(r_idx)
            t = self.entity_emb(t_idx)
            return (h * r * t).sum(dim=-1)

        if self.model_name == "complex":
            h = self.entity_emb(h_idx)
            r = self.relation_emb(r_idx)
            t = self.entity_emb(t_idx)
            h_re, h_im = h[..., :self.embedding_dim], h[..., self.embedding_dim:]
            r_re, r_im = r[..., :self.embedding_dim], r[..., self.embedding_dim:]
            t_re, t_im = t[..., :self.embedding_dim], t[..., self.embedding_dim:]
            return (
                h_re * r_re * t_re
                + h_im * r_re * t_im
                + h_re * r_im * t_im
                - h_im * r_im * t_re
            ).sum(dim=-1)

        if self.model_name == "rotate":
            h = self.entity_emb(h_idx)
            t = self.entity_emb(t_idx)
            phase = self.relation_emb(r_idx)
            h_re, h_im = h[..., :self.embedding_dim], h[..., self.embedding_dim:]
            t_re, t_im = t[..., :self.embedding_dim], t[..., self.embedding_dim:]
            r_re = torch.cos(phase)
            r_im = torch.sin(phase)
            rot_re = h_re * r_re - h_im * r_im
            rot_im = h_re * r_im + h_im * r_re
            diff_re = rot_re - t_re
            diff_im = rot_im - t_im
            return self.gamma - torch.sqrt(diff_re.pow(2) + diff_im.pow(2) + 1e-12).sum(dim=-1)

        raise ValueError(f"Unsupported model: {self.model_name}")

    def score_t(self, hr_batch: torch.LongTensor) -> torch.Tensor:
        h_idx = hr_batch[:, 0]
        r_idx = hr_batch[:, 1]

        if self.model_name == "transe":
            h = self.entity_emb(h_idx)
            r = self.relation_emb(r_idx)
            x = h + r
            all_t = self.entity_emb.weight
            return -torch.cdist(x, all_t, p=1)

        if self.model_name == "distmult":
            h = self.entity_emb(h_idx)
            r = self.relation_emb(r_idx)
            x = h * r
            return x @ self.entity_emb.weight.t()

        if self.model_name == "complex":
            h = self.entity_emb(h_idx)
            r = self.relation_emb(r_idx)
            h_re, h_im = h[:, :self.embedding_dim], h[:, self.embedding_dim:]
            r_re, r_im = r[:, :self.embedding_dim], r[:, self.embedding_dim:]
            x_re = h_re * r_re - h_im * r_im
            x_im = h_re * r_im + h_im * r_re

            all_t = self.entity_emb.weight
            t_re, t_im = all_t[:, :self.embedding_dim], all_t[:, self.embedding_dim:]
            return x_re @ t_re.t() + x_im @ t_im.t()

        if self.model_name == "rotate":
            h = self.entity_emb(h_idx)
            phase = self.relation_emb(r_idx)
            h_re, h_im = h[:, :self.embedding_dim], h[:, self.embedding_dim:]
            r_re = torch.cos(phase)
            r_im = torch.sin(phase)
            rot_re = h_re * r_re - h_im * r_im
            rot_im = h_re * r_im + h_im * r_re

            all_t = self.entity_emb.weight
            t_re, t_im = all_t[:, :self.embedding_dim], all_t[:, self.embedding_dim:]

            diff_re = rot_re[:, None, :] - t_re[None, :, :]
            diff_im = rot_im[:, None, :] - t_im[None, :, :]
            dist_val = torch.sqrt(diff_re.pow(2) + diff_im.pow(2) + 1e-12).sum(dim=-1)
            return self.gamma - dist_val

        raise ValueError(f"Unsupported model: {self.model_name}")

    def score_h(self, rt_batch: torch.LongTensor) -> torch.Tensor:
        r_idx = rt_batch[:, 0]
        t_idx = rt_batch[:, 1]

        if self.model_name == "transe":
            r = self.relation_emb(r_idx)
            t = self.entity_emb(t_idx)
            x = t - r
            all_h = self.entity_emb.weight
            return -torch.cdist(x, all_h, p=1)

        if self.model_name == "distmult":
            r = self.relation_emb(r_idx)
            t = self.entity_emb(t_idx)
            x = r * t
            return x @ self.entity_emb.weight.t()

        if self.model_name == "complex":
            r = self.relation_emb(r_idx)
            t = self.entity_emb(t_idx)
            r_re, r_im = r[:, :self.embedding_dim], r[:, self.embedding_dim:]
            t_re, t_im = t[:, :self.embedding_dim], t[:, self.embedding_dim:]

            x_re = r_re * t_re + r_im * t_im
            x_im = r_re * t_im - r_im * t_re

            all_h = self.entity_emb.weight
            h_re, h_im = all_h[:, :self.embedding_dim], all_h[:, self.embedding_dim:]
            return x_re @ h_re.t() + x_im @ h_im.t()

        if self.model_name == "rotate":
            phase = self.relation_emb(r_idx)
            t = self.entity_emb(t_idx)
            t_re, t_im = t[:, :self.embedding_dim], t[:, self.embedding_dim:]
            r_re = torch.cos(phase)
            r_im = torch.sin(phase)

            inv_re = r_re
            inv_im = -r_im
            h_re = t_re * inv_re - t_im * inv_im
            h_im = t_re * inv_im + t_im * inv_re

            all_h = self.entity_emb.weight
            a_re, a_im = all_h[:, :self.embedding_dim], all_h[:, self.embedding_dim:]
            diff_re = h_re[:, None, :] - a_re[None, :, :]
            diff_im = h_im[:, None, :] - a_im[None, :, :]
            dist_val = torch.sqrt(diff_re.pow(2) + diff_im.pow(2) + 1e-12).sum(dim=-1)
            return self.gamma - dist_val

        raise ValueError(f"Unsupported model: {self.model_name}")


def sample_negative_tails(
    batch_size: int,
    num_negs: int,
    num_entities: int,
    device: torch.device,
) -> torch.LongTensor:
    return torch.randint(
        low=0,
        high=num_entities,
        size=(batch_size, num_negs),
        device=device,
        dtype=torch.long,
    )


def sample_negative_heads(
    batch_size: int,
    num_negs: int,
    num_entities: int,
    device: torch.device,
) -> torch.LongTensor:
    return torch.randint(
        low=0,
        high=num_entities,
        size=(batch_size, num_negs),
        device=device,
        dtype=torch.long,
    )


def compute_batch_loss(
    model: KGEModel,
    batch: torch.LongTensor,
    num_entities: int,
    num_negs: int,
    adv_temperature: float = 1.0,
) -> torch.Tensor:
    device = batch.device
    h = batch[:, 0]
    r = batch[:, 1]
    t = batch[:, 2]

    pos_scores = model.score_hrt(h, r, t)

    bsz = batch.shape[0]
    neg_t = sample_negative_tails(bsz, num_negs, num_entities, device)
    neg_h = sample_negative_heads(bsz, num_negs, num_entities, device)

    h_tail = h.unsqueeze(1).expand(-1, num_negs)
    r_tail = r.unsqueeze(1).expand(-1, num_negs)
    t_head = t.unsqueeze(1).expand(-1, num_negs)
    r_head = r.unsqueeze(1).expand(-1, num_negs)

    neg_tail_scores = model.score_hrt(
        h_tail.reshape(-1),
        r_tail.reshape(-1),
        neg_t.reshape(-1),
    ).view(bsz, num_negs)

    neg_head_scores = model.score_hrt(
        neg_h.reshape(-1),
        r_head.reshape(-1),
        t_head.reshape(-1),
    ).view(bsz, num_negs)

    neg_scores = torch.cat([neg_tail_scores, neg_head_scores], dim=1)

    pos_loss = -F.logsigmoid(pos_scores).mean()
    neg_weights = F.softmax(neg_scores * adv_temperature, dim=1).detach()
    neg_loss = -(neg_weights * F.logsigmoid(-neg_scores)).sum(dim=1).mean()

    return 0.5 * (pos_loss + neg_loss)


@torch.no_grad()
def mean_reciprocal_rank(
    model: KGEModel,
    mapped_triples: torch.LongTensor,
    batch_size_eval: int,
    device: torch.device,
    max_queries: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    triples = mapped_triples[:max_queries] if max_queries is not None else mapped_triples
    n = int(triples.shape[0])

    mrr_tail = 0.0
    mrr_head = 0.0
    hits10_tail = 0.0
    hits10_head = 0.0

    for start in range(0, n, batch_size_eval):
        batch = triples[start:start + batch_size_eval].to(device)
        h = batch[:, 0]
        r = batch[:, 1]
        t = batch[:, 2]

        tail_scores = model.score_t(torch.stack([h, r], dim=1))
        true_tail_scores = tail_scores.gather(1, t.unsqueeze(1))
        tail_rank = (tail_scores > true_tail_scores).sum(dim=1) + 1

        head_scores = model.score_h(torch.stack([r, t], dim=1))
        true_head_scores = head_scores.gather(1, h.unsqueeze(1))
        head_rank = (head_scores > true_head_scores).sum(dim=1) + 1

        mrr_tail += (1.0 / tail_rank.float()).sum().item()
        mrr_head += (1.0 / head_rank.float()).sum().item()
        hits10_tail += (tail_rank <= 10).float().sum().item()
        hits10_head += (head_rank <= 10).float().sum().item()

    denom = max(n, 1)
    return {
        "mrr_tail_raw": mrr_tail / denom,
        "mrr_head_raw": mrr_head / denom,
        "mrr_raw": (mrr_tail + mrr_head) / (2.0 * denom),
        "hits10_tail_raw": hits10_tail / denom,
        "hits10_head_raw": hits10_head / denom,
        "hits10_raw": (hits10_tail + hits10_head) / (2.0 * denom),
        "num_eval_triples": denom,
    }


@torch.no_grad()
def dump_rankings_for_split(
    *,
    model: KGEModel,
    triples: Sequence[Tuple[str, str, str]],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    out_path: Path,
    split_name: str,
    top_m: int,
    mode: Literal["head", "tail", "both"] = "both",
    max_queries: Optional[int] = None,
    batch_size: int = 16,
    device: torch.device,
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entity_id_to_label = {v: k for k, v in entity_to_id.items()}

    model = model.to(device)
    model.eval()

    effective_triples = list(triples[:max_queries] if max_queries is not None else triples)
    total_queries = len(effective_triples) * (2 if mode == "both" else 1)
    written = 0
    started = time.time()

    def _iter_chunks(items: Sequence[Tuple[str, str, str]], n: int):
        for i in range(0, len(items), n):
            yield i, items[i: i + n]

    with out_path.open("w", encoding="utf-8") as f_out:
        if mode in {"tail", "both"}:
            for start_idx, chunk in _iter_chunks(effective_triples, batch_size):
                hr_pairs = []
                chunk_meta = []
                for j, (h, r, t_true) in enumerate(chunk):
                    h_id = entity_to_id.get(h)
                    r_id = relation_to_id.get(r)
                    if h_id is None or r_id is None:
                        continue
                    hr_pairs.append([h_id, r_id])
                    chunk_meta.append((start_idx + j, h, r, t_true))
                if not hr_pairs:
                    continue

                hr_batch = torch.as_tensor(hr_pairs, dtype=torch.long, device=device)
                scores = model.score_t(hr_batch)
                top_scores, top_ids = torch.topk(scores, k=min(top_m, scores.shape[1]), dim=1)

                for row_idx, meta in enumerate(chunk_meta):
                    qid, h, r, t_true = meta
                    cand_ids = top_ids[row_idx].detach().cpu().tolist()
                    cand_scores = top_scores[row_idx].detach().cpu().tolist()
                    record = {
                        "query_id": f"{split_name}_tail_{qid:09d}",
                        "split": split_name,
                        "mode": "tail",
                        "head": h,
                        "relation": r,
                        "true_tail": t_true,
                        "candidates": [
                            {"entity": entity_id_to_label[eid], "score": float(score)}
                            for eid, score in zip(cand_ids, cand_scores)
                        ],
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

        if mode in {"head", "both"}:
            for start_idx, chunk in _iter_chunks(effective_triples, batch_size):
                rt_pairs = []
                chunk_meta = []
                for j, (h_true, r, t) in enumerate(chunk):
                    t_id = entity_to_id.get(t)
                    r_id = relation_to_id.get(r)
                    if t_id is None or r_id is None:
                        continue
                    rt_pairs.append([r_id, t_id])
                    chunk_meta.append((start_idx + j, h_true, r, t))
                if not rt_pairs:
                    continue

                rt_batch = torch.as_tensor(rt_pairs, dtype=torch.long, device=device)
                scores = model.score_h(rt_batch)
                top_scores, top_ids = torch.topk(scores, k=min(top_m, scores.shape[1]), dim=1)

                for row_idx, meta in enumerate(chunk_meta):
                    qid, h_true, r, t = meta
                    cand_ids = top_ids[row_idx].detach().cpu().tolist()
                    cand_scores = top_scores[row_idx].detach().cpu().tolist()
                    record = {
                        "query_id": f"{split_name}_head_{qid:09d}",
                        "split": split_name,
                        "mode": "head",
                        "tail": t,
                        "relation": r,
                        "true_head": h_true,
                        "candidates": [
                            {"entity": entity_id_to_label[eid], "score": float(score)}
                            for eid, score in zip(cand_ids, cand_scores)
                        ],
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

    return {
        "output_path": str(out_path),
        "records_written": written,
        "requested_total_queries": total_queries,
        "elapsed_seconds": round(time.time() - started, 3),
        "top_m": top_m,
        "mode": mode,
        "max_queries": max_queries,
        "batch_size": batch_size,
        "device": str(device),
    }


def build_optimizer(name: str, params, lr: float, adam_betas: Tuple[float, float], eps: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=adam_betas, eps=eps)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, eps=eps)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DDP KGC trainer with raw top-M ranking dump (PyTorch native, no DGL-KE)."
    )
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--model", type=str, default="complex", choices=sorted(SUPPORTED_MODELS))
    p.add_argument("--embedding-dim", type=int, default=200)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024, help="Per-GPU batch size")
    p.add_argument("--batch-size-eval", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--num-negs", type=int, default=16)
    p.add_argument("--gamma", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adv-temperature", type=float, default=1.0)

    p.add_argument("--optimizer", type=str, default="adam", choices=sorted(SUPPORTED_OPTIMIZERS))
    p.add_argument("--adam-betas", type=float, nargs=2, default=(0.9, 0.999), metavar=("B1", "B2"))
    p.add_argument("--optimizer-eps", type=float, default=1e-8)

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--grad-clip-norm", type=float, default=None)

    p.add_argument("--validate-every", type=int, default=5)
    p.add_argument("--max-eval-queries", type=int, default=5000)

    p.add_argument("--clear-cuda-cache", action="store_true")
    p.add_argument("--alloc-expandable-segments", action="store_true")

    p.add_argument("--dump-topm", action="store_true")
    p.add_argument("--dump-topm-size", type=int, default=1000)
    p.add_argument("--dump-splits", nargs="*", default=["test"])
    p.add_argument("--dump-mode", type=str, default="both", choices=sorted(SUPPORTED_DUMP_MODES))
    p.add_argument("--dump-max-queries", type=int, default=None)
    p.add_argument("--dump-batch-size", type=int, default=16)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    _maybe_set_allocator_env(args.alloc_expandable_segments)

    rank, local_rank, world_size, device = setup_distributed()
    set_all_seeds(args.seed + rank)
    _maybe_clear_cuda_cache(args.clear_cuda_cache)

    for split_name in args.dump_splits:
        if split_name not in SUPPORTED_SPLITS:
            raise SystemExit(f"Unsupported dump split: {split_name}")

    output_dir: Path = args.output_dir
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    barrier()

    if is_main_process():
        config = {
            "processed_dir": str(args.processed_dir),
            "output_dir": str(output_dir),
            "model": args.model,
            "embedding_dim": args.embedding_dim,
            "epochs": args.epochs,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "batch_size_eval": args.batch_size_eval,
            "learning_rate": args.learning_rate,
            "num_negs": args.num_negs,
            "gamma": args.gamma,
            "seed": args.seed,
            "adv_temperature": args.adv_temperature,
            "optimizer": args.optimizer,
            "adam_betas": list(args.adam_betas),
            "optimizer_eps": args.optimizer_eps,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "grad_clip_norm": args.grad_clip_norm,
            "validate_every": args.validate_every,
            "max_eval_queries": args.max_eval_queries,
            "dump_topm": args.dump_topm,
            "dump_topm_size": args.dump_topm_size,
            "dump_splits": args.dump_splits,
            "dump_mode": args.dump_mode,
            "dump_max_queries": args.dump_max_queries,
            "dump_batch_size": args.dump_batch_size,
            "alloc_expandable_segments": args.alloc_expandable_segments,
            "clear_cuda_cache": args.clear_cuda_cache,
            "world_size": world_size,
            "env_PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        }
        save_json(output_dir / "config.json", config)

    if is_main_process():
        print("[1/5] Loading processed dataset...")

    entity_to_id, relation_to_id, train_mapped, valid_mapped, test_mapped, stats = load_processed_dataset(
        args.processed_dir
    )

    if is_main_process():
        save_json(output_dir / "dataset_stats.json", asdict(stats))

    train_dataset = TripleDataset(train_mapped)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    if is_main_process():
        print("[2/5] Building model and starting DDP training...")

    model = KGEModel(
        model_name=args.model,
        num_entities=stats.num_entities,
        num_relations=stats.num_relations,
        embedding_dim=args.embedding_dim,
        gamma=args.gamma,
    ).to(device)

    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        ddp_model = model

    optimizer = build_optimizer(
        args.optimizer,
        ddp_model.parameters(),
        lr=args.learning_rate,
        adam_betas=tuple(args.adam_betas),
        eps=args.optimizer_eps,
    )

    train_losses: List[float] = []
    eval_history: List[Dict[str, Any]] = []
    best_mrr = -1.0
    best_state_path = output_dir / "best_model_state.pt"

    started = time.time()

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        ddp_model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            loss = compute_batch_loss(
                model=ddp_model.module if isinstance(ddp_model, DDP) else ddp_model,
                batch=batch,
                num_entities=stats.num_entities,
                num_negs=args.num_negs,
                adv_temperature=args.adv_temperature,
            )
            loss.backward()

            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.grad_clip_norm)

            optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_steps += 1

        local_loss = torch.tensor([epoch_loss_sum, epoch_steps], dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)

        mean_epoch_loss = float(local_loss[0].item() / max(local_loss[1].item(), 1.0))
        if is_main_process():
            train_losses.append(mean_epoch_loss)
            print(f"Epoch {epoch:04d} | loss={mean_epoch_loss:.6f}")

        should_eval = (epoch % args.validate_every == 0) or (epoch == args.epochs)
        if should_eval:
            barrier()
            if is_main_process():
                eval_metrics = mean_reciprocal_rank(
                    model=ddp_model.module if isinstance(ddp_model, DDP) else ddp_model,
                    mapped_triples=valid_mapped,
                    batch_size_eval=args.batch_size_eval,
                    device=device,
                    max_queries=args.max_eval_queries,
                )
                eval_metrics["epoch"] = epoch
                eval_history.append(eval_metrics)
                print(
                    f"  valid mrr_raw={eval_metrics['mrr_raw']:.6f} "
                    f"hits10_raw={eval_metrics['hits10_raw']:.6f} "
                    f"n={eval_metrics['num_eval_triples']}"
                )

                if eval_metrics["mrr_raw"] > best_mrr:
                    best_mrr = eval_metrics["mrr_raw"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": (ddp_model.module if isinstance(ddp_model, DDP) else ddp_model).state_dict(),
                            "model_name": args.model,
                            "model_kwargs": {
                                "num_entities": stats.num_entities,
                                "num_relations": stats.num_relations,
                                "embedding_dim": args.embedding_dim,
                                "gamma": args.gamma,
                            },
                            "entity_to_id": entity_to_id,
                            "relation_to_id": relation_to_id,
                        },
                        best_state_path,
                    )
            barrier()

    train_elapsed = round(time.time() - started, 3)

    if is_main_process():
        print("[3/5] Saving final artifacts...")

        final_model = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model

        torch.save(
            {
                "model_state_dict": final_model.state_dict(),
                "model_name": args.model,
                "model_kwargs": {
                    "num_entities": stats.num_entities,
                    "num_relations": stats.num_relations,
                    "embedding_dim": args.embedding_dim,
                    "gamma": args.gamma,
                },
                "entity_to_id": entity_to_id,
                "relation_to_id": relation_to_id,
                "dataset_stats": asdict(stats),
            },
            output_dir / "base_model_checkpoint.pt",
        )

        metrics = {
            "training_losses": train_losses,
            "validation_history": eval_history,
            "train_elapsed_seconds": train_elapsed,
            "best_valid_mrr_raw": best_mrr,
        }
        save_json(output_dir / "metrics.json", metrics)

    barrier()

    if is_main_process():
        print("[4/5] Optional ranking dump...")
        dump_reports: Dict[str, Any] = {}
        if args.dump_topm:
            split_to_triples = {
                "train": read_triples(args.processed_dir / "train.tsv"),
                "valid": read_triples(args.processed_dir / "valid.tsv"),
                "test": read_triples(args.processed_dir / "test.tsv"),
            }
            ranking_dir = output_dir / "ranking_dumps"
            ranking_dir.mkdir(parents=True, exist_ok=True)

            final_model = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
            final_model.eval()

            for split_name in args.dump_splits:
                dump_reports[split_name] = dump_rankings_for_split(
                    model=final_model,
                    triples=split_to_triples[split_name],
                    entity_to_id=entity_to_id,
                    relation_to_id=relation_to_id,
                    out_path=ranking_dir / f"base_topm_{split_name}.jsonl",
                    split_name=split_name,
                    top_m=args.dump_topm_size,
                    mode=args.dump_mode,
                    max_queries=args.dump_max_queries,
                    batch_size=args.dump_batch_size,
                    device=device,
                )
            save_json(output_dir / "ranking_dump_report.json", dump_reports)
        else:
            dump_reports = {}
            print("Skipping ranking dump.")

        print("[5/5] Done.")
        summary = {
            "status": "ok",
            "processed_dir": str(args.processed_dir),
            "output_dir": str(output_dir),
            "model": args.model,
            "optimizer": args.optimizer,
            "world_size": world_size,
            "dataset_stats": asdict(stats),
            "train_elapsed_seconds": train_elapsed,
            "best_valid_mrr_raw": best_mrr,
            "ranking_dump_report": dump_reports,
        }
        save_json(output_dir / "run_summary.json", summary)

        print("\nTraining complete.")
        print(f"Model               : {args.model}")
        print(f"World size          : {world_size}")
        print(f"Per-GPU batch size  : {args.batch_size}")
        print(f"Global batch size   : {args.batch_size * world_size}")
        print(f"Output dir          : {output_dir}")
        print(f"Train elapsed (s)   : {train_elapsed}")
        print(f"Checkpoint          : {output_dir / 'base_model_checkpoint.pt'}")
        if args.dump_topm:
            print(f"Ranking dumps       : {output_dir / 'ranking_dumps'}")

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
