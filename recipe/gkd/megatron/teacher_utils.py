# Copyright 2025 Individual Contributor: furunding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for teacher model knowledge distillation.

Functions:
    get_teacher_knowledge: Retrieve teacher model's top-k predictions and log probabilities.
"""

import time
from collections import defaultdict
from math import ceil
from types import SimpleNamespace

import numpy as np
import torch

from verl import DataProto

teacher_topk_logps_padded, teacher_topk_indices_padded = None, None


def _pad_teacher_outputs(batch, attention_mask, teacher_topk_logps, teacher_topk_indices):
    topk = teacher_topk_logps[0].size(-1)
    logp_dtype = teacher_topk_logps[0].dtype
    idx_dtype = teacher_topk_indices[0].dtype
    teacher_knowledge_shape = list(batch.batch["input_ids"].shape) + [topk]

    teacher_topk_logps_padded = torch.zeros(*teacher_knowledge_shape, dtype=logp_dtype)
    teacher_topk_indices_padded = torch.zeros(*teacher_knowledge_shape, dtype=idx_dtype)

    batch_size = attention_mask.size(0)
    for i in range(batch_size):
        teacher_topk_logps_padded[i][attention_mask[i]] = teacher_topk_logps[i]
        teacher_topk_indices_padded[i][attention_mask[i]] = teacher_topk_indices[i]

    return teacher_topk_logps_padded, teacher_topk_indices_padded


def _normalize_teacher_infos(teacher_or_infos, n_server_workers):
    if isinstance(teacher_or_infos, list):
        return teacher_or_infos
    return [
        {
            "name": "teacher_0",
            "weight": 1.0,
            "n_server_workers": n_server_workers,
            "request_mode": "parallel",
            "client": teacher_or_infos,
        }
    ]

    
def _submit_teacher_requests(input_ids, teacher_client, n_server_workers, cb):
    batch_size = len(input_ids)
    effective_workers = min(max(1, int(n_server_workers)), batch_size)
    chunk_size = ceil(batch_size / effective_workers)
    curr_futures = []
    for i in range(0, batch_size, chunk_size):
        fut = teacher_client.submit(input_ids[i : i + chunk_size])
        fut.add_done_callback(cb)
        curr_futures.append(fut)
    return curr_futures

def get_teacher_knowledge(batch: DataProto, teacher_or_infos, n_server_workers=1, is_async=False):
    """
    Retrieve teacher model's top-k predictions and log probabilities for knowledge distillation.

    Args:
        batch (DataProto): Input batch containing input_ids and attention_mask
        teacher_or_infos: Single teacher client or a list of teacher info dictionaries
        n_server_workers (int): Legacy fallback number of parallel workers for single teacher inference
        is_async (bool): Whether to use asynchronous processing

    Returns:
        If is_async=True: SimpleNamespace with get() method to process futures
        If is_async=False: Processed DataProto containing teacher knowledge

    Raises:
        RuntimeError: If teacher model request fails
    """

    input_ids = []
    attention_mask = batch.batch["attention_mask"].to(torch.bool)
    teacher_infos = _normalize_teacher_infos(teacher_or_infos, n_server_workers)
    request_mode = teacher_infos[0].get("request_mode", "parallel")

    for ids, mask in zip(batch.batch["input_ids"], attention_mask, strict=False):
        input_ids.append(ids[mask].tolist())

    batch_size = len(input_ids)
    tik1 = time.time()
    tok1 = tik1

    def cb(future):
        nonlocal tok1
        tok1 = max(tok1, time.time())

    def submit_teacher_requests(teacher_info):
        teacher_client = teacher_info["client"]
        curr_n_server_workers = teacher_info["n_server_workers"]
        return _submit_teacher_requests(input_ids, teacher_client, curr_n_server_workers, cb)


    teacher_requests = []
    if request_mode == "parallel":
        for teacher_info in teacher_infos:
            teacher_requests.append((teacher_info, submit_teacher_requests(teacher_info)))

    def handle_futures():
        per_teacher_logps = []
        per_teacher_indices = []
        teacher_weights = []
        teacher_names = []

        requests = teacher_requests
        if request_mode != "parallel":
            requests = [(teacher_info, None) for teacher_info in teacher_infos]

        for teacher_info, futures in requests:
            if futures is None:
                futures = submit_teacher_requests(teacher_info)

            all_teacher_topk_logps = []
            all_teacher_topk_indices = []
            for future in futures:
                try:
                    _, teacher_topk_logps, teacher_topk_indices = future.result()
                except Exception as e:
                    teacher_name = teacher_info["name"]
                    raise RuntimeError(f"Teacher request failed for {teacher_name}: {e}") from e

                all_teacher_topk_logps.extend(teacher_topk_logps)
                all_teacher_topk_indices.extend(teacher_topk_indices)

            padded_logps, padded_indices = _pad_teacher_outputs(
                batch, attention_mask, all_teacher_topk_logps, all_teacher_topk_indices
            )
            per_teacher_logps.append(padded_logps)
            per_teacher_indices.append(padded_indices)
            teacher_weights.append(teacher_info["weight"])
            teacher_names.append(teacher_info["name"])

        tik2 = time.time()

        output_batch = DataProto.from_single_dict(
            data={"real_seq_lens": attention_mask.sum(dim=-1).to(torch.int32)},
        )

        if len(per_teacher_logps) == 1:
            output_batch.non_tensor_batch.update(
                {
                    "teacher_topk_logps": per_teacher_logps[0].numpy(),
                    "teacher_topk_indices": per_teacher_indices[0].numpy(),
                }
            )
        else:
            stacked_logps = torch.stack(per_teacher_logps, dim=2)
            stacked_indices = torch.stack(per_teacher_indices, dim=2)
            stacked_weights = (
                torch.tensor(teacher_weights, dtype=torch.float32)
                .view(1, 1, -1)
                .expand(attention_mask.size(0), attention_mask.size(1), -1)
            )
            output_batch.non_tensor_batch.update(
                {
                    "multi_teacher_topk_logps": stacked_logps.numpy(),
                    "multi_teacher_topk_indices": stacked_indices.numpy(),
                    "multi_teacher_weights": stacked_weights.numpy(),
                }
            )
            output_batch.meta_info["multi_teacher_names"] = teacher_names

        tok2 = time.time()

        output_batch.meta_info["timing"] = {"get_teacher_knowledge": (tok1 - tik1) + (tok2 - tik2)}

        return output_batch

    if is_async:
        return SimpleNamespace(get=handle_futures)
    else:
        return handle_futures()


def get_teacher_knowledge_by_teacher_name(
    batch: DataProto, teacher_clients_by_name, teacher_workers_by_name, is_async=False
):
    attention_mask = batch.batch["attention_mask"].to(torch.bool)
    teacher_names = batch.non_tensor_batch["teacher_name"]
    input_ids = []
    grouped_indices = defaultdict(list)
    for sample_idx, (ids, mask, teacher_name) in enumerate(
        zip(batch.batch["input_ids"], attention_mask, teacher_names, strict=False)
    ):
        grouped_indices[str(teacher_name)].append(sample_idx)
        input_ids.append(ids[mask].tolist())

    tik1 = time.time()
    tok1 = tik1

    def cb(future):
        nonlocal tok1
        tok1 = max(tok1, time.time())

    teacher_requests = {}
    for teacher_name, sample_indices in grouped_indices.items():
        if teacher_name not in teacher_clients_by_name:
            raise KeyError(f"Unknown teacher_name '{teacher_name}' found in batch")
        grouped_input_ids = [input_ids[idx] for idx in sample_indices]
        teacher_requests[teacher_name] = {
            "indices": sample_indices,
            "futures": _submit_teacher_requests(
                grouped_input_ids,
                teacher_clients_by_name[teacher_name],
                teacher_workers_by_name[teacher_name],
                cb,
            ),
        }

    def handle_futures():
        full_teacher_topk_logps = None
        full_teacher_topk_indices = None

        for teacher_name, request_info in teacher_requests.items():
            sample_indices = request_info["indices"]
            all_teacher_topk_logps = []
            all_teacher_topk_indices = []
            for future in request_info["futures"]:
                try:
                    _, teacher_topk_logps, teacher_topk_indices = future.result()
                except Exception as e:
                    raise RuntimeError(f"Teacher request failed for {teacher_name}: {e}") from e
                all_teacher_topk_logps.extend(teacher_topk_logps)
                all_teacher_topk_indices.extend(teacher_topk_indices)

            if len(all_teacher_topk_logps) != len(sample_indices):
                raise RuntimeError(
                    f"Teacher '{teacher_name}' returned {len(all_teacher_topk_logps)} samples, "
                    f"expected {len(sample_indices)}"
                )

            if full_teacher_topk_logps is None:
                topk = all_teacher_topk_logps[0].size(-1)
                teacher_knowledge_shape = list(batch.batch["input_ids"].shape) + [topk]
                full_teacher_topk_logps = torch.zeros(
                    *teacher_knowledge_shape, dtype=all_teacher_topk_logps[0].dtype
                )
                full_teacher_topk_indices = torch.zeros(
                    *teacher_knowledge_shape, dtype=all_teacher_topk_indices[0].dtype
                )

            for local_idx, sample_idx in enumerate(sample_indices):
                full_teacher_topk_logps[sample_idx][attention_mask[sample_idx]] = all_teacher_topk_logps[local_idx]
                full_teacher_topk_indices[sample_idx][attention_mask[sample_idx]] = all_teacher_topk_indices[local_idx]

        tik2 = time.time()
        output_batch = DataProto.from_single_dict(
            data={"real_seq_lens": attention_mask.sum(dim=-1).to(torch.int32)},
        )
        output_batch.non_tensor_batch.update(
            {
                "teacher_topk_logps": full_teacher_topk_logps.numpy(),
                "teacher_topk_indices": full_teacher_topk_indices.numpy(),
                #"teacher_name": np.asarray(teacher_names, dtype=object),
                #"distill_weight": np.asarray(batch.non_tensor_batch["distill_weight"], dtype=np.float32),
            }
        )
        tok2 = time.time()
        output_batch.meta_info["timing"] = {"get_teacher_knowledge": (tok1 - tik1) + (tok2 - tik2)}
        return output_batch

    if is_async:
        return SimpleNamespace(get=handle_futures)
    return handle_futures()


if __name__ == "__main__":
    batch = DataProto.load_from_disk("gen_batch_output")
    from teacher import TeacherClient

    teacher_client = TeacherClient(server_ip="10.215.192.141", server_port=15555)
    output_batch = get_teacher_knowledge(batch, teacher_client, 2)
    output_batch_chunks = output_batch.chunk(2)

    for data in output_batch_chunks:
        topk = data.meta_info["topk"]
        seq_lens = data.batch["seq_lens"]
        teacher_topk_logps = data.batch["teacher_topk_logps"].view(-1, topk)
        teacher_topk_indices = data.batch["teacher_topk_indices"].view(-1, topk)

        attention_mask = data.batch["attention_mask"]
        batch_size, sequence_length = attention_mask.size(0), attention_mask.size(1)
        teacher_topk_logps_padded = torch.zeros(batch_size, sequence_length, topk, dtype=teacher_topk_logps.dtype)
        teacher_topk_indices_padded = torch.zeros(batch_size, sequence_length, topk, dtype=teacher_topk_indices.dtype)

        teacher_topk_logps_padded[attention_mask] = teacher_topk_logps[: seq_lens.sum()]
        teacher_topk_indices_padded[attention_mask] = teacher_topk_indices[: seq_lens.sum()]

        data.batch["teacher_topk_logps"] = teacher_topk_logps_padded
        data.batch["teacher_topk_indices"] = teacher_topk_indices_padded

        assert (data.batch["teacher_topk_logps"] == data.batch["teacher_topk_logps_padded"]).all()
        assert (data.batch["teacher_topk_indices"] == data.batch["teacher_topk_indices_padded"]).all()
