import os

import datasets
import numpy as np

from recipe.gkd.megatron.dataset_teacher_router import build_file_to_teacher_route, load_dataset_teacher_router
from verl.utils.dataset.rl_dataset import RLHFDataset


class RoutedRLHFDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        router_path = config.get("dataset_teacher_router", None)
        if not router_path:
            raise ValueError("RoutedRLHFDataset requires data.dataset_teacher_router to be set")
        self.dataset_teacher_router = load_dataset_teacher_router(router_path)
        self.file_to_teacher_route = build_file_to_teacher_route(self.dataset_teacher_router)
        super().__init__(data_files, tokenizer, config, processor, max_samples)

    def _read_files_and_tokenize(self):
        dataframes = []
        for original_file, local_file in zip(self.original_data_files, self.data_files, strict=True):
            route_key = os.path.normpath(str(original_file))
            if route_key not in self.file_to_teacher_route:
                raise ValueError(f"File '{original_file}' is missing from dataset teacher router config")
            route_info = self.file_to_teacher_route[route_key]

            if local_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=local_file)["train"]
            elif local_file.endswith(".json"):
                dataframe = datasets.load_dataset("json", data_files=local_file)["train"]
            else:
                raise ValueError(f"Unsupported file format: {local_file}")

            n_rows = len(dataframe)
            dataframe = dataframe.add_column("teacher_name", [route_info["teacher_name"]] * n_rows)
            dataframe = dataframe.add_column("distill_weight", [route_info["distill_weight"]] * n_rows)
            dataframe = dataframe.add_column("dataset_route_name", [route_info["dataset_name"]] * n_rows)
            dataframes.append(dataframe)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
