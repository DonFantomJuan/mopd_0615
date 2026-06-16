import os

from omegaconf import ListConfig, OmegaConf


def _normalize_file_key(path):
    return os.path.normpath(str(path))


def load_dataset_teacher_router(router_path):
    if not router_path:
        raise ValueError("dataset_teacher_router path must be provided")

    router_cfg = OmegaConf.load(router_path)
    datasets_cfg = OmegaConf.select(router_cfg, "datasets")
    if not datasets_cfg:
        raise ValueError(f"No datasets found in dataset teacher router config: {router_path}")

    seen_files = set()
    for dataset_cfg in datasets_cfg:
        dataset_name = dataset_cfg.get("name", "unnamed_dataset")
        train_files = dataset_cfg.get("train_files")
        teacher_cfg = dataset_cfg.get("teacher")
        if not train_files:
            raise ValueError(f"Dataset '{dataset_name}' must define train_files")
        if teacher_cfg is None:
            raise ValueError(f"Dataset '{dataset_name}' must define teacher")
        for key in ("name", "server_ip", "server_port"):
            if teacher_cfg.get(key) is None:
                raise ValueError(f"Dataset '{dataset_name}' teacher is missing required key '{key}'")
        for train_file in train_files:
            normalized = _normalize_file_key(train_file)
            if normalized in seen_files:
                raise ValueError(f"Duplicate train file detected in dataset teacher router: {train_file}")
            seen_files.add(normalized)

    return router_cfg


def collect_train_files(router_cfg):
    train_files = []
    for dataset_cfg in OmegaConf.select(router_cfg, "datasets"):
        dataset_files = dataset_cfg.get("train_files")
        if isinstance(dataset_files, (list, ListConfig)):
            train_files.extend(list(dataset_files))
        else:
            train_files.append(dataset_files)
    return train_files


def build_file_to_teacher_route(router_cfg):
    route_mapping = {}
    for dataset_cfg in OmegaConf.select(router_cfg, "datasets"):
        dataset_name = dataset_cfg.get("name", "unnamed_dataset")
        teacher_cfg = dataset_cfg.teacher
        for train_file in dataset_cfg.train_files:
            route_mapping[_normalize_file_key(train_file)] = {
                "dataset_name": dataset_name,
                "teacher_name": teacher_cfg.name,
                "server_ip": teacher_cfg.server_ip,
                "server_port": teacher_cfg.server_port,
                "n_server_workers": int(teacher_cfg.get("n_server_workers", 1)),
                "distill_weight": float(teacher_cfg.get("distill_weight", 1.0)),
            }
    return route_mapping


def build_unique_teachers(router_cfg):
    unique_teachers = {}
    for dataset_cfg in OmegaConf.select(router_cfg, "datasets"):
        teacher_cfg = dataset_cfg.teacher
        teacher_name = teacher_cfg.name
        teacher_info = {
            "name": teacher_name,
            "server_ip": teacher_cfg.server_ip,
            "server_port": teacher_cfg.server_port,
            "n_server_workers": int(teacher_cfg.get("n_server_workers", 1)),
        }
        if teacher_name in unique_teachers:
            prev = unique_teachers[teacher_name]
            for key in ("server_ip", "server_port", "n_server_workers"):
                if prev[key] != teacher_info[key]:
                    raise ValueError(
                        f"Teacher '{teacher_name}' is defined with inconsistent '{key}' values "
                        f"across dataset router config"
                    )
        else:
            unique_teachers[teacher_name] = teacher_info
    return unique_teachers