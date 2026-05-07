from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse

try:
    import yaml
except ImportError as error:
    raise SystemExit("Missing dependency: PyYAML. Install with: pip install PyYAML") from error

try:
    import pandas as pd
except Exception:
    pd = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetScanResult:
    dataset_id: str
    display_name: str
    dtype: str
    raw_path: str
    ok: bool
    message: str
    total_files: int = 0
    total_bytes: int = 0
    ext_counts: Optional[Dict[str, int]] = None
    class_dirs: Optional[List[str]] = None
    class_counts: Optional[Dict[str, int]] = None
    tabular_shape: Optional[Tuple[int, int]] = None
    tabular_columns: Optional[List[str]] = None


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def human_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)

    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0

    return f"{value} B"


def resolve_path(root_dir: Path, path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return (root_dir / path).resolve()


def list_files_recursive(root: Path) -> List[Path]:
    if not root.exists():
        return []

    return [path for path in root.rglob("*") if path.is_file()]


def count_files(files: List[Path]) -> Tuple[int, int, Dict[str, int]]:
    ext_counter = Counter()
    total_bytes = 0

    for file_path in files:
        ext_counter[file_path.suffix.lower()] += 1

        try:
            total_bytes += file_path.stat().st_size
        except OSError:
            pass

    return len(files), total_bytes, dict(ext_counter)


def find_image_files(root: Path) -> List[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def infer_image_class_counts(root: Path) -> Tuple[List[str], Dict[str, int]]:
    first_level_dirs = [path for path in root.iterdir() if path.is_dir()]
    class_counts: Dict[str, int] = {}

    for directory in sorted(first_level_dirs, key=lambda item: item.name.lower()):
        image_files = find_image_files(directory)
        if image_files:
            class_counts[directory.name] = len(image_files)

    if len(class_counts) >= 2:
        return list(class_counts.keys()), class_counts

    leaf_counts: Dict[str, int] = {}

    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue

        direct_images = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]

        if direct_images:
            leaf_counts[directory.name] = len(direct_images)

    if leaf_counts:
        return list(leaf_counts.keys()), leaf_counts

    return [], {}


def scan_image_dataset(dataset_id: str, dataset_config: Dict[str, Any], root_dir: Path) -> DatasetScanResult:
    raw_dir_value = dataset_config.get("raw_dir")

    if not raw_dir_value:
        return DatasetScanResult(
            dataset_id=dataset_id,
            display_name=dataset_config.get("display_name", dataset_id),
            dtype=dataset_config.get("type", "image"),
            raw_path="",
            ok=False,
            message="Missing raw_dir in configs/datasets.yaml.",
        )

    dataset_path = resolve_path(root_dir, raw_dir_value)

    if not dataset_path.exists():
        return DatasetScanResult(
            dataset_id=dataset_id,
            display_name=dataset_config.get("display_name", dataset_id),
            dtype=dataset_config.get("type", "image"),
            raw_path=str(dataset_path),
            ok=False,
            message="Raw path does not exist. Run scripts/download_kaggle_data.py first.",
        )

    image_files = find_image_files(dataset_path)
    total_files, total_bytes, ext_counts = count_files(image_files)
    class_dirs, class_counts = infer_image_class_counts(dataset_path)

    message = "OK" if total_files > 0 else "No image files found."

    return DatasetScanResult(
        dataset_id=dataset_id,
        display_name=dataset_config.get("display_name", dataset_id),
        dtype=dataset_config.get("type", "image"),
        raw_path=str(dataset_path),
        ok=total_files > 0,
        message=message,
        total_files=total_files,
        total_bytes=total_bytes,
        ext_counts=ext_counts,
        class_dirs=class_dirs,
        class_counts=class_counts,
    )


def find_tabular_file(dataset_path: Path, dataset_config: Dict[str, Any]) -> Optional[Path]:
    file_name = dataset_config.get("file")

    if file_name:
        file_path = (dataset_path / file_name).resolve()
        return file_path if file_path.exists() else None

    candidates = (
        list(dataset_path.rglob("*.csv"))
        + list(dataset_path.rglob("*.xlsx"))
        + list(dataset_path.rglob("*.xls"))
    )

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda path: path.stat().st_size, reverse=True)
    return candidates[0]


def scan_tabular_dataset(dataset_id: str, dataset_config: Dict[str, Any], root_dir: Path) -> DatasetScanResult:
    raw_dir_value = dataset_config.get("raw_dir")

    if not raw_dir_value:
        return DatasetScanResult(
            dataset_id=dataset_id,
            display_name=dataset_config.get("display_name", dataset_id),
            dtype=dataset_config.get("type", "tabular"),
            raw_path="",
            ok=False,
            message="Missing raw_dir in configs/datasets.yaml.",
        )

    dataset_path = resolve_path(root_dir, raw_dir_value)

    if not dataset_path.exists():
        return DatasetScanResult(
            dataset_id=dataset_id,
            display_name=dataset_config.get("display_name", dataset_id),
            dtype=dataset_config.get("type", "tabular"),
            raw_path=str(dataset_path),
            ok=False,
            message="Raw path does not exist. Run scripts/download_kaggle_data.py first.",
        )

    all_files = list_files_recursive(dataset_path)
    total_files, total_bytes, ext_counts = count_files(all_files)
    tabular_file = find_tabular_file(dataset_path, dataset_config)

    if tabular_file is None:
        return DatasetScanResult(
            dataset_id=dataset_id,
            display_name=dataset_config.get("display_name", dataset_id),
            dtype=dataset_config.get("type", "tabular"),
            raw_path=str(dataset_path),
            ok=False,
            message="No CSV/XLS/XLSX file found.",
            total_files=total_files,
            total_bytes=total_bytes,
            ext_counts=ext_counts,
        )

    if pd is None:
        return DatasetScanResult(
            dataset_id=dataset_id,
            display_name=dataset_config.get("display_name", dataset_id),
            dtype=dataset_config.get("type", "tabular"),
            raw_path=str(dataset_path),
            ok=True,
            message=f"OK. Found tabular file: {tabular_file.name}. Pandas not available for preview.",
            total_files=total_files,
            total_bytes=total_bytes,
            ext_counts=ext_counts,
        )

    try:
        if tabular_file.suffix.lower() == ".csv":
            dataframe = pd.read_csv(tabular_file)
        else:
            dataframe = pd.read_excel(tabular_file)

        return DatasetScanResult(
            dataset_id=dataset_id,
            display_name=dataset_config.get("display_name", dataset_id),
            dtype=dataset_config.get("type", "tabular"),
            raw_path=str(dataset_path),
            ok=True,
            message=f"OK. Found tabular file: {tabular_file.name}",
            total_files=total_files,
            total_bytes=total_bytes,
            ext_counts=ext_counts,
            tabular_shape=dataframe.shape,
            tabular_columns=list(dataframe.columns.astype(str)),
        )

    except Exception as error:
        return DatasetScanResult(
            dataset_id=dataset_id,
            display_name=dataset_config.get("display_name", dataset_id),
            dtype=dataset_config.get("type", "tabular"),
            raw_path=str(dataset_path),
            ok=True,
            message=f"OK. Found tabular file: {tabular_file.name}. Preview failed: {error}",
            total_files=total_files,
            total_bytes=total_bytes,
            ext_counts=ext_counts,
        )


def scan_dataset(dataset_id: str, dataset_config: Dict[str, Any], root_dir: Path) -> DatasetScanResult:
    dataset_type = (dataset_config.get("type") or "").lower()

    if dataset_type == "tabular":
        return scan_tabular_dataset(dataset_id, dataset_config, root_dir)

    return scan_image_dataset(dataset_id, dataset_config, root_dir)


def print_result(result: DatasetScanResult) -> None:
    status = "OK" if result.ok else "MISSING"

    print(f"{status} | {result.dataset_id} | {result.display_name} | {result.dtype}")
    print(f"Path: {result.raw_path}")
    print(f"Message: {result.message}")
    print(f"Files: {result.total_files} | Size: {human_bytes(result.total_bytes)}")

    if result.ext_counts:
        top_extensions = sorted(result.ext_counts.items(), key=lambda item: item[1], reverse=True)[:8]
        print(f"Extensions: {top_extensions}")

    if result.class_counts:
        top_classes = sorted(result.class_counts.items(), key=lambda item: item[1], reverse=True)[:12]
        print(f"Classes: {top_classes}")

    if result.tabular_shape:
        print(f"Tabular shape: {result.tabular_shape}")

    if result.tabular_columns:
        preview_columns = result.tabular_columns[:12]
        suffix = " ..." if len(result.tabular_columns) > 12 else ""
        print(f"Columns: {preview_columns}{suffix}")

    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    config = load_yaml(root_dir / "configs" / "datasets.yaml")
    datasets = config.get("datasets", {})

    if not datasets:
        print("No datasets defined in configs/datasets.yaml under key: datasets")
        return 1

    if args.dataset:
        if args.dataset not in datasets:
            print(f"Unknown dataset: {args.dataset}")
            print("Available datasets:")
            for dataset_id in datasets:
                print(f"- {dataset_id}")
            return 1

        datasets_to_scan = {args.dataset: datasets[args.dataset]}
    else:
        datasets_to_scan = datasets

    results = [
        scan_dataset(dataset_id, dataset_config, root_dir)
        for dataset_id, dataset_config in datasets_to_scan.items()
    ]

    print(f"Project root: {root_dir}")
    print()

    for result in results:
        print_result(result)

    output_dir = root_dir / "artifacts" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = "scan_raw_summary.json"
    if args.dataset:
        output_name = f"scan_raw_summary_{args.dataset}.json"

    output_path = output_dir / output_name

    with output_path.open("w", encoding="utf-8") as file:
        json.dump([asdict(result) for result in results], file, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")

    missing = [result.dataset_id for result in results if not result.ok]

    if missing:
        print(f"Datasets with issues: {missing}")
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())