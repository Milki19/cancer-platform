# src/scan_raw.py

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

try:
    import yaml  # pyyaml
except ImportError as e:
    raise SystemExit("Missing dependency: pyyaml. Install with: pip install pyyaml") from e

# Optional (only for tabular preview)
try:
    import pandas as pd
except Exception:
    pd = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetScanResult:
    dataset_id: str
    dtype: str
    raw_path: str
    ok: bool
    message: str

    # image-ish
    class_dirs: Optional[List[str]] = None
    class_counts: Optional[Dict[str, int]] = None

    # general file stats
    total_files: int = 0
    total_bytes: int = 0
    ext_counts: Optional[Dict[str, int]] = None

    # tabular preview
    tabular_shape: Optional[Tuple[int, int]] = None
    tabular_columns: Optional[List[str]] = None


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


def list_files_recursive(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def scan_image_dataset(dataset_id: str, ds_cfg: Dict[str, Any], raw_root: Path) -> DatasetScanResult:
    raw_dir = ds_cfg.get("raw_dir", dataset_id)
    ds_path = (raw_root / raw_dir).resolve()

    if not ds_path.exists():
        return DatasetScanResult(
            dataset_id=dataset_id,
            dtype=ds_cfg.get("type", "image_classification"),
            raw_path=str(ds_path),
            ok=False,
            message="Raw path does not exist. Check configs/local.yaml data_root and datasets.yaml raw_dir.",
        )

    # Class dirs: if explicit classes list exists, use it; otherwise infer from first-level dirs
    classes = ds_cfg.get("classes")
    if classes:
        class_dirs = [ds_path / c for c in classes]
    else:
        class_dirs = [p for p in ds_path.iterdir() if p.is_dir()]

    class_counts: Dict[str, int] = {}
    ext_counter = Counter()
    total_bytes = 0
    total_files = 0

    for cdir in sorted(class_dirs, key=lambda x: x.name.lower()):
        if not cdir.exists():
            class_counts[cdir.name] = 0
            continue

        # Count images recursively (some Kaggle datasets nest folders)
        files = [p for p in cdir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        class_counts[cdir.name] = len(files)
        total_files += len(files)

        for f in files:
            ext_counter[f.suffix.lower()] += 1
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass

    # if dataset is not organized by class folders (rare), fallback count whole ds_path
    if total_files == 0:
        files = [p for p in ds_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        total_files = len(files)
        for f in files:
            ext_counter[f.suffix.lower()] += 1
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass

    return DatasetScanResult(
        dataset_id=dataset_id,
        dtype=ds_cfg.get("type", "image_classification"),
        raw_path=str(ds_path),
        ok=True,
        message="OK",
        class_dirs=[p.name for p in class_dirs],
        class_counts=class_counts,
        total_files=total_files,
        total_bytes=total_bytes,
        ext_counts=dict(ext_counter),
    )


def scan_tabular_dataset(dataset_id: str, ds_cfg: Dict[str, Any], raw_root: Path) -> DatasetScanResult:
    raw_dir = ds_cfg.get("raw_dir", dataset_id)
    ds_path = (raw_root / raw_dir).resolve()
    if not ds_path.exists():
        return DatasetScanResult(
            dataset_id=dataset_id,
            dtype=ds_cfg.get("type", "tabular"),
            raw_path=str(ds_path),
            ok=False,
            message="Raw path does not exist. Check configs/local.yaml and datasets.yaml.",
        )

    fname = ds_cfg.get("file")
    if not fname:
        # try to infer a single csv/xlsx in folder
        candidates = list(ds_path.glob("*.csv")) + list(ds_path.glob("*.xlsx")) + list(ds_path.glob("*.xls"))
        if len(candidates) == 1:
            fpath = candidates[0]
        else:
            return DatasetScanResult(
                dataset_id=dataset_id,
                dtype=ds_cfg.get("type", "tabular"),
                raw_path=str(ds_path),
                ok=False,
                message="Tabular file not specified and could not infer a single CSV/XLSX.",
            )
    else:
        fpath = (ds_path / fname).resolve()
        if not fpath.exists():
            return DatasetScanResult(
                dataset_id=dataset_id,
                dtype=ds_cfg.get("type", "tabular"),
                raw_path=str(ds_path),
                ok=False,
                message=f"Tabular file missing: {fpath.name}",
            )

    ext_counter = Counter()
    total_bytes = 0
    total_files = 0

    # Count all files in tabular folder (small)
    files = list_files_recursive(ds_path)
    total_files = len(files)
    for f in files:
        ext_counter[f.suffix.lower()] += 1
        try:
            total_bytes += f.stat().st_size
        except OSError:
            pass

    shape = None
    cols = None
    if pd is not None:
        try:
            if fpath.suffix.lower() == ".csv":
                df = pd.read_csv(fpath)
            else:
                df = pd.read_excel(fpath)  # openpyxl is usually available in Colab
            shape = df.shape
            cols = list(df.columns.astype(str))
        except Exception as e:
            # Don't fail the scan just because preview failed
            return DatasetScanResult(
                dataset_id=dataset_id,
                dtype=ds_cfg.get("type", "tabular"),
                raw_path=str(ds_path),
                ok=True,
                message=f"OK (tabular preview failed: {e})",
                total_files=total_files,
                total_bytes=total_bytes,
                ext_counts=dict(ext_counter),
            )

    return DatasetScanResult(
        dataset_id=dataset_id,
        dtype=ds_cfg.get("type", "tabular"),
        raw_path=str(ds_path),
        ok=True,
        message="OK",
        total_files=total_files,
        total_bytes=total_bytes,
        ext_counts=dict(ext_counter),
        tabular_shape=shape,
        tabular_columns=cols,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]  # project root (src/..)
    local_cfg = load_yaml(root / "configs" / "local.yaml")
    datasets_cfg = load_yaml(root / "configs" / "datasets.yaml")

    data_root = Path(local_cfg["data_root"]).resolve()
    raw_root = (data_root / "raw").resolve()

    datasets = datasets_cfg.get("datasets", {})
    if not datasets:
        print("No datasets defined in configs/datasets.yaml under key: datasets")
        return 1

    results: List[DatasetScanResult] = []

    for dataset_id, ds_cfg in datasets.items():
        dtype = (ds_cfg.get("type") or "").lower()
        if "tabular" in dtype:
            res = scan_tabular_dataset(dataset_id, ds_cfg, raw_root)
        else:
            res = scan_image_dataset(dataset_id, ds_cfg, raw_root)
        results.append(res)

    # Pretty print
    print(f"Data root: {data_root}")
    print(f"Raw root : {raw_root}\n")

    for r in results:
        status = "✅" if r.ok else "❌"
        print(f"{status} {r.dataset_id} [{r.dtype}]")
        print(f"   path: {r.raw_path}")
        print(f"   msg : {r.message}")
        print(f"   files: {r.total_files} | size: {human_bytes(r.total_bytes)}")
        if r.ext_counts:
            top_ext = sorted(r.ext_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   top ext: {top_ext}")
        if r.class_counts:
            # show top 8 classes by count
            top_classes = sorted(r.class_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            print(f"   classes: {top_classes}")
        if r.tabular_shape:
            print(f"   tabular: shape={r.tabular_shape}")
            print(f"   columns: {r.tabular_columns[:12]}{' ...' if len(r.tabular_columns) > 12 else ''}")
        print()

    # Save summary artifact (not in git by default if you put it in artifacts/)
    out_dir = root / "artifacts" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scan_raw_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
