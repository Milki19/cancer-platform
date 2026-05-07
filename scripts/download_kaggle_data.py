from pathlib import Path
import argparse
import shutil
import subprocess
import sys

import yaml


def load_config(root_dir: Path) -> dict:
    config_path = root_dir / "configs" / "datasets.yaml"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(root_dir: Path, path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return root_dir / path


def find_kaggle_executable() -> str:
    kaggle_path = shutil.which("kaggle")

    if kaggle_path:
        return kaggle_path

    possible_windows_path = (
        Path.home()
        / "AppData"
        / "Roaming"
        / "Python"
        / f"Python{sys.version_info.major}{sys.version_info.minor}"
        / "Scripts"
        / "kaggle.exe"
    )

    if possible_windows_path.exists():
        return str(possible_windows_path)

    print("Kaggle CLI was not found.")
    print("Install it with:")
    print("  pip install kaggle")
    print()
    print("On Windows, make sure this folder is added to PATH:")
    print(
        Path.home()
        / "AppData"
        / "Roaming"
        / "Python"
        / f"Python{sys.version_info.major}{sys.version_info.minor}"
        / "Scripts"
    )
    sys.exit(1)


def run_command(command: list[str]) -> None:
    result = subprocess.run(command)

    if result.returncode != 0:
        sys.exit(result.returncode)


def download_dataset(
    dataset_name: str,
    dataset_config: dict,
    root_dir: Path,
    kaggle_executable: str,
    force: bool,
) -> None:
    kaggle_slug = dataset_config.get("kaggle_slug")
    raw_dir_value = dataset_config.get("raw_dir")

    if not kaggle_slug:
        print(f"Missing kaggle_slug for dataset: {dataset_name}")
        sys.exit(1)

    if not raw_dir_value:
        print(f"Missing raw_dir for dataset: {dataset_name}")
        sys.exit(1)

    target_dir = resolve_path(root_dir, raw_dir_value)
    target_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(target_dir.glob("*"))

    if existing_files and not force:
        print(f"Skipping {dataset_name}: target directory already contains files.")
        print(f"Use --force to download again: {target_dir}")
        return

    print(f"Downloading {dataset_name} from Kaggle...")
    print(f"Target directory: {target_dir}")

    command = [
        kaggle_executable,
        "datasets",
        "download",
        "-d",
        kaggle_slug,
        "-p",
        str(target_dir),
        "--unzip",
    ]

    run_command(command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    config = load_config(root_dir)
    datasets = config.get("datasets", {})
    kaggle_executable = find_kaggle_executable()

    if not datasets:
        print("No datasets found in configs/datasets.yaml")
        sys.exit(1)

    print(f"Using Kaggle CLI: {kaggle_executable}")

    if args.dataset:
        if args.dataset not in datasets:
            print(f"Unknown dataset: {args.dataset}")
            print("Available datasets:")
            for dataset_name in datasets:
                print(f"- {dataset_name}")
            sys.exit(1)

        download_dataset(
            args.dataset,
            datasets[args.dataset],
            root_dir,
            kaggle_executable,
            args.force,
        )
    else:
        for dataset_name, dataset_config in datasets.items():
            download_dataset(
                dataset_name,
                dataset_config,
                root_dir,
                kaggle_executable,
                args.force,
            )

    print("Download workflow finished.")


if __name__ == "__main__":
    main()