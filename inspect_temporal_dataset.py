from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT / 'data' / 'temporal' / 'asl_dataset'
DEFAULT_LABEL_MAP_PATH = ROOT / 'models' / 'temporal' / 'temporal_label_map.json'
DEFAULT_TARGET_REPS = 15


def load_label_map(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def collect_dataset_summary(dataset_dir: Path) -> tuple[dict[str, int], int]:
    summary: dict[str, int] = {}
    total = 0

    if not dataset_dir.exists():
        return summary, total

    for sign_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        count = sum(1 for sample in sign_dir.iterdir() if sample.is_file() and sample.suffix == '.npy')
        if count == 0:
            continue
        summary[sign_dir.name] = count
        total += count

    return summary, total


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Inspect the shared temporal ASL dataset and compare it with the saved checkpoint label map.',
    )
    parser.add_argument('--dataset', default=str(DEFAULT_DATASET_DIR))
    parser.add_argument('--label-map', default=str(DEFAULT_LABEL_MAP_PATH))
    parser.add_argument('--target-reps', type=int, default=DEFAULT_TARGET_REPS)
    parser.add_argument('--json', action='store_true', dest='as_json')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    label_map_path = Path(args.label_map)

    counts, total_reps = collect_dataset_summary(dataset_dir)
    label_map = load_label_map(label_map_path)
    trained_classes = sorted(str(c) for c in label_map.get('classes', []))

    dataset_classes = sorted(counts)
    under_target = {word: n for word, n in counts.items() if n < args.target_reps}
    dataset_only = sorted(set(dataset_classes) - set(trained_classes))
    checkpoint_only = sorted(set(trained_classes) - set(dataset_classes))
    needs_retrain = bool(dataset_only or checkpoint_only)

    payload = {
        'dataset_dir': str(dataset_dir),
        'label_map_path': str(label_map_path),
        'dataset_classes': dataset_classes,
        'dataset_class_count': len(dataset_classes),
        'dataset_total_reps': total_reps,
        'trained_classes': trained_classes,
        'trained_class_count': len(trained_classes),
        'under_target': under_target,
        'dataset_only': dataset_only,
        'checkpoint_only': checkpoint_only,
        'needs_retrain': needs_retrain,
    }

    if args.as_json:
        print(json.dumps(payload, indent=2))
        return 0

    print('Temporal Dataset Summary')
    print(f'Dataset dir        : {dataset_dir}')
    print(f'Label map          : {label_map_path}')
    print(f'Dataset classes    : {len(dataset_classes)}')
    print(f'Total reps         : {total_reps}')
    print(f'Checkpoint classes : {len(trained_classes)}')
    print()

    if dataset_classes:
        print('Reps per word')
        for word in dataset_classes:
            count = counts[word]
            marker = ' OK' if count >= args.target_reps else ' LOW'
            print(f'  {word:<20} {count:>3} reps{marker}')
        print()
    else:
        print('No dataset classes found.\n')

    if under_target:
        print(f'Words below target ({args.target_reps} reps)')
        for word, count in under_target.items():
            print(f'  {word:<20} {count:>3}')
        print()

    if dataset_only:
        print('In dataset but not in checkpoint label map')
        for word in dataset_only:
            print(f'  {word}')
        print()

    if checkpoint_only:
        print('In checkpoint label map but missing from dataset')
        for word in checkpoint_only:
            print(f'  {word}')
        print()

    print(f'Retrain needed     : {"yes" if needs_retrain else "no"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
