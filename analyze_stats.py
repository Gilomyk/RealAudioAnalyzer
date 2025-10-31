import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Union

import config.config as cfg


def flatten_json(d: Union[Dict, List, float, int],
                 parent_key: str = "",
                 sep: str = "-") -> Dict[str, float]:
    """
    Rekurencyjnie spłaszcza strukturę JSON do kluczy i wartości liczbowych.
    Przykład:
      {"bands": {"bass": 0.1, "mid": 0.3}} -> {"bands-bass": 0.1, "bands-mid": 0.3}
    """
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_json(v, new_key, sep=sep))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}[{i}]"
            items.update(flatten_json(v, new_key, sep=sep))
    else:
        if isinstance(d, (int, float)):
            items[parent_key] = float(d)
    return items


def compute_min_max(frames: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Dla listy ramek (frames) wylicza minimalne i maksymalne wartości dla każdego klucza liczbowego.
    """
    all_values = defaultdict(list)
    for frame in frames:
        flat = flatten_json(frame)
        for k, v in flat.items():
            all_values[k].append(v)

    result = {}
    for k, vals in all_values.items():
        result[k] = {
            "min": min(vals),
            "max": max(vals)
        }
    return result


def append_stats_to_json(json_path: str, out_path: str = None):
    """
    Wczytuje plik JSON z analizą audio, liczy statystyki min/max
    i dopisuje je w sekcji "stats".
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        raise ValueError("Brak danych w sekcji 'frames' — czy plik jest poprawny?")

    stats = compute_min_max(frames)
    data["stats"] = stats

    out_file = Path(out_path) if out_path else json_path
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Dopisano statystyki do: {out_file}")
    print(f"Liczba zmiennych: {len(stats)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analiza statystyk min/max z pliku JSON")
    parser.add_argument("--input", help="Ścieżka do pliku JSON z analizą audio", default=cfg.DEFAULT_OUTPUT_PATH)
    parser.add_argument("--out", help="Opcjonalna ścieżka wyjściowa (zaktualizowany plik JSON)", default=cfg.DEFAULT_OUTPUT_STATS_PATH)
    args = parser.parse_args()

    append_stats_to_json(args.input, args.out)
