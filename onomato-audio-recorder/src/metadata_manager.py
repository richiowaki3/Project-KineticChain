# -*- coding: utf-8 -*-
"""
metadata_manager.py: Structured metadata management for onomatopoeia audio datasets.
Provides dataset persistence, querying, CSV export, and alignment with OnomaDict.
Part of the onomato-audio-recorder toolkit.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime


class MetadataManager:
    """
    Manages structured metadata for human recordings and TTS-generated onomatopoeia audio.
    """

    def __init__(self, dataset_dir: Optional[str] = None):
        if dataset_dir is None:
            self.dataset_dir = Path(__file__).resolve().parent.parent / "dataset_output"
        else:
            self.dataset_dir = Path(dataset_dir)

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.dataset_dir / "metadata.json"
        self._records: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> Dict[str, Dict[str, Any]]:
        """Loads metadata.json if it exists."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._records = {item["id"]: item for item in data if "id" in item}
                    elif isinstance(data, dict):
                        self._records = data
            except Exception as e:
                print(f"[Metadata Warning] Could not parse {self.metadata_file}: {e}")
                self._records = {}
        return self._records

    def save(self) -> None:
        """Saves current records to metadata.json."""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(list(self._records.values()), f, ensure_ascii=False, indent=2)

    def add_entry(
        self,
        entry_id: str,
        word: str,
        lang: str,
        file_path: str,
        variant: str = "root",
        spoken_text: Optional[str] = None,
        speaker_type: str = "human",  # 'human' or 'tts'
        voice_id: Optional[str] = None,
        sample_rate: int = 44100,
        bit_depth: int = 16,
        duration_sec: float = 0.0,
        onomadict_ref: Optional[Dict[str, Any]] = None,
        extra_tags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Adds or updates an audio recording/generation entry.
        """
        record = {
            "id": entry_id,
            "word": word,
            "lang": lang,
            "variant": variant,
            "spoken_text": spoken_text or word,
            "speaker_type": speaker_type,
            "voice_id": voice_id or ("mic" if speaker_type == "human" else "unknown"),
            "file_path": str(Path(file_path).resolve()),
            "file_name": Path(file_path).name,
            "sample_rate": sample_rate,
            "bit_depth": bit_depth,
            "duration_sec": round(duration_sec, 4),
            "created_at": datetime.now().isoformat(),
            "onomadict_ref": onomadict_ref or {},
            "tags": extra_tags or {},
        }
        self._records[entry_id] = record
        self.save()
        return record

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves entry by ID."""
        return self._records.get(entry_id)

    def list_entries(
        self,
        lang: Optional[str] = None,
        speaker_type: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filters entries by language, speaker type, or variant."""
        results = []
        for rec in self._records.values():
            if lang and rec.get("lang") != lang:
                continue
            if speaker_type and rec.get("speaker_type") != speaker_type:
                continue
            if variant and rec.get("variant") != variant:
                continue
            results.append(rec)
        return results

    def export_csv(self, output_path: Optional[str] = None) -> str:
        """Exports all metadata to a CSV file."""
        if output_path is None:
            csv_path = self.dataset_dir / "metadata.csv"
        else:
            csv_path = Path(output_path)

        entries = list(self._records.values())
        if not entries:
            return str(csv_path)

        fieldnames = [
            "id", "word", "lang", "variant", "spoken_text",
            "speaker_type", "voice_id", "file_name", "duration_sec",
            "sample_rate", "bit_depth", "created_at"
        ]

        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for e in entries:
                writer.writerow(e)

        return str(csv_path.resolve())

    @staticmethod
    def load_onomadict_words(
        dict_path: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Loads candidate onomatopoeia words from OnomaDict dictionary JSON.
        """
        candidates = [
            Path(dict_path) if dict_path else None,
            Path("data/onomatopoeia_dictionary.json"),
            Path(__file__).resolve().parent.parent.parent / "data" / "onomatopoeia_dictionary.json",
            Path("D:/Antigravity_Work/OnomaDict/data/onomatopoeia_dictionary.json"),
        ]
        chosen_path = None
        for c in candidates:
            if c and c.exists():
                chosen_path = c
                break

        if not chosen_path:
            return []

        with open(chosen_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        words = []
        for idx, item in enumerate(raw_data):
            w_lang = item.get("language", "JP")
            if languages and w_lang not in languages:
                continue
            words.append({
                "id": f"{w_lang}_{idx:04d}",
                "word": item.get("word", ""),
                "lang": "ja" if w_lang == "JP" else ("ko" if w_lang == "KR" else "sw"),
                "meaning_en": item.get("meaning_en", ""),
                "effort": item.get("effort", {}),
            })
        return words
