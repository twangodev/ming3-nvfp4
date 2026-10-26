"""Verified training-source audio extraction and deterministic question selection.

Clotho audio retains individual licenses and attribution in the selected replay metadata.
GSM8K is MIT; BELLE remains subject to research/noncommercial restrictions.
"""

from __future__ import annotations

import hashlib
import io
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from .common import disk_budget, file_hash, write_json

ZENODO = "https://zenodo.org/api/records/6473207/files/"
OFFICIAL = {
    "clotho_aqa_train.csv": ("508513b91001ab17e641eaaa162ca045", 1453735),
    "clotho_aqa_metadata.csv": ("e555d8a5a4b275ca7c437a33c0739d65", 427441),
    "audio_files.zip": ("48c2c46ccd71dc06121228e2ed68d6b9", 3140677494),
}


def verify_official(path: Path, name: str) -> None:
    checksum, size = OFFICIAL[name]
    if path.stat().st_size != size:
        raise ValueError(f"Wrong official source size: {name}")
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != checksum:
        raise ValueError(f"Wrong official source checksum: {name}")


def bounded_text(prompt: str, answer: str) -> bool:
    # Conservative UTF-8 byte ceiling, including teacher-forced answer; no truncation.
    return bool(prompt.strip() and answer.strip()) and len((prompt + answer).encode()) <= 1400


def choose_audio_questions(records: list[dict], count: int, seed: int) -> list[dict]:
    """One confident majority answer per recording, shuffled over the whole train split."""
    grouped = {}
    for row in records:
        if row["confidence"] == "yes":
            key = (row["file_name"], row["QuestionText"])
            grouped.setdefault(key, []).append(row["answer"])
    candidates = []
    for (filename, question), answers in grouped.items():
        frequencies = Counter(answers)
        answer, votes = frequencies.most_common(1)[0]
        if votes >= 2 and votes > len(answers) / 2:
            candidates.append(
                dict(
                    file_name=filename, question=question, answer=answer, agreeing_annotations=votes
                )
            )
    random.Random(seed).shuffle(candidates)
    selected, seen = [], set()
    for row in candidates:
        if row["file_name"] not in seen:
            selected.append(row)
            seen.add(row["file_name"])
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} distinct confident training recordings")


def fetch_clotho_audio(assets: Path, questions: list[dict]) -> dict:
    """Download selected immutable-record ZIP members; verify central-directory CRCs."""
    import requests

    archive_size = OFFICIAL["audio_files.zip"][1]
    url = ZENODO + "audio_files.zip/content"
    tail_size = 500000
    tail = assets / "audio_files.zip.central-tail"
    if not tail.exists():
        response = requests.get(url, headers={"Range": f"bytes=-{tail_size}"}, timeout=120)
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != tail_size:
            raise ValueError("Server did not honor bounded central-directory request")
        tail.write_bytes(response.content)
    archive = zipfile.ZipFile(io.BytesIO(tail.read_bytes()))
    names = {Path(info.filename).name: info for info in archive.infolist()}
    audio_dir = assets / "clotho-train"
    audio_dir.mkdir(exist_ok=True)
    cache = assets / "clotho-audio-range-receipts.json"
    receipts = json.loads(cache.read_text()) if cache.exists() else {}
    total = sum(names[q["file_name"]].compress_size for q in questions)
    started = time.monotonic()

    def fetch(item):
        name = item["file_name"]
        if name in receipts:
            receipt = receipts[name]
            path = audio_dir / (receipt["sha256"] + ".wav")
            if path.exists() and file_hash(path) == receipt["sha256"]:
                return name, receipt
        info = names[name]
        offset = info.header_offset + archive_size - tail_size
        end = (
            offset + 30 + len(info.filename.encode()) + len(info.extra) + info.compress_size + 2048
        )
        disk_budget(audio_dir, 100, info.file_size)
        for attempt in range(6):
            response = requests.get(url, headers={"Range": f"bytes={offset}-{end}"}, timeout=120)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(min(60, 10 * (attempt + 1)))
                continue
            response.raise_for_status()
            if response.status_code != 206:
                raise ValueError("Server did not honor bounded member request")
            raw = response.content
            if raw[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP local header")
            nlen, elen = struct.unpack_from("<HH", raw, 26)
            compressed = raw[30 + nlen + elen : 30 + nlen + elen + info.compress_size]
            audio = zlib.decompress(compressed, -15) if info.compress_type == 8 else compressed
            if len(audio) != info.file_size or zlib.crc32(audio) != info.CRC:
                raise ValueError("ZIP member failed size/CRC verification")
            checksum = hashlib.sha256(audio).hexdigest()
            path = audio_dir / (checksum + ".wav")
            path.write_bytes(audio)
            return name, dict(
                sha256=checksum,
                size_bytes=len(audio),
                crc32=info.CRC,
                archive_member=info.filename,
                byte_range=[offset, end],
            )
        raise RuntimeError(f"Cannot fetch {name}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, (name, receipt) in enumerate(executor.map(fetch, questions), 1):
            receipts[name] = receipt
            write_json(cache, receipts)
            if index % 16 == 0:
                print(
                    f"Clotho: {index}/{len(questions)} recordings; selected compressed total "
                    f"{total / 2**20:.1f} MiB; elapsed {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    return dict(
        url=url,
        revision="6473207",
        expected_archive_size_bytes=archive_size,
        publisher_expected_archive_md5=OFFICIAL["audio_files.zip"][0],
        whole_archive_checksum_verified=False,
        verification="Selected audio size and CRC32 checked against ZIP central directory; "
        "local SHA256 recorded. Whole archive not downloaded or hashed.",
        central_directory_tail_sha256=file_hash(tail),
        receipts_sha256=file_hash(cache),
        selected_compressed_bytes=total,
    )
