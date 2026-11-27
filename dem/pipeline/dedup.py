"""Exact + MinHash/LSH near-duplicate detection (pure Python, no deps)."""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from dataclasses import dataclass, field

PRIME = (1 << 61) - 1


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation-insensitive for hashing/shingles
    return re.sub(r"\s+", " ", text).strip()


def shingles(text: str, k: int = 5) -> set[str]:
    words = normalize(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


@dataclass
class DedupResult:
    unique: list = field(default_factory=list)
    duplicates: dict[str, str] = field(default_factory=dict)  # duplicate id -> kept id


class Deduplicator:
    def __init__(self, perms: int = 64, bands: int = 16, threshold: float = 0.85):
        self.perms = perms
        self.bands = bands
        self.rows = perms // bands
        self.threshold = threshold
        rng = random.Random(20261121)
        self.coeffs = [(rng.randrange(1, PRIME), rng.randrange(0, PRIME)) for _ in range(perms)]
        self.index: dict[tuple[int, tuple[int, ...]], list[str]] = {}
        self.hashes: dict[str, int] = {}
        self.sigs: dict[str, tuple[int, ...]] = {}
        self.shingle_sets: dict[str, set[str]] = {}

    def _signature(self, shingle_set: set[str]) -> tuple[int, ...]:
        values = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in shingle_set]
        if not values:
            return tuple([0] * self.perms)
        sig = []
        for a, b in self.coeffs:
            sig.append(min((a * v + b) % PRIME for v in values))
        return tuple(sig)

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def add(self, record) -> str | None:
        """Add a record; return the id of a duplicate it matches, else None."""
        text = record.prompt + "\n" + record.response
        digest = int(hashlib.sha256(normalize(text).encode()).hexdigest()[:16], 16)
        if digest in self.hashes:
            return self.hashes[digest]
        sh = shingles(text)
        sig = self._signature(sh)
        candidates: set[str] = set()
        for band in range(self.bands):
            key = (band, sig[band * self.rows:(band + 1) * self.rows])
            bucket = self.index.setdefault(key, [])
            candidates.update(bucket)
            bucket.append(record.id)
        for cand in candidates:
            if self._jaccard(sh, self.shingle_sets.get(cand, set())) >= self.threshold:
                return cand
        self.hashes[digest] = record.id
        self.sigs[record.id] = sig
        self.shingle_sets[record.id] = sh
        return None

    def run(self, records: list):
        """Deduplicate, keeping the highest-quality representative per cluster.

        A later record can match an earlier *duplicate* id, so matches are
        resolved to the canonical cluster root through an alias map.
        """
        groups: dict[str, list] = {}
        alias: dict[str, str] = {}
        for record in records:
            match = self.add(record)
            root = alias.get(match, match) if match else record.id
            groups.setdefault(root, []).append(record)
            alias[record.id] = root
        unique, duplicates = [], {}
        for keeper_id, group in groups.items():
            group.sort(key=lambda r: (r.quality.get("score", 0), r.id), reverse=True)
            keeper = group[0]
            unique.append(keeper)
            for dup in group[1:]:
                duplicates[dup.id] = keeper.id
        return DedupResult(unique=unique, duplicates=duplicates)
