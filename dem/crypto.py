"""Local key management and field-level encryption.

Keys live in <data_dir>/keys with 0600 permissions, separate from the vault.
Production should use the OS keychain / Secure Enclave; the file fallback is
documented as such. Key loss = data loss (by design).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from pathlib import Path

from cryptography.fernet import Fernet


class Keyring:
    def __init__(self, keys_dir: Path):
        keys_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.keys_dir = keys_dir
        self.fernet_path = keys_dir / "master.key"
        self.hmac_path = keys_dir / "hmac.key"
        if not self.fernet_path.exists():
            self._write_secret(self.fernet_path, Fernet.generate_key())
        if not self.hmac_path.exists():
            self._write_secret(self.hmac_path, secrets_token())

    @staticmethod
    def _write_secret(path: Path, data: bytes) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)

    @property
    def fernet(self) -> Fernet:
        return Fernet(self.fernet_path.read_bytes().strip())

    @property
    def hmac_key(self) -> bytes:
        return self.hmac_path.read_bytes().strip()

    def encrypt(self, text: str) -> bytes:
        return self.fernet.encrypt(text.encode("utf-8"))

    def decrypt(self, token: bytes) -> str:
        return self.fernet.decrypt(token).decode("utf-8")

    def tag(self, text: str) -> str:
        """Keyed HMAC — never a plain hash for pseudonymous IDs."""
        return hmac.new(self.hmac_key, text.encode("utf-8"), hashlib.sha256).hexdigest()


def secrets_token() -> bytes:
    return base64.urlsafe_b64encode(os.urandom(32))
