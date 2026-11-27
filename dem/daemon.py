"""Long-lived local daemon: capture proxies, retention, kill switch. No telemetry."""
from __future__ import annotations

import time

from .adapters.api_proxy import start_api_proxy
from .adapters.ollama import start_ollama_proxy
from .config import load as load_config
from .crypto import Keyring
from .vault import Vault


def main() -> int:
    config = load_config()
    dirs = config.ensure_layout()
    keys = Keyring(dirs["keys"])
    vault = Vault(dirs["vault"], keys)

    consented = vault.db.execute("SELECT COUNT(*) AS n FROM consent_events").fetchone()["n"]
    if not consented:
        print("DEM is disabled: no consent receipt. Run `dem onboard` first.")
        return 1

    servers = []
    if config.enabled("capture", "ollama_proxy"):
        server, _ = start_ollama_proxy(config, vault)
        servers.append(server)
        print(f"Ollama proxy listening on 127.0.0.1:{config.get('capture', 'ollama_port')} "
              f"-> {config.get('capture', 'ollama_upstream')}")
    api = start_api_proxy(config, vault)
    if api:
        servers.append(api[0])
        print(f"API proxy listening on 127.0.0.1:{config.get('capture', 'api_port')} "
              f"-> {config.get('capture', 'api_upstream')}")
    if not servers:
        print("No capture adapters enabled. Enable one with `dem config set capture ollama_proxy true`.")
        return 1

    stop_flag = config.data_dir / "stop"
    if stop_flag.exists():
        stop_flag.unlink()
    print("DEM daemon running. Ctrl-C to stop. `dem kill` triggers a clean stop.")

    tick = 0
    try:
        while True:
            if stop_flag.exists():
                print("Kill switch triggered — stopping capture.")
                stop_flag.unlink()
                break
            if tick % 6 == 0:
                removed = vault.ttl_cleanup(int(config.get("retention", "raw_ttl_days", 30)))
                if removed:
                    print(f"Retention: purged {removed} expired unapproved captures.")
            time.sleep(10)
            tick += 1
    except KeyboardInterrupt:
        print("\nStopping DEM daemon.")
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        vault.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
