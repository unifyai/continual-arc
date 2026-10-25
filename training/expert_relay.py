"""The expert's endpoint for the labeller: an LLMProxy on 127.0.0.1:PORT that forwards to OpenRouter with the key the
relay alone holds, every request pinned to the one expert model and its hosts. realtime.sh starts it on the node that
plays, beside the labeller; it can as well run on another machine, with the node's localhost:PORT tunnelled to it
(ssh -R), so that the node never holds a key and the tunnel reaches nothing but that model. The key is
OPENROUTER_API_KEY from the environment, or else from the baselines' .env file (BASELINES/.env, BASELINES defaulting to
the baselines folder of this repository).
Usage: python training/expert_relay.py [--port 18200] [--model z-ai/glm-5.3] [--providers fireworks,parasail,gmicloud]
"""
from __future__ import annotations

import argparse
import os
import signal
import threading
from pathlib import Path

from continual_arc_baselines.proxy import LLMProxy

KEY = "OPENROUTER_API_KEY"
ENV_FILE = Path(os.environ.get("BASELINES") or Path(__file__).resolve().parent.parent / "baselines") / ".env"


def read_key() -> str:
    """The OpenRouter key from the environment, or from the baselines' .env file."""
    if os.environ.get(KEY):
        return os.environ[KEY]
    for line in (ENV_FILE.read_text().splitlines() if ENV_FILE.is_file() else []):
        name, _, value = line.partition("=")
        if name.strip() == KEY:
            return value.strip().strip('"').strip("'")
    raise SystemExit(f"no {KEY} in the environment or {ENV_FILE}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=18200)
    p.add_argument("--model", default="z-ai/glm-5.3")
    p.add_argument("--providers", default="fireworks,parasail,gmicloud")
    p.add_argument("--max-connections", type=int, default=1024, help="the expert's calls open at once (labelling runs "
                   "hundreds in parallel; httpx's default is 100)")
    a = p.parse_args()
    pinned = {"model": a.model, "provider": {"order": a.providers.split(","), "allow_fallbacks": False}}
    proxy = LLMProxy("https://openrouter.ai/api/v1", api_key=read_key(), port=a.port, body_overrides=pinned,
                     declared_context_length=None, timeout=600.0, max_connections=a.max_connections).start()
    print(f"expert relay on {proxy.base_url}: {a.model} via {a.providers}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        proxy.stop()


if __name__ == "__main__":
    main()
