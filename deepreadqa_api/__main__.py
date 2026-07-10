"""Run the API server: python -m deepreadqa_api [--host H] [--port P]."""
from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from .app import create_app
from .config import ApiConfig


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(prog="deepreadqa_api")
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "8000")))
    args = ap.parse_args(argv)
    app = create_app(ApiConfig.from_env())
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                access_log=False)  # access lines come from our middleware


if __name__ == "__main__":
    main()
