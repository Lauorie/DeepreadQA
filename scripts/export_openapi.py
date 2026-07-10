"""Export the OpenAPI spec to docs/api/openapi.json (kept in sync with code)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepreadqa_api.app import create_app  # noqa: E402
from deepreadqa_api.config import ApiConfig  # noqa: E402


def main() -> None:
    # dummy key: config only gates startup, not schema generation
    app = create_app(ApiConfig(api_keys=("schema-export",)))
    out = Path(__file__).resolve().parent.parent / "docs" / "api" / "openapi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(app.openapi(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
