import json
import sys
from pathlib import Path


def decode_json_file(path_str: str) -> None:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Dump with ensure_ascii=False to write readable UTF-8
    text = json.dumps(data, ensure_ascii=False, indent=4)
    # Ensure trailing newline like many editors prefer
    if not text.endswith("\n"):
        text += "\n"
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/decode_json_unicode.py <json-file> [<json-file> ...]")
        return 1
    for file_arg in argv[1:]:
        decode_json_file(file_arg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

