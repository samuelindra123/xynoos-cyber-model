"""Check tracked files for obvious private identifiers and generated artifacts."""

from pathlib import Path
import re
import subprocess

MAX_FILE_BYTES = 256 * 1024
OBJECTIVE_A_V2_FILES = {
    "datasets/task006-objective-a-v2/manifest.json",
    "datasets/task006-objective-a-v2/sources.jsonl",
    "datasets/task006-objective-a-v2/documents.jsonl",
    "datasets/task006-objective-a-v2/tokenizer-fit.jsonl",
    "datasets/task006-objective-a-v2/tokenizer.tsv",
    "datasets/task006-objective-a-v2/task-index.json",
}
ARTIFACT_EXTENSIONS = {
    ".a", ".bin", ".ckpt", ".csv", ".dylib", ".exe", ".gguf", ".log",
    ".model", ".npy", ".npz", ".o", ".parquet", ".prof", ".pt",
    ".safetensors", ".so", ".sqlite", ".tensor", ".trace", ".zip",
}
PRIVATE_PATTERNS = {
    "personal absolute path": re.compile(
        r"/(?:home|Users)/[A-Za-z0-9_.-]+|[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+"
    ),
    "machine inventory": re.compile(
        r"^\s*(?:[-*] )?(?:hostname|cpu model|gpu model|product serial|machine-id)\s*[:=]",
        re.IGNORECASE,
    ),
    "credential pattern": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
        r"(?:AKIA|ASIA)[A-Z0-9]{16})\b|-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----"
    ),
}


def violations(path):
    value = path.as_posix()
    allowed_dataset = value in OBJECTIVE_A_V2_FILES or re.fullmatch(
        r"datasets/task006-objective-a-v2/payload/oa2-[0-9]{3}\.txt", value
    )
    if (
        any(part in {"target", "__pycache__", ".venv"} for part in path.parts)
        or path.name.startswith(("hardware-report", "system-info"))
        or (path.name.startswith(".env") and path.name != ".env.example")
        or (path.parts[0] == "datasets" and value != "datasets/README.md" and not allowed_dataset)
        or path.suffix.lower() in ARTIFACT_EXTENSIONS
    ):
        yield "unexpected artifact path"
        return
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        yield "unexpected file type or size"
        return
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        yield "non-UTF-8 file"
        return
    if "\0" in content:
        yield "binary file"
        return
    for line_number, line in enumerate(content.splitlines(), 1):
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(line):
                yield f"{label} on line {line_number}"


def main():
    entries = subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8")
    paths = [Path(entry) for entry in entries.split("\0") if entry]
    findings = [
        f"Tracked file {index}: {finding}"
        for index, path in enumerate(paths, 1)
        for finding in violations(path)
    ]
    if findings:
        # Avoid echoing content or filenames that may themselves be private.
        print("File numbers use the order from git ls-files.")
        raise SystemExit("\n".join(findings))
    print(f"Checked {len(paths)} tracked files for common privacy and artifact issues.")


if __name__ == "__main__":
    main()
