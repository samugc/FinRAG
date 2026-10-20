
"""Download and validate FinanceBench holdout documents."""

import json
import shutil
from pathlib import Path
from urllib.request import urlopen

from pypdf import PdfReader

base = Path("data/external/financebench")
manifest = json.loads(
    Path("data/eval/financebench_holdout.json").read_text()
)

repo_url = (
    "https://raw.githubusercontent.com/"
    "patronus-ai/financebench/main/pdfs"
)

base.mkdir(parents=True, exist_ok=True)
failed = []

for name in manifest["documents"]:
    destination = base / f"{name}.pdf"
    temporary = base / f"{name}.pdf.part"

    try:
        if destination.exists():
            pages = len(PdfReader(destination).pages)
            if pages > 0:
                print(f"Ya existe: {name} ({pages} páginas)")
                continue

        url = f"{repo_url}/{name}.pdf"

        with urlopen(url, timeout=120) as response, temporary.open("wb") as file:
            shutil.copyfileobj(response, file)

        with temporary.open("rb") as file:
            if file.read(5) != b"%PDF-":
                raise ValueError("El archivo descargado no es un PDF")

        pages = len(PdfReader(temporary).pages)
        if pages == 0:
            raise ValueError("PDF sin páginas")

        temporary.replace(destination)

        print(f"Descargado: {name} ({pages} páginas)")

    except Exception as error:  # noqa: BLE001 - record failures and continue
        failed.append(name)
        temporary.unlink(missing_ok=True)
        print(f"ERROR: {name}: {error}")

print(
    f"\nResultado: {len(manifest['documents']) - len(failed)}"
    f"/{len(manifest['documents'])} documentos disponibles"
)

if failed:
    raise SystemExit(f"Descargas fallidas: {failed}")
