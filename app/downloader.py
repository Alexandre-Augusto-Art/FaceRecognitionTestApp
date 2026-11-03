"""Download dos pesos dos modelos.

Todos os arquivos vem de fontes publicas e gratuitas, sem conta nem API key.
Depois do primeiro download o app funciona inteiramente offline.
"""

from __future__ import annotations

import bz2
import shutil
import sys
import zipfile
from pathlib import Path

import requests

from .config import MODEL_CATALOG, MODELS_DIR, ModelSpec

CHUNK = 1 << 16


def is_present(spec: ModelSpec) -> bool:
    return spec.target.exists() and spec.target.stat().st_size > 0


def missing_models(keys: list[str] | None = None) -> list[ModelSpec]:
    specs = [MODEL_CATALOG[k] for k in (keys or MODEL_CATALOG.keys())]
    return [s for s in specs if not is_present(s)]


def _download_raw(spec: ModelSpec, dest: Path, progress: bool = True) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(spec.url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(CHUNK):
                if not chunk:
                    continue
                handle.write(chunk)
                done += len(chunk)
                if progress and total:
                    # O GitHub serve conteudo comprimido, entao o Content-Length
                    # pode ser menor que o total realmente escrito.
                    pct = min(100.0, done * 100 / total)
                    sys.stdout.write(
                        f"\r  {spec.key:<22} {pct:5.1f}%  "
                        f"({done / 1048576:.1f}/{total / 1048576:.1f} MB)"
                    )
                    sys.stdout.flush()
    if progress:
        sys.stdout.write("\n")
    tmp.replace(dest)


def _post_process(spec: ModelSpec, downloaded: Path) -> None:
    """Extrai o arquivo desejado quando o download e um zip ou bz2."""
    if spec.archive == "zip":
        with zipfile.ZipFile(downloaded) as archive:
            names = archive.namelist()
            member = spec.member
            if member not in names:
                # Alguns pacotes vem sem o diretorio raiz no zip.
                tail = Path(spec.member or "").name
                matches = [n for n in names if Path(n).name == tail]
                if not matches:
                    raise FileNotFoundError(
                        f"{spec.member} nao encontrado em {downloaded.name}. "
                        f"Conteudo: {names[:20]}"
                    )
                member = matches[0]
            with archive.open(member) as src, open(spec.target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        downloaded.unlink(missing_ok=True)
    elif spec.archive == "bz2":
        with bz2.open(downloaded, "rb") as src, open(spec.target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        downloaded.unlink(missing_ok=True)


def fetch(spec: ModelSpec, progress: bool = True, force: bool = False) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if is_present(spec) and not force:
        return spec.target
    landing = MODELS_DIR / spec.filename if spec.archive else spec.target
    _download_raw(spec, landing, progress=progress)
    if spec.archive:
        _post_process(spec, landing)
    return spec.target


def fetch_all(keys: list[str] | None = None, progress: bool = True) -> dict[str, str]:
    """Baixa os modelos pedidos. Falha em um modelo nao aborta os demais."""
    specs = [MODEL_CATALOG[k] for k in (keys or MODEL_CATALOG.keys())]
    report: dict[str, str] = {}
    for spec in specs:
        if is_present(spec):
            report[spec.key] = "ja presente"
            if progress:
                print(f"  {spec.key:<22} ja presente")
            continue
        try:
            fetch(spec, progress=progress)
            report[spec.key] = "ok"
        except Exception as exc:  # noqa: BLE001 - queremos continuar mesmo assim
            report[spec.key] = f"falhou: {exc}"
            if progress:
                print(f"\n  {spec.key:<22} FALHOU: {exc}")
    return report


def main() -> int:
    total = sum(s.approx_mb for s in MODEL_CATALOG.values() if not is_present(s))
    print(f"Destino: {MODELS_DIR}")
    print(f"A baixar: ~{total:.0f} MB\n")
    report = fetch_all()
    failures = {k: v for k, v in report.items() if v.startswith("falhou")}
    print("\nResumo:")
    for key, status in report.items():
        print(f"  {key:<22} {status}")
    if failures:
        print(
            f"\n{len(failures)} modelo(s) falharam. Os detectores correspondentes "
            "aparecerao como indisponiveis na interface; o resto do app funciona."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
