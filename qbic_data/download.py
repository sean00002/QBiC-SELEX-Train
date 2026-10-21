"""Download the raw HT-SELEX reads listed in Table S1.

Three kinds of ``submitted_ftp`` entries:
* ``ftp.sra.ebi.ac.uk/...``: fetched over HTTPS from ENA.
* ``prefetch <run> fastq-dump --gzip``: SRA runs; fetched as fastq.gz from ENA's
  mirror (no SRA toolkit needed), saved under the Table S1 file name.
* the Rodriguez-Martinez et al. (2017) Zenodo archive: downloaded once and the
  needed 20 bp text files extracted.
"""

import json
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

ENA_FILEREPORT = ("https://www.ebi.ac.uk/ena/portal/api/filereport?accession={acc}"
                  "&result=read_run&fields=run_accession,fastq_ftp,fastq_bytes&format=json")


def _fetch(url: str, dest: Path, retries: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
                shutil.copyfileobj(r, out, length=1 << 20)
            tmp.replace(dest)
            return
        except Exception as e:  # network errors: retry, then give up
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"download failed after {retries} attempts: {url}: {e}") from e
            time.sleep(5 * attempt)


def ena_fastq_url(accession: str) -> str:
    """HTTPS URL of the fastq.gz that ENA serves for an SRA/ENA run."""
    with urllib.request.urlopen(ENA_FILEREPORT.format(acc=accession), timeout=60) as r:
        records = json.load(r)
    if not records or not records[0].get("fastq_ftp"):
        raise RuntimeError(f"ENA has no fastq file for {accession}")
    urls = records[0]["fastq_ftp"].split(";")
    url = next((u for u in urls if u.endswith(f"{accession}.fastq.gz")), urls[0])
    return "https://" + url


def source_url(submitted_ftp: str) -> str:
    entry = submitted_ftp.strip()
    if entry.startswith("ftp.sra.ebi.ac.uk"):
        return "https://" + entry
    if entry.startswith("prefetch"):
        return ena_fastq_url(entry.split()[1])
    if entry.startswith("https://"):
        return entry
    raise ValueError(f"unrecognised download entry: {submitted_ftp}")


def download_files(files, reads_dir, log=print) -> list[tuple]:
    """Download Table S1 cycle files into ``<reads_dir>/<source>/<file>``.

    Existing files are skipped. Returns a list of (file, error) for failures.
    """
    reads_dir = Path(reads_dir)
    failures, archives = [], {}
    for f in files:
        dest = f.path(reads_dir)
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            url = source_url(f.submitted_ftp)
            if ".tar.gz" in url:              # Rodriguez-Martinez et al. (2017) archive
                archives.setdefault(url, []).append(f)
                continue
            log(f"  {f.source}/{f.filename}")
            _fetch(url, dest)
        except Exception as e:
            failures.append((f, str(e)))
            log(f"  FAILED {f.source}/{f.filename}: {e}")
    for url, members in archives.items():
        try:
            _extract_from_archive(url, members, reads_dir, log)
        except Exception as e:
            failures += [(f, str(e)) for f in members]
    return failures


def _extract_from_archive(url, members, reads_dir: Path, log) -> None:
    archive = reads_dir / "_archives" / url.split("/")[-1].split("?")[0]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        log(f"  {archive.name} (Zenodo archive, {len(members)} files needed)")
        _fetch(url, archive)
    wanted = {f.filename: f for f in members}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = member.name.split("/")[-1]
            if member.isfile() and name in wanted:
                dest = wanted[name].path(reads_dir)
                with tar.extractfile(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                del wanted[name]
    if wanted:
        raise RuntimeError(f"not found in {archive.name}: {sorted(wanted)}")
