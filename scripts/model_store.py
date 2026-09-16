"""Model files live in GCS; this puts them on disk before a run needs them.

The models are not in git: the repository is public and they were trained on client
labels. They sit in the private `farmdar_data_catalog` bucket under a versioned folder,
and are downloaded once into `model_files/`, beside where the code already looked for
them. A file already on disk with the right checksum is used as it is, so a machine
that has the models never touches the network.

Every model carries an MD5, which is what GCS itself stores, so a truncated download or
a file replaced in the bucket stops the run instead of producing a map. The sidecar is
listed with the static model for the same reason: without it inference still runs, at
threshold 0.5 and with no domain guard, and nothing says so.

A new model version goes in a new folder (`v5/`), never over an existing object, so a
map made last month can still be reproduced from the model that made it.

Credentials, first found wins: `--gcs-key` / `GCS_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`,
the git-ignored key beside this file, then `gcloud auth application-default login`.

    python3 model_store.py            # fetch and verify every model
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPTS_DIR.parent / "model_files"
BUCKET = "farmdar_data_catalog"

log = logging.getLogger("model_store")

#: local file name -> (gs:// URI, md5 hex)
MODELS: Dict[str, tuple] = {
    "best_rf_classifier_v4.joblib": (
        f"gs://{BUCKET}/cropscan/cane/models/v4/best_rf_classifier_v4.joblib",
        "bd373168648fd605896ade30bf0b44aa"),
    "fao_cane_xgb_model_v4.json": (
        f"gs://{BUCKET}/cropscan/cane/models/v4/fao_cane_xgb_model_v4.json",
        "7dae08eaaf0bee3598f9ee36ae2ad623"),
    "fao_cane_xgb_model_v4.sidecar.json": (
        f"gs://{BUCKET}/cropscan/cane/models/v4/fao_cane_xgb_model_v4.sidecar.json",
        "acd65fcdcd47d1f90ebc183dee4082f1"),
}

#: A model and the files that must arrive with it.
COMPANIONS = {"fao_cane_xgb_model_v4.json": ["fao_cane_xgb_model_v4.sidecar.json"]}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _client(key: Optional[str]):
    from google.cloud import storage

    key = key or os.environ.get("GCS_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key:
        found = sorted(SCRIPTS_DIR.glob("gcs_data_downloader*.json"))
        key = str(found[0]) if found else None
    if key:
        return storage.Client.from_service_account_json(key)
    return storage.Client()     # application-default credentials


def _fetch_one(name: str, key: Optional[str]) -> Path:
    uri, md5 = MODELS[name]
    local = MODEL_DIR / name
    if local.exists():
        if _md5(local) == md5:
            return local
        raise RuntimeError(f"{local} exists but its checksum does not match {uri}. "
                           f"Move it aside rather than let a run use the wrong model.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bucket, blob_path = uri[len("gs://"):].split("/", 1)
    partial = local.with_name(local.name + ".part")
    log.info("downloading %s", uri)
    _client(key).bucket(bucket).blob(blob_path).download_to_filename(str(partial), timeout=900)
    got = _md5(partial)
    if got != md5:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{uri} downloaded with md5 {got}, expected {md5}")
    partial.replace(local)
    return local


def ensure_model(name: str, key: Optional[str] = None) -> Path:
    """The local path of a model, downloading it and its companions first if needed."""
    for companion in COMPANIONS.get(name, []):
        _fetch_one(companion, key)
    return _fetch_one(name, key)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gcs-key", default=None, help="service-account key JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name in MODELS:
        print(f"ok  {ensure_model(name, args.gcs_key)}")


if __name__ == "__main__":
    main()
