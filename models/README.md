# Local model assets

LASS keeps the EXAONE runtime snapshot under `models/exaone-4.0-1.2b/` so that
inference never needs a remote API or a shared global model cache.

```bash
python scripts/manage_models.py download
python scripts/manage_models.py status
python scripts/manage_models.py verify
```

The downloaded weights are about 2.56 GB. Large model binaries are intentionally
excluded from Git; the generated `lass-model-manifest.json` records their exact
size and SHA-256 checksum. Review LG AI Research's EXAONE license in the downloaded
snapshot before redistribution.
