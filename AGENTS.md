# Patch backups

Before changing application source or configuration, create a timestamped source backup:

```powershell
.venv/Scripts/python.exe -c "from src.backup import backup_sources; print(backup_sources('.'))"
```

Use `LASS_BACKUP_DIR` for a user-selected destination; otherwise use the Windows Documents/Backups folder. A backup failure must stop the patch. One snapshot is sufficient for a continuous patch task; create another before a new task. Do not include models, build outputs, or user databases in this source snapshot.

For Git patch files, use `scripts/apply_patch.py`, which checks the patch and backs up automatically before applying it. Builds and installer upgrades also perform automatic source/configuration snapshots. Keep user data and existing unrelated edits intact.
