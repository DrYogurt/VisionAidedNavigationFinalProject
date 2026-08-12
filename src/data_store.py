import os
import json
import subprocess
import hashlib
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

class VersionedDataStore:
    """
    Handles versioned persistence and caching of experiment data payloads.
    Stores raw metrics, environment layouts, and metadata to prevent re-running completed tests.
    """

    VERSION = "v8.0.0"

    def __init__(self, base_dir: str = "results/data_v8"):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def get_git_revision(self) -> str:
        try:
            cmd = ["git", "rev-parse", "HEAD"]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode('utf-8').strip()
            return out
        except Exception:
            return "unknown_commit"

    def get_source_fingerprint(self) -> str:
        """Hash the executable experiment source, including uncommitted files."""
        project_root = Path(__file__).resolve().parent.parent
        source_files = sorted((project_root / "src").glob("*.py"))
        source_files.append(project_root / "run_experiments.py")
        digest = hashlib.sha256()
        for path in source_files:
            digest.update(str(path.relative_to(project_root)).encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()[:16]

    def _cache_fingerprint(self, config_dict: dict) -> str:
        identity = {
            "version": self.VERSION,
            "source": self.get_source_fingerprint(),
            "config": config_dict,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

    def _get_filename(self, env_id: int, num_classes: int, num_trials: int, config_dict: dict) -> str:
        fingerprint = self._cache_fingerprint(config_dict)
        return os.path.join(
            self.base_dir,
            f"env_{env_id:02d}_M{num_classes}_trials{num_trials}_{fingerprint}_{self.VERSION}.json",
        )

    def exists(self, env_id: int, num_classes: int, num_trials: int, config_dict: dict) -> bool:
        filepath = self._get_filename(env_id, num_classes, num_trials, config_dict)
        return os.path.exists(filepath)

    def save(self, env_id: int, num_classes: int, num_trials: int, env_objects: list, results: dict, config_dict: dict):
        filepath = self._get_filename(env_id, num_classes, num_trials, config_dict)

        # Serialize objects
        serialized_objects = []
        for obj in env_objects:
            serialized_objects.append({
                "id": obj.id,
                "gt_class": obj.gt_class,
                "gt_pose": obj.gt_pose.tolist()
            })

        # Serialize results array to nested lists
        serializable_results = {}
        for mode, data in results.items():
            serializable_results[mode] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}

        payload = {
            "metadata": {
                "env_id": env_id,
                "num_candidate_classes": num_classes,
                "num_trials": num_trials,
                "timestamp": datetime.now().isoformat(),
                "version": self.VERSION,
                "git_revision": self.get_git_revision(),
                "source_fingerprint": self.get_source_fingerprint(),
                "cache_fingerprint": self._cache_fingerprint(config_dict),
                "config": config_dict
            },
            "environment": {
                "objects": serialized_objects
            },
            "results": serializable_results
        }

        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2)

    def load(self, env_id: int, num_classes: int, num_trials: int, config_dict: dict) -> Optional[dict]:
        filepath = self._get_filename(env_id, num_classes, num_trials, config_dict)
        if not os.path.exists(filepath):
            return None

        with open(filepath, "r") as f:
            payload = json.load(f)

        results = {}
        for mode, m_data in payload["results"].items():
            results[mode] = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in payload["results"][mode].items()
            }

        payload["results"] = results
        return payload
