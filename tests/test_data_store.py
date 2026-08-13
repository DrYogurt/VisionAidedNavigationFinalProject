import json
import os
import tempfile
import unittest

from src.data_store import VersionedDataStore


class TestVersionedDataStore(unittest.TestCase):
    def test_cache_identity_changes_with_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VersionedDataStore(directory)
            base = {
                "num_steps": 10,
                "num_samples": 1000,
                "modes": ["viewpoint_dependent", "geometric_only"],
            }
            changed = dict(base, num_steps=11)
            self.assertNotEqual(
                store._get_filename(0, 2, 50, base),
                store._get_filename(0, 2, 50, changed),
            )

    def test_source_fingerprint_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VersionedDataStore(directory)
            self.assertEqual(store.get_source_fingerprint(), store.get_source_fingerprint())
            self.assertEqual(len(store.get_source_fingerprint()), 16)

    def test_allowlisted_checkpoint_with_exact_config_is_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VersionedDataStore(directory)
            config = {"num_steps": 10, "modes": ["geometric_only"]}
            old_path = os.path.join(
                directory, "env_12_M2_trials50_old_v8.0.0.json"
            )
            payload = {
                "metadata": {
                    "env_id": 12,
                    "num_candidate_classes": 2,
                    "num_trials": 50,
                    "version": store.VERSION,
                    "source_fingerprint": "bef21a006773792f",
                    "config": config,
                },
                "results": {
                    "geometric_only": {"da_entropy": [[0.0]]},
                },
            }
            with open(old_path, "w") as handle:
                json.dump(payload, handle)

            self.assertTrue(store.exists(12, 2, 50, config))
            loaded = store.load(12, 2, 50, config)
            self.assertEqual(loaded["results"]["geometric_only"]["da_entropy"].shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
