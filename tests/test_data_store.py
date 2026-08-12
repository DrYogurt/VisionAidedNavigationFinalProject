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


if __name__ == "__main__":
    unittest.main()
