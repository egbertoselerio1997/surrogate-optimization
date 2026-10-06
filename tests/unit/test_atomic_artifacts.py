"""Interrupted publication preserves the previous file and cleans temporary files."""

from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from surrogate_optimization.runtime import artifacts


class AtomicPublicationTests(unittest.TestCase):
    def test_transient_windows_lock_retries_before_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            artifacts.atomic_json(path, {"value": "old"})
            replace = os.replace
            failure = PermissionError("sharing violation")
            failure.winerror = 32
            with (
                patch.object(
                    artifacts.os, "replace", side_effect=[failure, None]
                ) as mocked,
                patch.object(artifacts, "sleep"),
            ):

                def publish(source, destination):
                    if mocked.call_count == 1:
                        raise failure
                    return replace(source, destination)

                mocked.side_effect = publish
                artifacts.atomic_json(path, {"value": "new"})
                self.assertEqual(mocked.call_count, 2)
            self.assertEqual(json.loads(path.read_text()), {"value": "new"})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_persistent_lock_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            artifacts.atomic_json(path, {"value": "old"})
            failure = PermissionError("sharing violation")
            failure.winerror = 32
            with (
                patch.object(artifacts.os, "replace", side_effect=failure),
                patch.object(artifacts, "sleep"),
            ):
                with self.assertRaises(PermissionError):
                    artifacts.atomic_json(path, {"value": "new"})
            self.assertEqual(json.loads(path.read_text()), {"value": "old"})
            self.assertEqual(list(Path(directory).iterdir()), [path])
