"""Check that the exporter cannot silently include local research material."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import prepare_github as exporter


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.source = self.base / "source"
        (self.source / "packaging").mkdir(parents=True)
        (self.source / "src").mkdir()
        (self.source / "src/core.jl").write_text("module Example\nend\n")
        (self.source / "Project.toml").write_text('name = "Example"\n')
        (self.source / "README.md").write_text("# Example\n")
        self.config = {"directories": ["src"], "files": ["Project.toml", "README.md"],
                       "max_file_bytes": 1024, "max_total_bytes": 4096}
        self.write_config()
        context = patch.object(exporter, "SOURCE", self.source)
        context.start()
        self.addCleanup(context.stop)

    def write_config(self):
        (self.source / "packaging/upload.json").write_text(json.dumps(self.config))

    def test_selection_archive_and_hashes(self):
        (self.source / "src/.DS_Store").write_text("local metadata")
        research = self.source / "validation/results"
        research.mkdir(parents=True)
        (research / "private.bin").write_bytes(b"do not upload")
        output = self.base / "upload"
        exporter.export_package(output, archive=True)
        self.assertFalse((output / "validation").exists())
        self.assertFalse((output / "src/.DS_Store").exists())
        report = json.loads((output / "UPLOAD_CONTENTS.json").read_text())
        for item in report["files"]:
            data = (output / item["path"]).read_bytes()
            self.assertEqual(item["bytes"], len(data))
            self.assertEqual(item["sha256"], hashlib.sha256(data).hexdigest())
        with zipfile.ZipFile(self.base / "upload.zip") as archive:
            self.assertEqual(len(archive.namelist()), 4)
            self.assertTrue(all(name.startswith("TensorNetworksJulia/") for name in archive.namelist()))
            self.assertIsNone(archive.testzip())

    def test_existing_target_is_preserved(self):
        output = self.base / "upload"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(ValueError):
            exporter.export_package(output, archive=False)
        self.assertEqual(marker.read_text(), "keep")

    def test_bad_selection_fails_before_output_creation(self):
        (self.source / "README.md").write_text("x" * 2048)
        output = self.base / "upload"
        with self.assertRaises(ValueError):
            exporter.export_package(output, archive=True)
        self.assertFalse(output.exists())
        self.assertFalse((self.base / "upload.zip").exists())

    def test_symlink_and_traversal_are_rejected(self):
        outside = self.base / "outside.txt"
        outside.write_text("outside")
        (self.source / "src/link.txt").symlink_to(outside)
        with self.assertRaises(ValueError):
            exporter.selected_files()
        self.config["files"] = ["../outside.txt"]
        self.config["directories"] = []
        self.write_config()
        with self.assertRaises(ValueError):
            exporter.selected_files()

    def test_machine_path_is_rejected(self):
        (self.source / "README.md").write_text("/" + "Users/example/private/work")
        with self.assertRaises(ValueError):
            exporter.selected_files()

    def test_output_inside_source_is_rejected(self):
        with self.assertRaises(ValueError):
            exporter.export_package(self.source / "upload", archive=False)

    def test_document_override_is_explicit(self):
        (self.source / "packaging/short.md").write_text("Portable overview\n")
        self.config["overrides"] = {"README.md": "packaging/short.md"}
        self.write_config()
        selection = exporter.selected_files()
        self.assertEqual(selection[Path("README.md")].read_text(), "Portable overview\n")


if __name__ == "__main__":
    unittest.main()
