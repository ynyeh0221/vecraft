import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from vecraft_db.client.k8s_aware_vecraft_client import K8sAwareVecraftClient
from vecraft_db.client.vecraft_client import VecraftClient


class TestK8sAwareVecraftClient(unittest.TestCase):
    def setUp(self):
        # Create a temporary root path
        self.temp_root = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up
        shutil.rmtree(self.temp_root)

    def test_directory_creation(self):
        """If the target root doesn't exist, it should be created."""
        new_dir = os.path.join(self.temp_root, "data_dir")
        self.assertFalse(os.path.exists(new_dir))

        # Patch VecraftClient.__init__ so we don't actually initialize the real client
        with patch.object(VecraftClient, "__init__", return_value=None) as mock_super_init:
            K8sAwareVecraftClient(new_dir, foo="bar")

            # The directory should now exist
            self.assertTrue(os.path.isdir(new_dir))

            # And VecraftClient.__init__ should have been called with the unmodified kwargs
            mock_super_init.assert_called_once_with(new_dir, foo="bar")

    def test_config_file_applied(self):
        """When CONFIG_FILE points to a YAML with vector_index, its keys join the init kwargs."""
        # Write a small config.yaml under temp_root
        config_path = os.path.join(self.temp_root, "config.yaml")
        yaml_text = (
            "vector_index:\n"
            "  dim: 64\n"
            "  metric: cosine\n"
        )
        with open(config_path, "w") as f:
            f.write(yaml_text)

        # Inject the env var
        with patch.dict(os.environ, {"CONFIG_FILE": config_path}):
            with patch.object(VecraftClient, "__init__", return_value=None) as mock_super_init:
                K8sAwareVecraftClient(self.temp_root, batch_size=32)

                # Expect batch_size=32 plus dim & metric from the YAML
                expected_kwargs = {
                    "batch_size": 32,
                    "dim": 64,
                    "metric": "cosine"
                }
                mock_super_init.assert_called_once_with(self.temp_root, **expected_kwargs)

    def test_missing_config_file_ignored(self):
        """If CONFIG_FILE is set but the file doesn't exist, fall back to the passed kwargs."""
        fake_path = os.path.join(self.temp_root, "does_not_exist.yaml")
        with patch.dict(os.environ, {"CONFIG_FILE": fake_path}):
            with patch.object(VecraftClient, "__init__", return_value=None) as mock_super_init:
                K8sAwareVecraftClient(self.temp_root, alpha=0.1)

                # Should just forward alpha=0.1
                mock_super_init.assert_called_once_with(self.temp_root, alpha=0.1)


if __name__ == "__main__":
    unittest.main()
