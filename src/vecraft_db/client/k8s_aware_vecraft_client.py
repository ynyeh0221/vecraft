import os
from pathlib import Path

import yaml

from src.vecraft_db.client.vecraft_client import VecraftClient


class K8sAwareVecraftClient(VecraftClient):
    def __init__(self, root: str, **kwargs):
        # Ensure dir exists
        Path(root).mkdir(parents=True, exist_ok=True)

        # Support get config from config files
        config_file = os.getenv("CONFIG_FILE")
        if config_file and os.path.exists(config_file):
            with open(config_file) as f:
                config = yaml.safe_load(f)
                vector_cfg = config.get('vector_index', {}) or {}
                # Merge vector_index settings into kwargs
                kwargs.update(vector_cfg)

        super().__init__(root, **kwargs)