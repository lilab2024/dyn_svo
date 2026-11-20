"""Read YAML settings from config/ and fingerprint them."""
import hashlib

import yaml

from dyn_svo.data.manifest import CONFIG_DIR


def config_path(name):
    return CONFIG_DIR / (name + ".yaml")


def load_config(name):
    """Return the parsed config/<name>.yaml."""
    with open(config_path(name), encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_md5(name):
    """md5 of the raw bytes of config/<name>.yaml (recorded with every result)."""
    return hashlib.md5(config_path(name).read_bytes()).hexdigest()
