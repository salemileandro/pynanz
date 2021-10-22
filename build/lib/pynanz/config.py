import os
from pathlib import Path


class Config:
    HOME_DIRECTORY = str(Path.home())
    CACHE_DIRECTORY = os.path.join(HOME_DIRECTORY, ".cache/pynanz")
    CONFIG_DIRECTORY = os.path.join(HOME_DIRECTORY, ".config/pynanz")

    def __init__(self):
        raise NotImplementedError("Config class cannot be used to create instances (static class only)")

