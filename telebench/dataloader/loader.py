from abc import ABC
from pathlib import Path


class DatasetLoader(ABC):
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def __iter__(self):
        for file in self.path.iterdir():
            if file.is_file():
                yield file
