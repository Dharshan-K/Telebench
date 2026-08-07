from .loader import DatasetLoader

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac"}


class LocalLoader(DatasetLoader):
    def __iter__(self):
        for file in super().__iter__():
            if file.suffix.lower() in AUDIO_SUFFIXES:
                yield file
