from telebench.dataloader.localloader import LocalLoader


def test_yields_only_audio_files(tmp_path):
    (tmp_path / "a.wav").write_text("x")
    (tmp_path / "b.mp3").write_text("x")
    (tmp_path / "c.flac").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    names = sorted(f.name for f in LocalLoader(tmp_path))
    assert names == ["a.wav", "b.mp3", "c.flac"]


def test_skips_subdirectories(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "inner.wav").write_text("x")
    (tmp_path / "top.wav").write_text("x")
    names = [f.name for f in LocalLoader(tmp_path)]
    assert names == ["top.wav"]


def test_expands_user_home(tmp_path, make_wav, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    make_wav(home / "audio.wav")
    monkeypatch.setenv("HOME", str(home))
    files = [f.name for f in LocalLoader("~/")]
    assert files == ["audio.wav"]