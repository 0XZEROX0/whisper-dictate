import subprocess

from whisper_dictate import updater


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t",
                    "-c", "init.defaultBranch=main", *args], check=True, capture_output=True)


def test_check_detects_new_commits(tmp_path, monkeypatch):
    origin, work, clone = tmp_path / "origin.git", tmp_path / "work", tmp_path / "clone"
    git(tmp_path, "init", "--bare", str(origin))
    git(tmp_path, "init", str(work))
    (work / "a").write_text("1")
    git(work, "add", "a")
    git(work, "commit", "-m", "first")
    git(work, "push", str(origin), "HEAD:main")
    git(tmp_path, "clone", str(origin), str(clone))
    monkeypatch.setattr(updater, "repo_dir", lambda: clone)

    info = updater.check()
    assert not info.error and not info.available

    (work / "a").write_text("2")
    git(work, "commit", "-am", "Add settings window")
    git(work, "push", str(origin), "HEAD:main")
    info = updater.check()
    assert info.available and info.changes == ["Add settings window"]


def test_check_without_repo(monkeypatch):
    monkeypatch.setattr(updater, "repo_dir", lambda: None)
    assert updater.check().error
