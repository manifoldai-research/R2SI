from unittest.mock import patch
import pytest
from rsi.storage import Workspace
from rsi.storage_preflight import check_storage

def test_real_filesystem_and_no_hardlink_requirement(tmp_path):
    ws=Workspace(tmp_path)
    with patch("os.link",side_effect=OSError(38,"unsupported")):
        check_storage(ws)
    assert ws.path("storage-capability.json").exists()
    assert not list(ws.path(".storage-probes").iterdir())

def test_unsupported_mount_refuses_before_dispatch(tmp_path):
    from rsi.episode_jobs import EpisodeJobs
    ws=Workspace(tmp_path)
    with patch("rsi.storage_preflight.os.rename",side_effect=OSError(38,"unsupported")):
        with pytest.raises(RuntimeError,match="POSIX"):
            EpisodeJobs(ws,{}).evaluate("release",[])
    assert not ws.path("episode-jobs/pending").exists()
