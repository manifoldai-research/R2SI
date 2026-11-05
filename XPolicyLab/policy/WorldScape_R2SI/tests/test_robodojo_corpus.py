"""Real codecs/engine, tiny official-layout fixtures; no CUDA/model downloads."""
import base64
import io
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import av
import h5py
import numpy as np
from PIL import Image
import pytest

from XPolicyLab.policy.WorldScape_R2SI import robodojo_corpus as corpus
from XPolicyLab.policy.WorldScape_R2SI import robodojo_retrieval as retrieval
from XPolicyLab.policy.WorldScape_R2SI.icl_worker import prepare
from XPolicyLab.utils.process_data import encode_image_bit


@pytest.fixture
def mirror(tmp_path):
    root, work = tmp_path/'raw', tmp_path/'work'
    (root/'rag').mkdir(parents=True)
    records = []
    for missing in (False, True):
        path = root/f'real-{missing}.hdf5'
        red = np.full((32, 32, 3), [220, 10, 30], dtype=np.uint8)
        legacy = io.BytesIO()
        Image.new('RGB', (32, 32), (30, 10, 220)).save(legacy, format='JPEG', quality=100)
        with h5py.File(path, 'w') as f:
            f['instruction'] = np.bytes_('Move the object')
            if not missing:
                f['additional_info/frequency'] = 10
            frames = f.create_dataset('vision/cam_head/colors', (2,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
            frames[0] = np.frombuffer(legacy.getvalue(), dtype=np.uint8)
            frames[1] = np.frombuffer(encode_image_bit(red), dtype=np.uint8)
        records.append(dict(domain='real', relative_path=path.name, sha256=corpus.digest(path),
            instruction='Move the object', embodiment='fixture', fps=None if missing else 10, frames=2))
    path = root/'combined.mp4'
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('mpeg4', rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 64, 'yuv420p'
        for i in range(40):
            rgb = np.full((64, 64, 3), [220, 10, 30] if i < 20 else [20, 10, 230], dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    # Parquet is provenance only for image-only S1; the reader must not fabricate poses/actions.
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({'frame_index': range(40)}), root/'data.parquet')
    records.append(dict(domain='simulation', episode_index=1, relative_path='data.parquet',
        sha256=corpus.digest(root/'data.parquet'), parquet_row_start=20, parquet_row_stop=40,
        instruction='Move the object', embodiment='fixture', frames=20, fps=10,
        videos={'observation.images.cam_high': dict(relative_path=path.name,
            sha256=corpus.digest(path), from_timestamp=2.0, to_timestamp=4.0)}))
    (root/'rag/episodes.jsonl').write_bytes(b''.join(corpus.canonical(r)+b'\n' for r in records))
    corpus.write_json(root/'manifest.json', {'files': [dict(path=p.name, size=p.stat().st_size,
        sha256=corpus.digest(p)) for p in root.iterdir() if p.is_file()]})
    corpus.write_json(root/'COMPLETED.json', dict(state='complete', total_episodes=3,
        rag_artifacts_sha256={'rag/episodes.jsonl': corpus.digest(root/'rag/episodes.jsonl')}))
    corpus.build_catalog(root, work)
    yield root, work, records
    corpus.install_readers()  # Leave plain HDF5 behavior available to other test modules.


def reader(mirror, i):
    root, work, records = mirror
    return corpus.install_readers(root)(work/'episodes'/(corpus.record_identity(records[i])+'.json'))


def test_legacy_and_marked_rgb_missing_fps_read_only(mirror):
    root, _, records = mirror
    for i in (0, 1):
        path = root/records[i]['relative_path']
        before = corpus.digest(path)
        with reader(mirror, i) as episode:
            assert episode.source == 'robodojo'
            assert episode.fps == (10 if i == 0 else 25)
            assert episode.provenance['fps_fallback_used'] == (i == 1)
            if i == 1:
                assert episode.provenance['fps_source'] == corpus.FPS_SOURCE
            for index in (0, 1):
                assert np.max(np.abs(np.array(episode.get_frame(index, 'cam_head')).mean((0, 1))-[220, 10, 30])) < 5
            assert len(episode.fingerprint()) == 64
        assert corpus.digest(path) == before


def test_lerobot_shared_mp4_seek_stays_in_second_episode(mirror):
    with reader(mirror, 2) as episode:
        view = 'observation.images.cam_high'
        # Non-monotonic access includes the boundary and last frame of the second episode.
        for index in (19, 0, 10, 1):
            assert np.max(np.abs(np.array(episode.get_frame(index, view)).mean((0, 1))-[20, 10, 230])) < 6
        assert episode.robot_state() == {} and episode.gripper_series() == {}
        assert not episode.capabilities['calibration'] and not episode.capabilities['actions']
        with pytest.raises((ValueError, IndexError)):
            episode.get_frame(20, view)


@pytest.mark.parametrize('field,value', [('fps', 0), ('frames', 0), ('domain', 'unknown')])
def test_invalid_episode_metadata_is_rejected(mirror, field, value):
    _, work, records = mirror
    path = work/'episodes'/(corpus.record_identity(records[1])+'.json')
    doc = json.loads(path.read_text())
    doc['record'][field] = value
    corpus.write_json(path, doc)
    with pytest.raises(ValueError):
        reader(mirror, 1)


def test_source_identity_and_catalog_integrity_fail_closed(mirror):
    root, work, records = mirror
    path = work/'episodes'/(corpus.record_identity(records[1])+'.json')
    doc = json.loads(path.read_text())
    doc['episode_id'] = 'wrong'
    corpus.write_json(path, doc)
    with pytest.raises(ValueError, match='identity'):
        reader(mirror, 1)
    with (root/'rag/episodes.jsonl').open('ab') as stream:
        stream.write(b'\n')
    with pytest.raises(ValueError, match='SHA256'):
        corpus.build_catalog(root, work)


def test_source_content_rehashed_before_compilation(mirror):
    root, _, records = mirror
    with (root/records[0]['relative_path']).open('ab') as stream:
        stream.write(b'changed')  # HDF5 remains readable, but its content identity changed.
    with reader(mirror, 0) as episode, pytest.raises(ValueError, match='SHA256'):
        episode.fingerprint()


def test_paths_and_declared_corpus_are_enforced(mirror, tmp_path):
    root, work, records = mirror
    outside = tmp_path/'outside'
    outside.write_text('outside')
    with pytest.raises(ValueError):
        corpus.source_path(root, '../outside')
    with pytest.raises(ValueError):
        corpus.build_catalog(root, root/'cache')
    path = work/'episodes'/(corpus.record_identity(records[0])+'.json')
    with pytest.raises(ValueError, match='configured corpus'):
        corpus.install_readers()(path)


@pytest.mark.parametrize('selected', [1, 2])
def test_real_engine_worker_compiles_both_official_formats(mirror, selected):
    _, work, records = mirror
    eid = corpus.record_identity(records[selected])
    rows = [json.loads(line) for line in (work/'catalog.jsonl').read_text().splitlines()]
    (work/'catalog.jsonl').write_bytes(corpus.canonical(next(r for r in rows if r['episode_id'] == eid))+b'\n')
    config = work/'r2si.toml'
    config.write_text(config.read_text().replace('backend = "http"', 'backend = "lexical"'))
    with reader(mirror, selected) as episode:
        query = work/'query.png'
        episode.get_frame(0, retrieval.head_view(episode)).save(query)
    result = prepare(config, dict(instruction='Move the object', image=str(query), max_images=64, max_text_chars=32000))
    assert result['status'] == 'ready'
    assert result['metadata']['episode_id'] == eid
    assert result['metadata']['recipe'] == 's1'
    assert not result['metadata']['command_replay_verified']


class FixtureEncoder:
    signature = {'fixture_only': True}

    def encode(self, records):
        rows = []
        for r in records:
            vector = [1., 0., 0.] if 'text' in r else np.array(r['image']).mean((0, 1)).tolist()
            rows.append(vector)
        matrix = np.array(rows, dtype=np.float32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_index_scene_ranking_exclusions_http_and_stale_detection(mirror):
    _, work, records = mirror
    encoder = FixtureEncoder()
    assert retrieval.build_index(work, encoder, batch_size=2)['episodes'] == 3
    searcher = retrieval.Searcher(work, encoder)
    query = Image.new('RGB', (64, 64), (20, 10, 230))
    eid = corpus.record_identity(records[2])
    result = searcher.search('Move the object', query)
    assert result['results'][0]['episode_id'] == eid
    assert result['scene_used_for_ranking']
    assert eid not in [r['episode_id'] for r in searcher.search('Move the object', query, 3, ['robodojo:'+eid])['results']]
    with pytest.raises(ValueError):
        searcher.search('Move the object', None)
    with retrieval.make_server(searcher, 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            buf = io.BytesIO()
            query.save(buf, format='PNG')
            payload = dict(task='Move the object', image_base64=base64.b64encode(buf.getvalue()).decode())
            url = f'http://127.0.0.1:{server.server_port}/search'
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(urllib.request.Request(url, data=corpus.canonical(payload))) as response:
                assert json.load(response)['results'][0]['episode_id'] == eid
            with pytest.raises(urllib.error.HTTPError) as error:
                opener.open(urllib.request.Request(url, data=b'{}'))
            assert error.value.code == 400
        finally:
            server.shutdown()
            thread.join()
    # Descriptor changes must invalidate an already-built index even if the catalog is unchanged.
    path = work/'episodes'/(eid+'.json')
    with path.open('ab') as stream:
        stream.write(b'\n')
    with pytest.raises(ValueError, match='stale'):
        retrieval.Searcher(work, encoder)


def test_vector_corruption_is_rejected(mirror):
    _, work, _ = mirror
    encoder = FixtureEncoder()
    retrieval.build_index(work, encoder)
    meta = json.loads((work/'index/ready.json').read_text())
    with (work/'index'/meta['vectors']).open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        retrieval.Searcher(work, encoder)
