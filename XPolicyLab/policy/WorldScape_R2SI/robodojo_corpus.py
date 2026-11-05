"""Read-only adapters and catalog materialization for the official OSS mirror.

The source corpus is immutable. Descriptors, packages and indexes belong on a
local POSIX filesystem. No HDF5 file or LeRobot video is rewritten.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

DEFAULT_ROOT = Path('/mnt/oss/datasets/robodojo_official')
DEFAULT_WORK = Path('/mnt/workspace/r2si_icl')
SCHEMA = 'r2si-robodojo-episode/1'
FPS_SOURCE = 'https://robodojo-benchmark.com/doc/usage/install-and-download/'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(canonical(value) + b'\n')
    temporary.replace(path)


def source_path(root, relative):
    path = (Path(root) / relative).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('Source path must be a file inside the declared corpus')
    return path


def record_identity(record):
    if record['domain'] == 'real':
        return 'robodojo_' + record['sha256'][:20]
    if record['domain'] != 'simulation':
        raise ValueError('Unknown RoboDojo domain')
    identity = {k: record[k] for k in ('episode_index', 'relative_path', 'sha256',
                                      'parquet_row_start', 'parquet_row_stop')}
    identity['videos'] = {v: {k: item[k] for k in ('relative_path', 'sha256',
                                                  'from_timestamp', 'to_timestamp')}
                          for v, item in record['videos'].items()}
    return 'robodojo_sim_' + hashlib.sha256(canonical(identity)).hexdigest()[:20]


def build_catalog(root=DEFAULT_ROOT, work=DEFAULT_WORK, *, fallback_fps=25.0,
                  fps_source=FPS_SOURCE, endpoint='http://127.0.0.1:8090'):
    """Export all 5,300 episodes without rereading/reencoding 393 GB of media.

Source hashes are checked when selected media is compiled, not trusted as an
mtime-only package identity. The completed mirror's catalog hash is checked here.
"""
    root, work = Path(root).resolve(strict=True), Path(work).resolve()
    if work.is_relative_to(root):
        raise ValueError('Writable ICL workspace must be outside the source corpus')
    if not math.isfinite(fallback_fps) or fallback_fps <= 0 or not fps_source:
        raise ValueError('An explicit positive fallback FPS and its source are required')
    completed = json.loads((root / 'COMPLETED.json').read_text())
    if completed.get('state') != 'complete':
        raise ValueError('The official mirror has not completed verification')
    records_path = root / 'rag/episodes.jsonl'
    expected = completed['rag_artifacts_sha256']['rag/episodes.jsonl']
    if digest(records_path) != expected:
        raise ValueError('Mirror episode catalog SHA256 mismatch')
    manifest = json.loads((root / 'manifest.json').read_text())
    files = {f['path']: f for f in manifest['files']}
    rows, seen, checked_files = [], set(), set()
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if len(records) != completed['total_episodes']:
        raise ValueError('Mirror episode count mismatch')
    for record in records:
        eid = record_identity(record)
        if eid in seen:
            raise ValueError('Duplicate episode identity')
        seen.add(eid)
        references = {record['relative_path']: record['sha256']}
        if record['domain'] == 'simulation':
            references.update({v['relative_path']: v['sha256'] for v in record['videos'].values()})
        for relative, sha in references.items():
            f = files[relative]
            if sha != f['sha256']:
                raise ValueError('Episode and source manifest hashes disagree')
            if relative not in checked_files:
                if source_path(root, relative).stat().st_size != f['size']:
                    raise ValueError('Source size differs from the completed manifest')
                checked_files.add(relative)
        descriptor = work / 'episodes' / (eid + '.json')
        write_json(descriptor, dict(schema=SCHEMA, root=str(root), episode_id=eid,
                                   record=record, files=references,
                                   fallback_fps=fallback_fps, fps_source=fps_source))
        rows.append(dict(source='robodojo', episode_id=eid, task=record['instruction'],
                         instructions=[record['instruction']], path=str(descriptor),
                         domain=record['domain'], embodiment=record['embodiment']))
    work.mkdir(parents=True, exist_ok=True)
    temporary = work / 'catalog.jsonl.tmp'
    temporary.write_bytes(b''.join(canonical(row) + b'\n' for row in rows))
    temporary.replace(work / 'catalog.jsonl')
    # JSON quoted strings are valid TOML strings for these filesystem/HTTP paths.
    q = lambda s: json.dumps(str(s))
    config = ('schema_version = 1\n[paths]\n'
              f'catalog = {q(work / "catalog.jsonl")}\npackages = {q(work / "packages")}\n'
              f'contexts = {q(work / "contexts")}\n[retrieval]\nbackend = "http"\n'
              f'endpoint = {q(endpoint)}\n[compiler]\nrecipe = "s1"\nimage_budget = 16\n'
              '[context]\nmax_demo_images = 64\nmax_demo_text_chars = 32000\n'
              f'[data]\nofficial_root = {q(root)}\n')
    (work / 'r2si.toml').write_text(config, encoding='utf-8')
    summary = dict(episodes=len(rows), source_root=str(root), catalog=str(work/'catalog.jsonl'),
                   config=str(work/'r2si.toml'), missing_fps=sum(r.get('fps') is None for r in records),
                   source_catalog_sha256=expected, fallback_fps=fallback_fps, fps_source=fps_source)
    write_json(work / 'corpus.json', summary)
    return summary


def install_readers(official_root=None):
    """Install process-local readers; preserve legacy HDF5 and DROID behavior."""
    from icl_engine import sources
    from PIL import Image
    from XPolicyLab.utils.process_data import decode_image_bit
    base = getattr(sources, '_r2si_original_robodojo', sources.RoboDojoEpisode)
    sources._r2si_original_robodojo = base

    class XPolicyLabEpisode(base):
        def get_frame(self, index, view):
            self._check(index, view)
            return Image.fromarray(decode_image_bit(self._file[f'vision/{view}/colors'][int(index)]))

    class OfficialEpisode(XPolicyLabEpisode):
        def __init__(self, path):
            self.path = Path(path).resolve(strict=True)
            doc = json.loads(self.path.read_text())
            if doc.get('schema') != SCHEMA or official_root is None:
                raise ValueError('Official descriptors require an explicitly configured corpus')
            self.root = Path(doc['root']).resolve(strict=True)
            if self.root != Path(official_root).resolve(strict=True):
                raise ValueError('Descriptor corpus does not match the configured corpus')
            self.record, self.files = doc['record'], doc['files']
            self.episode_id = record_identity(self.record)
            if self.episode_id != doc['episode_id']:
                raise ValueError('Descriptor episode identity mismatch')
            self.frame_count = int(self.record['frames'])
            self.task = self.record['instruction']
            self.instructions = [self.task]
            missing_fps = self.record.get('fps') is None
            if missing_fps and (self.record['domain'] != 'real' or not doc.get('fps_source')):
                raise ValueError('A real-data fallback rate requires documented provenance')
            self.fps = float(doc['fallback_fps'] if missing_fps else self.record['fps'])
            if self.frame_count < 1 or not self.task or not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError('Invalid episode frame count, instruction or rate')
            self._file, self._containers, self._gripper_keys = None, {}, []
            self._sim = self.record['domain'] == 'simulation'
            references = {self.record['relative_path']: self.record['sha256']}
            if self._sim:
                references.update({v['relative_path']: v['sha256'] for v in self.record['videos'].values()})
            if references != self.files:
                raise ValueError('Descriptor source files disagree with the episode record')
            self._paths = {name: source_path(self.root, name) for name in self.files}
            if self._sim:
                self.views = sorted(self.record['videos'])
                if self.record['parquet_row_stop'] - self.record['parquet_row_start'] != self.frame_count:
                    raise ValueError('LeRobot episode row range and frame count disagree')
                calibration = False
            else:
                import h5py
                self._file = h5py.File(self._paths[self.record['relative_path']], 'r')
                try:
                    self.views = sorted(v for v in self._file['vision'] if 'colors' in self._file['vision'][v])
                    if not self.views or any(len(self._file[f'vision/{v}/colors']) != self.frame_count for v in self.views):
                        raise ValueError('HDF5 and descriptor frame counts disagree')
                    actual = self._file['instruction'][()]
                    if isinstance(actual, bytes):
                        actual = actual.decode('utf-8')
                    if str(actual) != self.task:
                        raise ValueError('HDF5 and descriptor instructions disagree')
                    if 'additional_info/frequency' in self._file:
                        if float(self._file['additional_info/frequency'][()]) != self.fps:
                            raise ValueError('HDF5 and descriptor rates disagree')
                    self._gripper_keys = [n for n in self._file.get('state', {}) if n.endswith('_ee_joint_states')]
                    if any(len(self._file['state'][n]) != self.frame_count for n in self._gripper_keys):
                        raise ValueError('Gripper state and images have inconsistent lengths')
                    calibration = all('intrinsic_matrix' in self._file['vision'][v] and
                                      'extrinsic_matrix' in self._file['vision'][v] for v in self.views)
                except BaseException:
                    self._file.close()
                    raise
            fallback = self.record.get('fps') is None
            self.provenance = dict(source_files=[str(p) for p in self._paths.values()],
                source_camera_names_preserved=True, success_label='unknown', success_independently_verified=False,
                time_basis=('dataset-declared fallback rate; no per-frame clock verification' if fallback
                            else 'source metadata frame rate'), fps_source=doc['fps_source'] if fallback else 'source metadata',
                fps_fallback_used=fallback, adapter_sha256=digest(__file__), domain=self.record['domain'])
            self.capabilities = dict(images=True, state=not self._sim and 'state' in self._file,
                actions=not self._sim and 'action' in self._file,
                calibration=calibration, geometry_verified=False, gripper_semantics_verified=False)

        def fingerprint(self):
            if not hasattr(self, '_fingerprint'):
                for relative, expected in self.files.items():
                    if digest(self._paths[relative]) != expected:
                        raise ValueError('Official source content SHA256 mismatch: ' + relative)
                material = dict(episode=self.episode_id, files=self.files, fps=self.fps,
                                record=self.record, adapter=self.provenance['adapter_sha256'])
                self._fingerprint = hashlib.sha256(canonical(material)).hexdigest()
            return self._fingerprint

        def get_frame(self, index, view):
            self._check(index, view)
            if not self._sim:
                return super().get_frame(index, view)
            import av
            video = self.record['videos'][view]
            target = float(video['from_timestamp']) + int(index) / self.fps
            if target > float(video['to_timestamp']) + 1 / self.fps:
                raise ValueError('Requested frame crosses the official episode video interval')
            if view not in self._containers:
                self._containers[view] = av.open(str(self._paths[video['relative_path']]))
            container = self._containers[view]
            stream = container.streams.video[0]
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is not None and float(frame.pts * stream.time_base) >= target - 0.5 / self.fps:
                    if float(frame.pts * stream.time_base) > target + 1 / self.fps:
                        raise ValueError('Decoded video frame is outside the requested timestamp')
                    # LeRobot MP4 is a standard video, not an XPolicyLab JPEG buffer.
                    return frame.to_image()
            raise ValueError('Requested LeRobot video frame could not be decoded')

        def gripper_series(self):
            return {} if self._sim else super().gripper_series()

        def robot_state(self):
            return {} if self._sim else super().robot_state()

        def calibration(self, index, view):
            if self._sim:
                raise ValueError('LeRobot export does not provide verified camera calibration')
            return super().calibration(index, view)

        def close(self):
            if self._file is not None:
                self._file.close()
            for container in self._containers.values():
                container.close()

    def reader(path):
        if Path(path).suffix == '.json':
            return OfficialEpisode(path)
        return XPolicyLabEpisode(path)

    sources.RoboDojoEpisode = reader
    return reader
