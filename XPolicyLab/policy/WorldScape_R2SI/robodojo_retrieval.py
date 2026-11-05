"""RoboDojo-native scene/task retrieval through the ICL Engine HTTP contract.

This is a new, unvalidated ranking method, not the engine's DROID v5/v6/v7.
Model inference is imported lazily; catalog export does not require CUDA.
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import hashlib
import json
from pathlib import Path
import sys

from .robodojo_corpus import (DEFAULT_ROOT, DEFAULT_WORK, build_catalog, canonical,
                             digest, install_readers, write_json)

METHOD = 'robodojo_qwen3vl_v1'


class Encoder:
    """Use the pinned engine's Qwen implementation with strict image preprocessing."""

    def __init__(self, model, device=0):
        import icl_engine
        import torch
        from XPolicyLab.utils import process_data
        model = Path(model).resolve(strict=True)
        vendor = Path(icl_engine.__file__).resolve().parents[2] / 'legacy/retrieval/vendor'
        if not (vendor / 'qwen3_vl_embedding.py').is_file():
            raise ValueError('Install the complete editable ICL Engine 0.2.0 checkout')
        if not torch.cuda.is_available() or device >= torch.cuda.device_count() or device < 0:
            raise ValueError('Select an available CUDA device for Qwen retrieval')
        sys.path.insert(0, str(vendor))
        from qwen3_vl_embedding import Qwen3VLEmbedder
        weights = sorted(model.glob('*.safetensors'))
        if not weights:
            raise ValueError('A local Qwen3-VL-Embedding-2B snapshot is required')
        assets = sorted(set(weights + list(model.glob('*.json')) + list(model.glob('*.jinja'))
                            + list(model.glob('*.txt'))))
        self.signature = dict(model_files={p.name: digest(p) for p in assets},
            vendor_sha256=digest(vendor/'qwen3_vl_embedding.py'),
            decoder_sha256=digest(process_data.__file__),
            max_pixels=256 * 1024, max_length=8192, dtype='bfloat16', attention='sdpa')
        with torch.cuda.device(device):
            self.backend = Qwen3VLEmbedder(str(model), max_length=8192,
                max_pixels=self.signature['max_pixels'], torch_dtype=torch.bfloat16,
                attn_implementation='sdpa', local_files_only=True)
        self.backend.model.config.use_cache = False

    def encode(self, records):
        import torch
        from qwen_vl_utils import process_vision_info
        with torch.inference_mode():
            conversations = [self.backend.format_model_input(**r) for r in records]
            text = self.backend.processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=False)
            # Do not use vendor.process(): it catches image failures and embeds NULL.
            images, videos, kwargs = process_vision_info(conversations, image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True)
            if videos is not None:
                raise ValueError('This retrieval method accepts still images only')
            inputs = self.backend.processor(text=text, images=images, padding=True,
                truncation=False, do_resize=False, return_tensors='pt', **kwargs)
            if inputs['input_ids'].shape[1] > self.backend.max_length:
                raise ValueError('Embedding input exceeds the model token budget')
            outputs = self.backend.forward(inputs.to(self.backend.model.device))
            vectors = self.backend._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])
            return torch.nn.functional.normalize(vectors.float(), dim=-1).cpu().numpy()


def catalog_identity(work):
    from icl_engine.retrieval import load_catalog
    from XPolicyLab.utils import process_data
    from . import robodojo_corpus
    work = Path(work)
    rows = load_catalog(work/'catalog.jsonl')
    if not rows or any(r['source'] != 'robodojo' for r in rows):
        raise ValueError('A nonempty RoboDojo catalog is required')
    return rows, dict(catalog_sha256=digest(work/'catalog.jsonl'),
        descriptors_sha256=hashlib.sha256(canonical(
            [digest(r['path']) for r in rows])).hexdigest(),
        adapter_sha256=digest(robodojo_corpus.__file__),
        decoder_sha256=digest(process_data.__file__),
        retrieval_sha256=digest(__file__))


def head_view(episode):
    for name in ('cam_head', 'observation.images.cam_high'):
        if name in episode.views:
            return name
    raise ValueError('Official episode has no declared head camera')


def _save_vectors(path, **arrays):
    import numpy as np
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as out:
        np.savez_compressed(out, **arrays)
    temporary.replace(path)


def build_index(work, encoder, *, batch_size=8):
    """Resumable first-head-frame gallery. Never silently skip an episode."""
    import numpy as np
    from icl_engine.sources import open_episode
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    work = Path(work)
    rows, corpus = catalog_identity(work)
    root = json.loads((work/'corpus.json').read_text())['source_root']
    install_readers(root)
    signature = dict(method=METHOD, corpus=corpus, encoder=encoder.signature,
                     gallery_frame=0, gallery_view='head', batch_size=batch_size)
    key = hashlib.sha256(canonical(signature)).hexdigest()
    staging = work / 'index' / ('build-' + key)
    staging.mkdir(parents=True, exist_ok=True)
    tasks = sorted({r['task'] for r in rows})
    text_path = staging / 'tasks.npz'
    if not text_path.exists():
        text = np.concatenate([encoder.encode([{'text': t} for t in tasks[i:i+batch_size]])
                               for i in range(0, len(tasks), batch_size)])
        _save_vectors(text_path, vectors=text)
    with np.load(text_path, allow_pickle=False) as data:
        text = data['vectors']
    chunks, image_hashes = [], []
    for start in range(0, len(rows), batch_size):
        path = staging / f'{start:07d}.npz'
        if not path.exists():
            images, hashes = [], []
            for row in rows[start:start+batch_size]:
                with open_episode(row['source'], row['path']) as episode:
                    if episode.episode_id != row['episode_id'] or episode.task != row['task']:
                        raise ValueError('Reader and retrieval catalog identities disagree')
                    frame = episode.get_frame(0, head_view(episode))
                    hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
                    images.append({'image': frame})
            _save_vectors(path, vectors=encoder.encode(images), image_sha256=np.array(hashes))
        with np.load(path, allow_pickle=False) as data:
            chunks.append(data['vectors'])
            image_hashes.extend(data['image_sha256'].tolist())
        print(json.dumps({'indexed': min(start+batch_size, len(rows)), 'total': len(rows)}), flush=True)
    images = np.concatenate(chunks)
    tasks_by_id = {t: i for i, t in enumerate(tasks)}
    text = text[[tasks_by_id[r['task']] for r in rows]]
    if text.shape != images.shape or len(images) != len(rows):
        raise ValueError('Incomplete retrieval index')
    for matrix in (text, images):
        if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
            raise ValueError('Invalid embedding vectors')
    vector_path = work/'index'/f'{key}.npz'
    _save_vectors(vector_path, text=text, image=images)
    metadata = dict(signature=signature, vectors=vector_path.name, sha256=digest(vector_path),
        episodes=len(rows), tasks=len(tasks), dimensions=images.shape[1],
        episode_ids=[r['episode_id'] for r in rows], image_sha256=image_hashes,
        source_verification='completed mirror manifest; selected package sources are rehashed')
    write_json(work/'index'/'ready.json', metadata)
    return {k: metadata[k] for k in ('episodes', 'tasks', 'dimensions', 'sha256')}


class Searcher:
    def __init__(self, work, encoder, *, text_weight=0.8):
        import numpy as np
        if not 0 < text_weight < 1:
            raise ValueError('Both task and image must have nonzero ranking weight')
        self.encoder, self.text_weight = encoder, text_weight
        work = Path(work)
        self.rows, corpus = catalog_identity(work)
        meta = json.loads((work/'index/ready.json').read_text())
        signature = meta['signature']
        if (signature['method'] != METHOD or signature['corpus'] != corpus
                or signature['encoder'] != encoder.signature
                or meta['episode_ids'] != [r['episode_id'] for r in self.rows]):
            raise ValueError('Retrieval index is stale; rebuild for this corpus/model/adapter')
        path = work/'index'/meta['vectors']
        if path.parent.resolve() != (work/'index').resolve() or digest(path) != meta['sha256']:
            raise ValueError('Retrieval vector file integrity check failed')
        with np.load(path, allow_pickle=False) as data:
            self.text, self.image = data['text'], data['image']
        expected = (len(self.rows), meta['dimensions'])
        for matrix in (self.text, self.image):
            if matrix.shape != expected or not np.isfinite(matrix).all():
                raise ValueError('Invalid retrieval matrix')

    def search(self, task, image, top_k=1, exclude_ids=()):
        import numpy as np
        if not isinstance(task, str) or not task.strip() or len(task) > 8192:
            raise ValueError('A nonempty instruction of at most 8192 characters is required')
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError('top_k must be between 1 and 100')
        if not isinstance(exclude_ids, (list, tuple)) or not all(isinstance(x, str) for x in exclude_ids):
            raise ValueError('exclude_ids must be a list of episode identities')
        if image is None:
            raise ValueError('A real current-scene image is required')
        query = self.encoder.encode([{'text': task}, {'image': image}])
        task_scores, image_scores = self.text @ query[0], self.image @ query[1]
        scores = self.text_weight * task_scores + (1-self.text_weight) * image_scores
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite retrieval scores')
        exclusions, results = set(exclude_ids), []
        for i in np.argsort(-scores, kind='stable'):
            row = self.rows[i]
            if row['episode_id'] in exclusions or 'robodojo:'+row['episode_id'] in exclusions:
                continue
            results.append(dict(source='robodojo', episode_id=row['episode_id'], rank=len(results)+1,
                score=float(scores[i]), task_score=float(task_scores[i]), scene_score=float(image_scores[i]),
                score_type='weighted_cosine_uncalibrated'))
            if len(results) == top_k:
                break
        return dict(task=task, method=METHOD, status='neural_retrieval_quality_unvalidated',
            ranking_inputs=['task', 'image'], scene_used_for_ranking=True,
            gallery_episodes=len(self.rows), text_weight=self.text_weight, results=results)


def make_server(searcher, port=8090):
    """Loopback-only, serial inference service (one GPU model, no request races)."""
    from PIL import Image

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != '/search':
                self.send_error(404)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 14 * 1024**2:
                    raise ValueError('Invalid request size')
                self.connection.settimeout(30)
                payload = json.loads(self.rfile.read(size))
                data = base64.b64decode(payload['image_base64'], validate=True)
                if len(data) > 10 * 1024**2:
                    raise ValueError('Query image exceeds 10 MiB')
                # HTTP query is a standard PNG/JPEG, never a stored XPolicyLab JPEG buffer.
                with Image.open(io.BytesIO(data)) as source:
                    if source.width * source.height > 16_000_000:
                        raise ValueError('Query image dimensions exceed the service limit')
                    frame = source.convert('RGB')
                result = searcher.search(payload['task'], frame, payload.get('top_k', 1),
                                         payload.get('exclude_ids', []))
                body, status = canonical(result), 200
            except (ValueError, KeyError, TypeError, OSError):
                body, status = b'{"error":"invalid retrieval request"}', 400
            except Exception:
                body, status = b'{"error":"retrieval inference failed"}', 500
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return HTTPServer(('127.0.0.1', port), Handler)


def serve(searcher, port=8090):
    with make_server(searcher, port) as server:
        print(json.dumps({'status': 'serving', 'endpoint': f'http://127.0.0.1:{port}',
                          'episodes': len(searcher.rows), 'method': METHOD}), flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['catalog', 'index', 'serve'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    parser.add_argument('--model', type=Path, default=DEFAULT_WORK/'models/Qwen3-VL-Embedding-2B')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--port', type=int, default=8090)
    args = parser.parse_args()
    if args.command == 'catalog':
        result = build_catalog(args.root, args.work, endpoint=f'http://127.0.0.1:{args.port}')
    else:
        encoder = Encoder(args.model, args.device)
        if args.command == 'index':
            result = build_index(args.work, encoder, batch_size=args.batch_size)
        else:
            return serve(Searcher(args.work, encoder), args.port)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
