"""Load a pinned public WAM on CUDA without an Agent, simulator or robot actions."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['flash', 'memory'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if os.environ.get('WRSI_POLICY_CLIENT_CONFIG'):
        parser.error('Unset WRSI_POLICY_CLIENT_CONFIG to test local weight loading')
    import torch
    from XPolicyLab.policy.WorldScape_R2SI.harness.release import FrozenRelease, ROOT
    from XPolicyLab.policy.WorldScape_R2SI.harness.wam import create_model
    release = FrozenRelease(ROOT / 'release.json')
    key = 'worldscape-wam-memory' if args.variant == 'memory' else 'worldscape-wam-nomemory'
    spec = release.data['wam_skills'][key]
    os.environ[spec['checkpoint_env']] = str(args.checkpoint.resolve(strict=True))
    os.environ[spec['device_env']] = args.device
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('This real-weight smoke test requires CUDA')
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model = create_model(spec, {'seed': 0})
    try:
        model.reset()
        torch.cuda.synchronize(device)
        assert model.get_runtime_info()['requires_explicit_commit']
        from worldscape_policy.native_builder import _load_native_model_config
        config = _load_native_model_config(args.checkpoint.resolve()).model
        capacity = int(config.wam.core.parameters['max_chunk_size'])
        assert capacity == (24 if args.variant == 'memory' else 1)
        assert model.backend.wam_max_chunk_size == capacity
        assert model.backend.wam_history_mode == 'rolling'
        assert model.backend.use_observation_history
        assert model.backend.use_causal_memory
        assert not model.backend.reset_independent_blocks
        print(json.dumps({'status': 'passed', 'variant': args.variant,
            'action_horizon': model.backend.action_horizon,
            'context_chunks': capacity,
            'runtime_history_mode': model.backend.wam_history_mode,
            'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
            'elapsed_seconds': round(time.monotonic() - start, 2),
            'release_sha256': release.sha256,
            'action_inference': False, 'robot_execution': False}, indent=2))
    finally:
        model.close()


if __name__ == '__main__':
    main()
