"""Regression checks for the RoboDojo-only policy boundary and checkpoint geometry."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
POLICY = PROJECT / 'XPolicyLab/policy/WorldScape_RSI'
WSP = POLICY / 'worldscape-policy'
for path in (PROJECT, WSP, WSP / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from XPolicyLab.policy.WorldScape_RSI.xpolicylab_adapter.runtime_config import validate_config


def test_default_and_explicit_deployments_select_robodojo():
    assert validate_config({})['runtime_profile'] == 'robodojo'
    for name in ('wam_deploy.yml', 'wam_deploy_robodojo.yml'):
        cfg = validate_config(yaml.safe_load((POLICY / name).read_text()))
        assert cfg['runtime_profile'] == 'robodojo'
        assert cfg['diffusion_view_layout'] == 'mosaic_2x2'
        assert cfg['action_horizon'] == 24


@pytest.mark.parametrize('cfg', [
    {'runtime_profile': 'robotwin2'},
    {'bench_name': 'RoboTwin'},
    {'diffusion_view_layout': 'robotwin_concat'},
    {'allow_dummy_policy': True},
])
def test_unsupported_runtime_rejected_before_loading(cfg):
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize('script,args', [
    ('train.sh', ['RoboTwin', 'test', 'arx_x5', 'joint', '62', '0']),
    ('process_data.sh', ['RoboTwin', 'test', 'arx_x5', 'joint']),
    ('eval.sh', ['RoboTwin']),
    ('setup_eval_policy_server.sh', ['RoboTwin']),
    ('setup_eval_env_client.sh', ['RoboTwin']),
])
def test_unsupported_benchmark_rejected_before_environment_setup(script, args):
    result = subprocess.run(['bash', str(POLICY / script), *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'only bench_name=RoboDojo' in result.stderr


@pytest.mark.parametrize('multi', [False, True])
def test_model_dispatches_only_robodojo_backends(multi):
    from XPolicyLab.policy.WorldScape_RSI.wam_model import Model
    module = 'multi_gpu.MultiGpuModel' if multi else 'robodojo_model.RoboDojoModel'
    cfg = {'eval_batch': multi, 'multi_gpu': {'enabled': multi}}
    backend = SimpleNamespace(model_cfg=cfg, mode=SimpleNamespace(value='interactive'),
                              action_horizon=24, supports_batch=True, requires_explicit_commit=True)
    with patch('XPolicyLab.policy.WorldScape_RSI.xpolicylab_adapter.' + module, return_value=backend):
        model = Model(cfg)
    assert model.backend is backend
    assert model.get_runtime_info()['runtime_profile'] == 'robodojo'
    assert model.get_runtime_info()['requires_explicit_commit']


@pytest.mark.parametrize('name,chunks,checkpointing', [('robodojo.yaml', 1, False), ('robodojo_memory.yaml', 24, True)])
def test_pretrained_adapter_row_and_training_geometry_preserved(monkeypatch, name, chunks, checkpointing):
    from worldscape_policy.cli.config_composition import load_composed_config
    from worldscape_policy.cli.config_profiles import resolve_config_profiles
    for key in ('WAM_MAX_CHUNK_SIZE', 'WAM_GRADIENT_CHECKPOINTING', 'ACTION_HORIZON'):
        monkeypatch.delenv(key, raising=False)
    cfg = resolve_config_profiles(load_composed_config(WSP / 'configs/posttrain' / name))
    assert cfg.model.pretrained_action_adapter_index == 4
    assert cfg.data_loader.dataset_kwargs.action_horizon == 24
    assert cfg.data_loader.dataset_kwargs.max_chunk_size == chunks
    assert cfg.data_loader.dataset_kwargs.normalization == 'zscore'
    assert cfg.training.gradient_checkpointing is checkpointing


def test_three_camera_mosaic_keeps_camera_order_and_empty_quadrant():
    from worldscape_policy.visual_mosaic import compose_three_view_mosaic
    views = torch.stack([torch.full((2, 3, 4, 6), n) for n in (10., 20., 30.)], dim=1)[None]
    mosaic = compose_three_view_mosaic(views)
    assert tuple(mosaic.shape) == (1, 2, 3, 8, 12)
    assert torch.all(mosaic[..., :4, :6] == 10)
    assert torch.all(mosaic[..., 4:, :6] == 20)
    assert torch.all(mosaic[..., :4, 6:] == 30)
    assert torch.all(mosaic[..., 4:, 6:] == 0)
