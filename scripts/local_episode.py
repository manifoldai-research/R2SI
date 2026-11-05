"""One native local RSI episode in its own simulator and Agent processes."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'XPolicyLab')]
from rsi.storage import Workspace


def execute(workspace, input_path):
    ws = Workspace(workspace)
    value = json.loads(Path(input_path).read_text()); config = value['config']; meta = value['metadata']
    from rsi.official_layouts import MODE, validate_case, reject_open, uses_official_layouts
    reject_open(meta)
    if config.get('robodojo_root') or uses_official_layouts(meta):
        validate_case(meta)
    os.environ.update(ws.environment())
    os.environ.update(config.get('environment', {}))
    for path in reversed(config.get('python_paths', [])):
        sys.path.insert(0, path)
    if config.get('pool_bootstrap'):
        file = Path(config['pool_bootstrap'])
        spec = importlib.util.spec_from_file_location('local_pool_bootstrap', file)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.configure_pool()
    app = env = model = None
    episode_owned = False
    result = dict(meta, infrastructure_error=True, terminal_confirmed=False,
                  error='Episode did not reach execution')
    try:
        if config.get('robodojo_root'):
            from isaaclab.app import AppLauncher
            from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
            parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
            args = parser.parse_args([])
            args.headless, args.enable_cameras, args.device = True, True, value['device']
            runtime = ws.path('simulator',meta['episode_id']); runtime.mkdir(parents=True)
            args.kit_args = '--allow-root --portable-root '+str(runtime)+' --/app/settings/persistent=false --/log/file='+str(runtime/'kit.log')
            add_zero_delay_kit_args(args)
            app = AppLauncher(args).app
            from rsi.native_robodojo import configure, make_environment
            configure(app, config['robodojo_root'])
        else:
            module, name = config['environment_factory'].split(':')
            make_environment = getattr(importlib.import_module(module), name)
        from rsi.registry import Registry, verify_code
        from rsi.worker import VersionedModel
        from rsi.evaluator import run_episode
        env = make_environment(dict(meta, device=value['device']))
        model = VersionedModel(ws.root, dict(policy_name='WorldScape_R2SI', bench_name='RoboDojo',
            env_cfg_type='arx_x5', action_type='joint', eval_batch=False, seed=meta['seed']),
            validation_release=meta['release_id'], registry=Registry(ws, verifier=verify_code),
            timeout_s=config.get('agent_timeout_s', 600),
            memory_device=config.get('memory_device', 'cuda:0'), flash_device=config.get('flash_device', 'cuda:0'))
        episode_owned = True
        result = run_episode(env, model, ws, meta,
                             max_wall_seconds=config['max_wall_seconds_per_episode'], record=value['record'])
    except BaseException:
        result = dict(meta, infrastructure_error=True, terminal_confirmed=False, error=traceback.format_exc(limit=12))
    finally:
        # The parent publishes a result only after observing this process exit.
        # SimulationApp.close can exit directly, so retain a private receipt first.
        cleanup_errors = []
        operations = []
        if model is not None:
            operations.append(model.close)
        if env is not None and not episode_owned:
            operations.append(env.close)
        for operation in operations:
            try:
                operation()
            except BaseException as error:
                cleanup_errors.append(type(error).__name__ + ': ' + str(error))
        if cleanup_errors:
            result = dict(meta, infrastructure_error=True, terminal_confirmed=False,
                          error='Cleanup incomplete: ' + '; '.join(cleanup_errors))
        ws.atomic('episode-jobs/provisional/'+meta['episode_id']+'.json',
                  dict(schema='rsi.episode-exit-receipt.v1', metadata=meta,
                       cleanup_before_app=not cleanup_errors, result=result))
        if app is not None:
            app.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True); parser.add_argument('--input', required=True)
    args = parser.parse_args()
    execute(args.workspace, args.input)
