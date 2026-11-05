"""Private worker process connecting the external ICL Engine to XPolicyLab data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "XPolicyLab"))


def prepare(config_path, request):
    import icl_engine
    from icl_engine.compiler import verify_package
    from icl_engine.config import load_config
    from icl_engine.engine import ICLEngine
    from XPolicyLab.utils.process_data import decode_image_bit

    cfg = load_config(config_path)
    from XPolicyLab.policy.WorldScape_R2SI.robodojo_corpus import install_readers
    install_readers(cfg.get("data", {}).get("official_root"))
    # Never reuse a package compiled with the engine's standard-JPEG-only reader.
    decoder_hash = hashlib.sha256(Path(sys.modules[decode_image_bit.__module__].__file__).read_bytes()).hexdigest()[:16]
    cfg["paths"]["packages"] = str(Path(cfg["paths"]["packages"]) / ("xpl-rgb1-" + decoder_hash))
    if "backend" not in cfg.get("retrieval", {}):
        raise ValueError("Select an explicit ICL retrieval backend")
    context = cfg.setdefault("context", {})
    context["max_demo_images"] = min(context.get("max_demo_images", 64), request["max_images"])
    context["max_demo_text_chars"] = min(context.get("max_demo_text_chars", 256000), request["max_text_chars"])
    bundle = ICLEngine(cfg).prepare(request["instruction"], image=request["image"], top_k=1)
    manifest = verify_package(bundle["package"])
    # Engine context ends with the current query. R2SI supplies its own fresh live state/images.
    messages = bundle["messages"][:-1]
    retrieval = bundle["retrieval"]
    selected = bundle["selected_episode"]
    metadata = {"status": "ready", "engine_version": icl_engine.__version__,
                "package_id": bundle["package_id"], "prefix_sha256": bundle["prefix_sha256"],
                "source": selected["source"], "episode_id": selected["episode_id"],
                "backend": retrieval["backend"], "method": retrieval.get("method"),
                "retrieval_status": retrieval.get("status"),
                "scene_used_for_ranking": retrieval.get("scene_used_for_ranking", False),
                "recipe": manifest["recipe"], "image_count": manifest["image_count"],
                "selection_scope": "episode_start", "all_demo_resent_every_decision": True,
                "coordinate_transfer_verified": False, "command_replay_verified": False,
                "decoder": "XPolicyLab.utils.process_data.decode_image_bit"}
    return {"status": "ready", "messages": messages, "metadata": metadata}


def main():
    config, request_path, result_path = sys.argv[1:]
    try:
        result = prepare(config, json.loads(Path(request_path).read_text(encoding="utf-8")))
        code = 0
    except Exception as error:
        result, code = {"status": "error", "error_type": type(error).__name__}, 1
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
