"""Serve one immutable trained WAM generation beside the existing service."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import ssl
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'XPolicyLab')]

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    from rsi.policy_service import serve
    context=None
    if config.get('certificate'):
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(config['certificate'],config['private_key'])
    asyncio.run(serve(Path(config['project']),config['workspace'],config['host'],[config['port']],
        config.get('max_trials',60), device_profiles={config['profile']:[config['gpu']]},
        checkpoints=config['checkpoints'],ssl_context=context))
