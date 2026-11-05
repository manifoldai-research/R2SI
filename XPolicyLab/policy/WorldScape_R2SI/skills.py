"""Shared generic experience for the code-writing Agent and outer RSI loop."""
from pathlib import Path

from .harness.vendor.policy.RoboRSI_RoboProbe.library import SkillLibrary

ROOT = Path(__file__).resolve().parent / 'harness'


def library():
    return SkillLibrary(ROOT / 'vendor/policy/RoboRSI_RoboProbe/skills')


def shared_context():
    """Load candidate Memory on each episode; never reuse another version's cache."""
    rows = ['# Shared Memory']
    for path in sorted((ROOT / 'memory').glob('*.md')):
        if path.is_symlink():
            raise ValueError('Memory must be a regular release file')
        rows.append(path.name + '\n' + path.read_text(encoding='utf-8'))
    return '\n\n'.join(rows) + '\n\n'
