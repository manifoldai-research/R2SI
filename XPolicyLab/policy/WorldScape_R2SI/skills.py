"""Shared generic experience for the code-writing Agent and outer RSI loop."""
from pathlib import Path

from .harness.vendor.policy.RoboRSI_RoboProbe.library import SkillLibrary

ROOT = Path(__file__).resolve().parent / 'harness'


def library():
    return SkillLibrary(ROOT / 'vendor/policy/RoboRSI_RoboProbe/skills')


def memory_files():
    """The Memory rule files every episode loads, in order."""
    return [path.name for path in sorted((ROOT / 'memory').glob('*.md'))]


def shared_context():
    """Load candidate Memory rules on each episode; never reuse another version's cache."""
    rows = ['# Memory rules']
    for path in sorted((ROOT / 'memory').glob('*.md')):
        if path.is_symlink():
            raise ValueError('Memory must be a regular release file')
        rows.append(path.name + '\n' + path.read_text(encoding='utf-8'))
    return '\n\n'.join(rows) + '\n\n'
