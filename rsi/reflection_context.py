"""Read the retained execution evidence and candidate source in bounded pages."""
import base64
import importlib.util
import json
from pathlib import Path

HARNESS = 'XPolicyLab/policy/WorldScape_RSI/harness'
VENDOR = 'vendor/policy/RoboRSI_RoboProbe'


def load_library(root):
    path = Path(root)/HARNESS/VENDOR/'library.py'
    spec = importlib.util.spec_from_file_location('candidate_skill_library', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SkillLibrary(path.parent/'skills')


def editable_path(path):
    p = Path(path)
    return (not p.is_absolute() and '..' not in p.parts and
            (p.as_posix().startswith(VENDOR+'/') and p.suffix == '.py'))


def read_episode(row, start=0, count=4, images=False):
    """All steps are addressable. Image pages contain one step, with three cameras."""
    if row.get("record_purpose") != "collection":
        raise ValueError("Reflection may read collection records only")
    import h5py
    from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit
    from .experience import CHANNELS, CAMERAS
    if type(start) is not int or start < 0 or type(count) is not int or not 1 <= count <= 8:
        raise ValueError('Use a nonnegative start and count in 1..8')
    if not row.get('trace_path'):
        return {'trajectory_available': False, 'reason': 'Historical record contains only outcome metadata'}, []
    frames = []
    with h5py.File(row['trace_path'], 'r') as f:
        total = int(f.attrs['frames'])
        count = 1 if images else count
        end = min(total, start+count)
        steps = []
        for i in range(start, end):
            obs = json.loads(f['execution/observation'].asstr()[i]) if 'execution/observation' in f else {
                'instruction': f['instruction'].asstr()[i],
                'state': {c: f['state/'+c+'s'][i].tolist() for c in CHANNELS}}
            steps.append({'step': i, 'observation': obs,
                          'action': {c: f['action/'+c+'s'][i].tolist() for c in CHANNELS},
                          'feedback': json.loads(f['execution/feedback'].asstr()[i]) if 'execution/feedback' in f else {}})
        events = []
        if 'execution/tool_events' in f:
            for raw in f['execution/tool_events'].asstr():
                event = json.loads(raw)
                if start <= event.get('policy_step', 0) < max(end, start+1):
                    events.append(event)
        result = dict(total_steps=total, steps=steps, tool_events=events,
                      next_step=end if end < total else None)
        if start >= total and 'last_observation' in f:
            result['final_observation'] = json.loads(f['last_observation'].attrs['public'])
        if images and start <= total and (start < total or 'last_observation' in f):
            for cam in CAMERAS:
                dataset = f['vision/'+cam+'/colors'][start] if start < total else f['last_observation'][cam][:]
                rgb = decode_image_bit(dataset)
                # Shared RGB encoder is also used for the reflection image payload.
                payload = base64.b64encode(encode_image_bit(rgb)).decode()
                frames += [{'type': 'text', 'text': f'{cam}, observation {start}'},
                           {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+payload}}]
        return result, frames


def context_tools():
    def tool(name, description, properties):
        return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'additionalProperties': False,
                           'properties': properties, 'required': list(properties)}}}
    return [tool('read_episode', 'Read any retained execution step, actions, observations and tool feedback. '
                 'images=true supplies three camera frames for one step; start=total_steps reads the final observation.',
                 {'episode_id': {'type': 'string'}, 'start': {'type': 'integer'},
                  'count': {'type': 'integer'}, 'images': {'type': 'boolean'}}),
            tool('read_artifact', 'Read a Skill by its ID, a memory/<id>.md rule, or editable tool source path, in pages.',
                 {'path': {'type': 'string'}, 'offset': {'type': 'integer'}})]


def read_artifact(root, skills, path, offset):
    if path in skills:
        text = skills[path] if isinstance(skills[path], str) else skills[path]['text']
    else:
        rel = Path(path)
        if not (editable_path(path) or (rel.parent.as_posix() == 'memory' and rel.suffix == '.md')):
            raise ValueError('Read a listed Skill, Memory rule or tool source')
        text = (Path(root)/HARNESS/rel).read_text(encoding='utf8')
    if type(offset) is not int or not 0 <= offset <= len(text):
        raise ValueError('Invalid offset')
    end = min(len(text), offset+16000)
    return dict(text=text[offset:end], next_offset=end if end < len(text) else None, total_chars=len(text))
