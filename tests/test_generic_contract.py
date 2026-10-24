import unittest, tempfile
from pathlib import Path
from rsi.generic_contract import check_generic, check_text
from rsi.outer import OuterLoop, reflection_tool
from rsi.release_builder import build_release
class GenericContract(unittest.TestCase):
    def test_schema_only_generic(self):
        props=reflection_tool({})['function']['parameters']['properties']
        self.assertEqual(props['skill_updates']['items']['properties']['scope']['enum'],['generic'])
    def test_scope_and_selector_rejected(self):
        for row in ({'scope':'task','text':'Move safely'}, {'scope':'generic','task':'demo','text':'Move safely'}):
            with self.assertRaises(ValueError):check_generic(row)
    def test_task_aliases_rejected(self):
        for text in ('use align_blocks now', 'Use ALIGN BLOCKS.', 'Use align-blocks'):
            with self.assertRaises(ValueError):check_text(text,['align_blocks'])
        check_text('Align contact geometry before advancing',['align_blocks'])
    def test_task_memory_rejected(self):
        p=dict(hypothesis='Check contact',evidence_episode_ids=['e'],skill_updates=[],training_requested=False,
               training_reason='',expected_behavior=[],risks=[],memory_updates=[dict(id='contact',text='For align_blocks move left')])
        with self.assertRaises(ValueError):OuterLoop.validate_proposal(p,{}, {'e'},['align_blocks'])
    def test_builder_refuses_before_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                build_release(folder,'/not-a-source',{},skill_updates=[dict(id='x',scope='task',task='demo',text='Move')])
            self.assertFalse(list(Path(folder).iterdir()))
if __name__=='__main__':unittest.main()
