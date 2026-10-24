"""Nested subprocess temporary-root propagation, with no model inference."""
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rsi.storage import Workspace

def child_environment(root, environment, output):
    os.environ.update(environment)
    env=Workspace(root).environment()
    output.send({k:env[k] for k in ("TMPDIR","TMP","TEMP")})
    output.close()

class SharedTemporaryRoot(unittest.TestCase):
    def test_nested_spawn_keeps_short_shared_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ws=Workspace(root/("long-workspace-"*12))
            supplied={"R2SI_STORAGE_ROOT":str(root),"R2SI_TEMP_ROOT":str(root/"t")}
            first=ws.environment(supplied)
            self.assertLess(len(first["TMPDIR"]),len(str(ws.path("tmp"))))
            ctx=multiprocessing.get_context("spawn");read,write=ctx.Pipe(False)
            proc=ctx.Process(target=child_environment,args=(str(ws.root),first,write))
            proc.start();write.close()
            try:
                self.assertTrue(read.poll(20),"child did not return environment")
                result=read.recv()
                self.assertEqual(result,{k:first[k] for k in result})
            finally:
                proc.join(5)
                if proc.is_alive():proc.terminate();proc.join()
                read.close()
            self.assertEqual(proc.exitcode,0)
    def test_unconfigured_tmp_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            ws=Workspace(directory)
            self.assertEqual(ws.environment({"TMPDIR":"/tmp"})["TMPDIR"],str(ws.path("tmp")))
    def test_missing_storage_boundary_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                Workspace(directory).environment({"R2SI_TEMP_ROOT":directory})
    def test_temp_outside_shared_root_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inside=root/"shared";inside.mkdir()
            with self.assertRaises(ValueError):
                Workspace(inside/"workspace").environment({"R2SI_STORAGE_ROOT":str(inside),"R2SI_TEMP_ROOT":str(root/"outside")})
    def test_workspace_outside_shared_root_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inside=root/"shared";inside.mkdir()
            with self.assertRaises(ValueError):
                Workspace(root/"outside").environment({"R2SI_STORAGE_ROOT":str(inside),"R2SI_TEMP_ROOT":str(inside/"t")})
    def test_redirecting_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ws=Workspace(root/"w");temp=root/"t";temp.mkdir();other=root/"other";other.mkdir()
            identity=hashlib.sha256(str(ws.root).encode()).hexdigest()[:16]
            (temp/identity).symlink_to(other,target_is_directory=True)
            with self.assertRaises(ValueError):
                ws.environment({"R2SI_STORAGE_ROOT":str(root),"R2SI_TEMP_ROOT":str(temp)})

if __name__=="__main__":unittest.main()
