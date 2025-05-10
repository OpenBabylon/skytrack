import os
import json
import unittest
from tempfile import NamedTemporaryFile
from skytrack.cli import generate_grid, apply_resource_rules, JobState, load_job_states, save_job_states

class TestSkyTuneLogic(unittest.TestCase):
    def test_generate_grid(self):
        params = {"a": [1,2], "b": ["x","y"]}
        grid = generate_grid(params)
        expected = [
            {"a":1,"b":"x"}, {"a":1,"b":"y"},
            {"a":2,"b":"x"}, {"a":2,"b":"y"}
        ]
        self.assertEqual(len(grid), 4)
        self.assertCountEqual(grid, expected)

    def test_apply_resource_rules(self):
        params = {"model": "A", "size": 10}
        rules = [
            {"if": {"model": "A"}, "resources": {"accelerators": "V100:2", "cpus": 4}},
            {"if": {"model": "B"}, "resources": {"accelerators": "V100:1", "cpus": 2}}
        ]
        res = apply_resource_rules(params, rules)
        self.assertEqual(res, {"accelerators": "V100:2", "cpus": 4})
        # If no rule matches, expect empty dict
        res2 = apply_resource_rules({"model": "C"}, rules)
        self.assertEqual(res2, {})

    def test_jobstate_save_load(self):
        # Create some dummy job states
        jobs = [JobState({"lr": 0.1}), JobState({"lr": 0.01})]
        jobs[0].status = "DONE"
        jobs[0].attempts = 1
        jobs[1].status = "FAILED"
        jobs[1].attempts = 2
        jobs[1].error = "Out of memory"
        # Save to temp file
        tmp = NamedTemporaryFile(delete=False, mode='w+')
        try:
            save_job_states(tmp.name, jobs)
            # Load back
            loaded = load_job_states(tmp.name)
            self.assertEqual(len(loaded), 2)
            # Verify contents match
            self.assertEqual(loaded[0].params, jobs[0].params)
            self.assertEqual(loaded[0].status, "DONE")
            self.assertEqual(loaded[1].status, "FAILED")
            self.assertEqual(loaded[1].error, "Out of memory")
        finally:
            tmp.close()
            os.unlink(tmp.name)

if __name__ == '__main__':
    unittest.main()
