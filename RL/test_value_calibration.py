import unittest
import numpy as np
import torch
from direct_value_calibration import SnapshotEnv,CASES


class SnapshotTests(unittest.TestCase):
    def test_physics_and_previous_action_preserved_but_success_timer_reset(self):
        source=SnapshotEnv();source.reset(((.24,0.),.2))
        torque=torch.full((6,),.01)
        for _ in range(50):source.step(torque)
        source.streak=800;source.arrival=.05
        state=source.snapshot('unit','moving')
        restored=SnapshotEnv();obs=restored.restore(state)
        self.assertTrue(torch.allclose(obs,state['observation']))
        self.assertTrue(torch.equal(restored.previous,torque))
        self.assertEqual(restored.streak,0);self.assertIsNone(restored.arrival)
        self.assertEqual(restored.data.time,0.)
        source.step(torque);restored.step(torque)
        self.assertTrue(np.allclose(source.data.qpos,restored.data.qpos,atol=1e-10,rtol=0.))
        self.assertTrue(np.allclose(source.data.qvel,restored.data.qvel,atol=1e-10,rtol=0.))

    def test_source_cases_are_disjoint(self):
        sets=[set(cases) for cases in CASES.values()]
        for i in range(len(sets)):
            for j in range(i+1,len(sets)):self.assertFalse(sets[i]&sets[j])


if __name__=='__main__':unittest.main()
