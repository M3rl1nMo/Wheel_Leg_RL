"""独立网络的输入与运行隔离检查，不用通过率代替接口检查。"""
import unittest
from unittest.mock import patch
import torch
import direct_navigation as direct


class DirectNavigationTests(unittest.TestCase):
    def test_evaluation_never_constructs_teacher_or_position_outer(self):
        torch.manual_seed(9)
        policy=direct.PolicyNetwork(27)
        seen=[]
        def remember(module,args):
            if len(seen)<2:seen.append(args[0].clone())
        handle=policy.register_forward_pre_hook(remember)
        try:
            with patch('planar_control.PlanarPositionController',side_effect=AssertionError('outer used')), \
                 patch('planar_control.PlanarLQRTeacher',side_effect=AssertionError('teacher used')):
                result=direct.rollout(policy,((.2,.2),0.),seconds=2.)[2]
            self.assertFalse(result['runtime_position_outer'])
            self.assertEqual(seen[0].shape,(1,27))
            self.assertTrue(torch.equal(seen[0][:,-6:],torch.zeros(1,6)))
            with torch.no_grad():expected=policy(seen[0]).tanh()
            self.assertTrue(torch.allclose(seen[1][:,-6:],expected))
        finally:handle.remove()

    def test_schema_excludes_teacher_internal_state(self):
        self.assertEqual(len(direct.ORDER),27)
        for forbidden in ('command_velocity','command_turn_rate','remaining_time_fraction','integral','holding'):
            self.assertNotIn(forbidden,direct.ORDER)
        self.assertFalse(direct.CONFIG['runtime_position_outer'])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
