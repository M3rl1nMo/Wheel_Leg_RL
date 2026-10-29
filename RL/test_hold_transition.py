"""保持模式切换的行为测试；从项目根目录运行 python RL/test_hold_transition.py。"""
import copy
import unittest
from planar_control import PlanarPositionController
from planar_navigation import CONFIG, LEGACY_CONFIG, validate_policy_config


class HoldTransitionTests(unittest.TestCase):
    def test_transition_matches_previous_command(self):
        for velocity in (0., .03, .073):
            with self.subTest(velocity=velocity):
                control=PlanarPositionController()
                control.integral=.2
                control.previous_velocity=velocity
                *_,command,_=control.update((0.,0.),0.,(.014,.02),.002)
                self.assertTrue(control.holding)
                self.assertAlmostEqual(command,velocity,places=12)
                self.assertLess(abs(control.integral),.2)
                # 后续步仍根据误差积分，不能每步都重置为切换时的值。
                integral=control.integral
                *_,next_command,_=control.update((0.,0.),0.,(.014,.02),.002)
                self.assertGreater(control.integral,integral)
                self.assertGreater(next_command,command)

    def test_clamps_integral_when_previous_command_is_unreachable(self):
        control=PlanarPositionController()
        control.previous_velocity=.15
        *_,command,_=control.update((0.,0.),0.,(-.02,0.),.002)
        self.assertEqual(control.integral,.2)
        self.assertLessEqual(abs(command),.15)
        # 受积分限幅影响时允许残余跳变，不为强行匹配而突破积分范围。
        self.assertLess(command,.15)

    def test_reset_and_exit(self):
        control=PlanarPositionController()
        self.assertEqual(control.previous_velocity,0.)
        control.update((0.,0.),0.,(.01,0.),.002)
        self.assertTrue(control.holding)
        control.update((0.,0.),0.,(.3,0.),.002)
        self.assertFalse(control.holding)

    def test_known_checkpoint_configs_only(self):
        validate_policy_config(CONFIG)
        validate_policy_config(LEGACY_CONFIG)
        validate_policy_config(dict(LEGACY_CONFIG,policy_activation='relu'))
        bad=copy.deepcopy(CONFIG);bad['observation_dim']=22
        with self.assertRaises(ValueError):validate_policy_config(bad)
        bad=copy.deepcopy(CONFIG);bad['outer_controller']['ki']=999.
        with self.assertRaises(ValueError):validate_policy_config(bad)


if __name__=='__main__':
    unittest.main()
