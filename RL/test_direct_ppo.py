import unittest
from unittest.mock import patch
import numpy as np
import torch
import direct_ppo as p
import direct_navigation as d


class DirectPPOTests(unittest.TestCase):
    def test_bounded_shaping_and_terminal(self):
        gamma=.9998
        self.assertEqual(p.potential(0.),1.)
        self.assertGreater(p.shaping(.05,.04,gamma,False),p.shaping(.3,.29,gamma,False))
        self.assertLessEqual(p.shaping(.05,.05,gamma,False),0.)
        self.assertAlmostEqual(p.shaping(.1,0.,gamma,True),-p.potential(.1))
        distances=[.3,.2,.1,.0]
        total=sum(gamma**i*p.shaping(distances[i],distances[i+1],gamma,i==2) for i in range(3))
        self.assertAlmostEqual(total,-p.potential(.3))

    def test_no_outer_and_actual_previous_action(self):
        with patch('planar_control.PlanarPositionController',side_effect=AssertionError('outer used')), \
             patch('planar_control.PlanarLQRTeacher',side_effect=AssertionError('teacher used')):
            env=p.DirectEnv(max_seconds=.004)
            obs=env.reset(((.2,.1),0.))
            self.assertEqual(tuple(obs.shape),(27,))
            self.assertTrue(torch.equal(obs[-6:],torch.zeros(6)))
            tau=torch.ones(6)*.01
            nxt,reward,term,trunc,_=env.step(tau)
            self.assertTrue(torch.allclose(nxt[-6:],tau/p.LIMITS))
            self.assertTrue(np.isfinite(reward));self.assertFalse(term);self.assertFalse(trunc)
            _,_,term,trunc,_=env.step(tau)
            self.assertFalse(term);self.assertTrue(trunc)

    def test_gae_termination_and_truncation(self):
        a,_=p.gae(torch.tensor([1.,2.]),torch.tensor([.5,.2]),torch.tensor([9.,3.]),
                  torch.tensor([1.,0.]),torch.tensor([1.,1.]),.9,.95)
        self.assertTrue(torch.allclose(a,torch.tensor([.5,4.5])))


if __name__=='__main__':unittest.main()
