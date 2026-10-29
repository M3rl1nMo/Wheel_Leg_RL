import unittest
import torch
import direct_residual_ppo as r
import direct_temporal_ppo as t
from direct_expanded_imitation import ExpandedPolicy

class TrainingVariantTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(6)
    def test_expanded_initially_preserves_base(self):
        p=ExpandedPolicy();x=torch.randn(4,27)
        self.assertTrue(torch.equal(p(x),p.base(x)))
        self.assertTrue(all(not v.requires_grad for v in p.base.parameters()))
    def test_residual_preserves_physics_and_reward(self):
        p=r.m.PolicyNetwork(27);a=r.Residual()
        e=r.m.p.DirectEnv();f=r.m.p.DirectEnv();o=e.reset(((.2,0.),0.));q=f.reset(((.2,0.),0.))
        expected=0.;discount=1.
        with torch.no_grad():
            for _ in range(r.REPEAT):
                q,reward,_,_,_=f.step(p(q[None])[0].tanh()*r.m.LIMITS)
                expected+=discount*reward;discount*=f.gamma
            o,reward,_,_,_,actual_discount=r.advance(e,p,o,a(o[None])[0])
        self.assertTrue(torch.equal(o,q));self.assertAlmostEqual(reward,expected,places=12)
        self.assertAlmostEqual(discount,actual_discount,places=12)
    def test_gae_horizon_changes_only_targets(self):
        e=t.c.SnapshotEnv();e.reset(((.2,0.),0.));s=e.snapshot('test','normal')
        p=t.PolicyNetwork(27);v=t.ValueNetwork(28)
        a,_,_=t.collect(p,v,[s],16,9,2);b,_,_=t.collect(p,v,[s],16,9,10)
        self.assertTrue(torch.equal(a['raw'],b['raw']));self.assertTrue(torch.equal(a['obs'],b['obs']))
        self.assertFalse(torch.equal(a['returns'],b['returns']));self.assertTrue(torch.isfinite(b['returns']).all())

if __name__=='__main__':unittest.main()
