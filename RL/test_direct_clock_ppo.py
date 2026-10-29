import unittest
import torch
import direct_clock_ppo as m

class ClockTests(unittest.TestCase):
    def test_clock_and_actor_separation(self):
        env=m.c.SnapshotEnv();obs=env.reset(((.2,0.),0.))
        a=m.critic_obs(env,obs);env.streak=500;b=m.critic_obs(env,obs)
        self.assertEqual(a.shape,(28,));self.assertEqual(float(b[-1]),.5)
        self.assertTrue(torch.equal(a[:27],b[:27]))
        actor=m.ActorView(m.PolicyNetwork(27))
        self.assertTrue(torch.equal(actor(a),actor(b)))
    def test_mixed_collection_and_update(self):
        torch.set_num_threads(1)
        env=m.c.SnapshotEnv();env.reset(((.2,0.),0.));state=env.snapshot('test','normal')
        policy=m.PolicyNetwork(27);value=m.ValueNetwork(28)
        batch,_,diag=m.collect(policy,value,[state,state],32,42)
        self.assertEqual(batch['obs'].shape,(32,28));self.assertEqual(len(diag),2)
        self.assertTrue(torch.isfinite(batch['returns']).all())
        actor=m.ActorView(policy);ref=m.copy.deepcopy(actor)
        stats=m.p.update(actor,value,ref,torch.optim.Adam(actor.parameters(),lr=1e-5),
            torch.optim.Adam(value.parameters(),lr=.0003),batch,.01,epochs=1)
        self.assertTrue(m.math.isfinite(stats['final_kl']))

if __name__=='__main__':unittest.main()
