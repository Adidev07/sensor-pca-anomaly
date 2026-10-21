import unittest
from project import *
class ProjectTests(unittest.TestCase):
    def test_spike_increases_error(self):
        rng=np.random.default_rng(5);m=fit(normal(rng,200));x=normal(rng,100);bad=x.copy();bad[:,3]+=4
        self.assertGreater(float(score(m,bad).mean()),float(score(m,x).mean())*10)
    def test_component_orthogonality(self):
        m=fit(normal(np.random.default_rng(1),200));np.testing.assert_allclose(m['components']@m['components'].T,np.eye(2),atol=1e-10)
    def test_full_rank_reconstruction(self):
        x=normal(np.random.default_rng(1),200);self.assertLess(float(score(fit(x,5),x).max()),1e-20)
    def test_no_input_mutation(self):
        x=normal(np.random.default_rng(1),100);copy=x.copy();fit(x);np.testing.assert_array_equal(x,copy)
