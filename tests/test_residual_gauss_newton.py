"""Generated numerical contracts; no GRAIL assets or native model forwards."""
from collections import Counter
import unittest
import numpy as np
import torch
from grail.optimization.residual_gauss_newton import (
    residual_jacobian, solve_quadratic, agreement, capture_vector_jacobian,
    capture_scalar_gradient, check_objective_gradient,
)


class ResidualGaussNewtonTests(unittest.TestCase):
    def capture(self, r, x, scalar=None):
        counts = Counter(); saved = []
        result = residual_jacobian(r, x, (r*r).sum() if scalar is None else scalar,
            counts=counts, max_residuals=20, max_reverse_calls=len(x)+2,
            identity_atol=2e-5, identity_rtol=2e-4,
            checkpoint=lambda i,J:saved.append((i,J.copy())))
        self.assertEqual(counts['autograd_reverse_invocations_returned'],len(x)+2)
        self.assertEqual(len(saved),len(x))
        return result

    def test_analytic_jacobian_and_connected_zero_column(self):
        x=torch.tensor([.2,-.7,.3],dtype=torch.float64,requires_grad=True)
        A=torch.tensor([[2.,1.,0.],[-1.,3.,0.],[.5,0.,0.],[1.,1.,0.]],dtype=x.dtype)
        r=A@x+torch.stack([x[0].sin(),x[1].sin(),x[0]*x[1],x[0]*0])
        expected=A.numpy().copy();expected[0,0]+=np.cos(.2);expected[1,1]+=np.cos(-.7)
        expected[2,:2]+=[-.7,.2]
        _,J,report=self.capture(r,x)
        np.testing.assert_allclose(J,expected,rtol=1e-12,atol=1e-12)
        self.assertTrue(report['passed']);np.testing.assert_array_equal(J[:,2],0)

    def test_mixed_float_projection_against_independent_row_jacobian(self):
        def residual(x):
            p=torch.stack([x[0]+.2,x[1]-.1,x[2]+2.]).float()
            q=torch.stack([p[0]/p[2],p[1]/p[2],torch.sin(p[0]*p[1]),p.norm()])
            return q.double()
        x=torch.tensor([.1,.3,.2],dtype=torch.float32,requires_grad=True)
        expected=torch.autograd.functional.jacobian(residual,x).double().numpy()
        _,J,_=self.capture(residual(x),x)
        np.testing.assert_allclose(J,expected,rtol=1e-6,atol=1e-7)

    def test_scalar_mismatch_rejected(self):
        x=torch.tensor([.2,.3],dtype=torch.float64,requires_grad=True);r=x*x
        with self.assertRaisesRegex(ValueError,'gradient identity'):
            self.capture(r,x,2*(r*r).sum())

    def test_argument_rejection_before_reverse(self):
        x=torch.tensor([.2],requires_grad=True);r=x*x;counts=Counter()
        with self.assertRaises(ValueError):
            residual_jacobian(r,x,r.sum(),counts=counts,max_residuals=3,max_reverse_calls=2,identity_atol=1e-6,identity_rtol=0)
        self.assertFalse(counts)

    def test_constrained_direction_matches_analytic_boundary(self):
        # min (x-2)^2+y^2 with x<=.3 and radius1 has solution [.3,0].
        trace={};ans=solve_quadratic([-2.,0.],np.eye(2),[[1.,0.]],[.3],radius=1.,blocks=[],trace=trace)
        self.assertEqual(ans['status'],'direction_ready')
        np.testing.assert_allclose(ans['direction'],[.3,0.],atol=1e-7)
        self.assertEqual(trace['solver_calls_returned'],1)
        self.assertFalse(ans['separate_optimality_certificate'])

    def test_initial_infeasible_and_invalid_block_do_not_solve(self):
        trace={};ans=solve_quadratic([-2.],[[1.]],[[1.]],[-.1],radius=1.,blocks=[],trace=trace)
        self.assertEqual(ans['status'],'initial_zero_infeasible');self.assertFalse(trace)
        with self.assertRaises(ValueError):
            solve_quadratic([-2.],[[1.]],[[1.]],[.3],radius=1.,blocks=[dict(indices=[.5],radius=1.)],trace={})

    def test_agreement_uses_actual_step_and_rejects_non_improvement(self):
        yes=agreement([2.],[[1.]],[-.5],4.,2.25)
        self.assertEqual(yes['rho'],1.);self.assertTrue(yes['ridge_excluded'])
        self.assertFalse(agreement([2.],[[1.]],[-.5],4.,4.1)['pass_model_agreement'])
        self.assertFalse(agreement([2.],[[1.]],[-.5],4.,float('nan'))['pass_model_agreement'])

    def test_signed_constraints_stay_outside_objective_identity(self):
        x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        residual = torch.stack((x[0]-1., 2*x[1]+.3))
        guards = torch.stack((-.002-.001*x[0], 1000*x[1]+200))
        counts = Counter()
        values, J = capture_vector_jacobian(torch.cat((residual, guards)), x,
            counts=counts, max_outputs=4, max_reverse_calls=4)
        g = capture_scalar_gradient(residual.square().sum(), x,
            counts=counts, max_reverse_calls=4)
        np.testing.assert_array_equal(J[2], [-.001, 0.])
        self.assertEqual(values[2], -.002)
        report = check_objective_gradient(values[:2], J[:2], g,
            identity_atol=1e-12, identity_rtol=1e-12)
        self.assertTrue(report['passed'])
        self.assertEqual(counts['autograd_reverse_invocations_returned'], 4)
        with self.assertRaisesRegex(ValueError, 'gradient identity'):
            check_objective_gradient(values, J, g,
                identity_atol=1e-12, identity_rtol=1e-12)

    def test_vector_capture_has_no_implicit_scalar_reverse(self):
        x = torch.tensor([.3, .8], dtype=torch.float64, requires_grad=True)
        vector = torch.stack((x[0].square(), x[0]+1.))
        counts = Counter()
        _, J = capture_vector_jacobian(vector, x, counts=counts,
            max_outputs=2, max_reverse_calls=3)
        np.testing.assert_array_equal(J[:, 1], np.zeros(2))
        self.assertEqual(counts['autograd_reverse_invocations_returned'], 3)
        self.assertEqual(counts['independent_scalar_merit_reverses_started'], 0)


if __name__=='__main__':unittest.main()
