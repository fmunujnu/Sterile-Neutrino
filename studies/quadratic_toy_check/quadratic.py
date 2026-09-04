"""Fixed Gaussian quadratic difference; characteristic-function inversion.

No Toy fitting and no normal approximation. This is NOT the exact law of a
reprofiled statistic. See Imhof (1961), doi:10.1093/biomet/48.3-4.419.
"""
import numpy as np
from scipy.linalg import cholesky, cho_factor, cho_solve, eigh
from scipy.integrate import simpson


class QuadraticLaw:
    """T = constant + sum(eigenvalue*z**2 + linear*z), independent z~N(0,1)."""
    def __init__(self, constant, eigenvalues, linear):
        self.constant = float(constant)
        self.eigenvalues = np.asarray(eigenvalues, float)
        self.linear = np.asarray(linear, float)
        if self.eigenvalues.shape != self.linear.shape:
            raise ValueError("Coefficient dimensions differ")
        self.mean = self.constant + self.eigenvalues.sum()
        self.sigma = np.sqrt(2*np.sum(self.eigenvalues**2)+np.sum(self.linear**2))
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Non-finite or degenerate law")

    def cf_standardized(self, frequencies):
        """Characteristic function of (T-mean)/sigma; no noncentral division."""
        t = np.asarray(frequencies, float)
        result = np.empty(t.size, complex)
        a, b = self.eigenvalues/self.sigma, self.linear/self.sigma
        for start in range(0, t.size, 256):
            u = t[start:start+256, None]
            denominator = 1-2j*u*a
            log_phi = (-1j*u*a-.5*np.log(denominator)
                       -.5*u*u*b*b/denominator).sum(axis=1)
            result[start:start+256] = np.exp(log_phi)
        return result

    def tail_bound(self, cutoff):
        """Absolute CDF integral remainder bound using quadratic eigenvalues.

        |phi(t)| <= product(2|a_i|t)^(-1/2). Minimize over leading k factors.
        The omitted noncentral exponential has magnitude <=1.
        """
        a = np.sort(np.abs(self.eigenvalues/self.sigma))[::-1]
        a = a[a > 0]
        if len(a):
            k = np.arange(1, len(a)+1)
            logs = -.5*np.cumsum(np.log(2*a))-.5*k*np.log(cutoff)-np.log(k/2)-np.log(np.pi)
            return float(np.exp(np.min(logs)))
        # Pure linear case: exp(-t^2/2), integral exp(-t^2/2)/t <= exp(-U^2/2)/U^2.
        return float(np.exp(-cutoff**2/2)/(np.pi*cutoff**2))

    def evaluate(self, values):
        """Return density, SF, and refinement diagnostics (not a rigorous quad error).

        Gil-Pelaez inversion of the CF, composite Simpson and mesh halving.
        A bound controls CDF truncation; mesh agreement is a numerical check only.
        """
        x = (np.asarray(values, float)-self.mean)/self.sigma
        cutoff = 32.
        while self.tail_bound(cutoff) > 1e-9:
            cutoff *= 2
            if cutoff > 4096:
                raise ArithmeticError("CF decay too slow for this pilot integrator")
        intervals = int(np.ceil(cutoff/min(.025, np.pi/(12*(1+np.max(np.abs(x)))))/2))*2
        previous = None
        for refinement in range(4):
            t = np.linspace(0, cutoff, intervals+1)
            phi = self.cf_standardized(t)
            pdf, sf = np.empty(x.size), np.empty(x.size)
            for start in range(0, x.size, 32):
                y = x[start:start+32]
                product = np.exp(-1j*t[:, None]*y)*phi[:, None]
                pdf[start:start+32] = simpson(product.real, x=t, axis=0)/(np.pi*self.sigma)
                integrand = np.empty(product.shape, float)
                integrand[1:] = product.imag[1:]/t[1:, None]
                integrand[0] = -y  # standardized CF has zero first moment
                sf[start:start+32] = .5+simpson(integrand, x=t, axis=0)/np.pi
            if previous is not None:
                error = max(np.max(np.abs(sf-previous[1])), self.sigma*np.max(np.abs(pdf-previous[0])))
                if error < 2e-7:
                    if np.min(sf) < -2e-7 or np.max(sf) > 1+2e-7 or np.min(pdf)*self.sigma < -2e-7:
                        raise ArithmeticError("Inversion produced invalid probabilities")
                    return pdf, sf, dict(cutoff=cutoff, intervals=intervals,
                        cdf_truncation_bound=self.tail_bound(cutoff), mesh_difference=float(error))
            previous = (pdf, sf)
            intervals *= 2
        raise ArithmeticError("CF inversion mesh did not converge")


def from_hypotheses(pairs, generator_index):
    constant, eigenvalues, linear = 0., [], []
    for null, tested in pairs:
        generated = (null, tested)[generator_index]
        lower = cholesky(generated.covariance, lower=True)
        f3, f4 = cho_factor(null.covariance, lower=True), cho_factor(tested.covariance, lower=True)
        o3, o4 = generated.mean-null.mean, generated.mean-tested.mean
        r3, r4 = cho_solve(f3, o3), cho_solve(f4, o4)
        constant += o4@r4-o3@r3
        matrix = lower.T@(cho_solve(f4, lower)-cho_solve(f3, lower))
        matrix = (matrix+matrix.T)/2
        weights, rotation = eigh(matrix)
        eigenvalues.extend(weights)
        linear.extend(rotation.T@(2*lower.T@(r4-r3)))
    return QuadraticLaw(constant, eigenvalues, linear)
