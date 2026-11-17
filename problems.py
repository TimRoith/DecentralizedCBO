import numpy as np
import cbx.objectives

class _ProblemFactory:
    """Picklable f_factory for CentralizedCBOAsync worker processes."""
    def __init__(self, problem_cls, init_kwargs):
        self.problem_cls = problem_cls
        self.init_kwargs = init_kwargs
    def __call__(self):
        return self.problem_cls(**self.init_kwargs).generate()['objective']

class NoisyObjective:
    """Wrap a cbx objective to add Gaussian noise to the output."""
    def __init__(self, f, sigma = 0.1):
        self.f     = f
        self.sigma = sigma

    def __call__(self, x):
        eval = self.f(x)
        eval += self.sigma * np.random.randn(*eval.shape)
        return eval

class HeavyTailedObjective:
    """f(x) plus Student-t noise, as a stress test the Gaussian model cannot pose.

    The consensus weight is exp(-alpha*(f + eps)), so one large negative draw
    takes essentially all of the weight and the consensus becomes that
    particle. Gaussian tails make that vanishingly rare; Student-t makes it
    routine. The point of interest is structural: with one node a corrupted
    consensus is the only consensus, while with N nodes it is one of N and the
    mixing dilutes it -- so decentralisation should help here *because* it is
    decentralised. It can also cut the other way, since z propagates the
    corruption to neighbours, which is the empirical question.

    `scale` is the t distribution's scale parameter, NOT its standard
    deviation: for df > 2 the sd is scale*sqrt(df/(df-2)), and for df <= 2
    there is no finite variance at all (df = 1 is Cauchy, with no mean).
    """

    def __init__(self, f, scale=0.1, df=3.0):
        self.f, self.scale, self.df = f, scale, df

    def __call__(self, x):
        e = self.f(x)
        return e + self.scale * np.random.standard_t(self.df, size=np.shape(e))

class Problem:
    """Base class. Subclasses must set d, domain, f_min and implement generate()."""
    d:      int
    domain: tuple              # (low, high) scalars, used for initialisation
    f_min:  float              # known global minimum value
    name:   str    = 'default' # Name of the method
    safety_box: float = None   # explicit clip bound; None -> box_margin * domain
    box_margin: float = 4.0    # default clip, as a multiple of the domain half-width
    x_min:  float  = 0.0       # the base objective's minimiser, as c in c*ones(d)

    def generate(self):
        """Return {'objective': f} with a numpy-compatible f. Used by the cbx pipeline."""
        raise NotImplementedError

    def sample(self, n_nodes, n_particles):
        """Uniform sample over domain, returns (n_nodes, n_particles, d)."""
        low, high = self.domain
        return low + (high - low) * np.random.rand(n_nodes, n_particles, self.d)

    def post_process(self, dyn):
        """Clip particles to the safety box after each CBO step.
        """
        b = self.safety_box or self.box_margin * max(abs(self.domain[0]),
                                                     abs(self.domain[1]))
        np.nan_to_num(dyn.x, copy=False, nan=self.domain[1])
        np.clip(dyn.x, -b, b, out=dyn.x)

    def sample_offset(self, rng, frac=0.8):
        """Offset placing the minimiser at a random point instead of at x_min.

        Most of these objectives have their minimiser at the origin, and CBO
        started uniformly on a symmetric domain already has its consensus
        there, so an unshifted benchmark flatters it. `frac` keeps the moved
        minimiser inside the middle of the domain rather than in a corner.
        """
        low, high = self.domain
        mid, half = (low + high) / 2, (high - low) / 2 * frac
        target = mid + half * (2 * rng.random(self.d) - 1)
        return target - self.x_min

    def make_factory(self):
        """Return a picklable f_factory for CentralizedCBOAsync worker processes."""
        return _ProblemFactory(type(self), {'d': self.d})

class Shifted:
    """`f` with its minimiser moved by `offset`. Module level, so it pickles."""

    def __init__(self, f, offset):
        self.f, self.offset = f, offset

    def __call__(self, x):
        return self.f(x - self.offset)

class RastriginProblem(Problem):
    def __init__(self, d=10, b=2):
        self.d, self.b = d, b
        self.domain = (-5.12, 5.12)
        self.f_min  = 0.0
        self.x_min  = b          # b != 0 on purpose: keeps the optimum off the origin

    def generate(self):
        return {'objective': cbx.objectives.Rastrigin(b=self.b)}

    def make_factory(self):
        return _ProblemFactory(RastriginProblem, {'d': self.d, 'b': self.b})

class AckleyProblem(Problem):
    def __init__(self, d=10):
        self.d      = d
        self.domain = (-32.768, 32.768)
        self.f_min  = 0.0

    def generate(self):
        return {'objective': cbx.objectives.Ackley(minimum=np.zeros(self.d))}

class _StyblinskiTang:
    def __call__(self, x):
        return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=-1)

class StyblinskiTangProblem(Problem):
    def __init__(self, d=10):
        self.d      = d
        self.domain = (-5.0, 5.0)
        self.f_min  = -39.166165703771 * d
        self.x_min  = -2.903534024325

    def generate(self):
        return {'objective': _StyblinskiTang()}

class _Alpine:
    """Alpine-1: sum of |x_i * sin(x_i) + 0.1 * x_i|."""
    def __call__(self, x):
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=-1)

class AlpineProblem(Problem):
    def __init__(self, d=10):
        self.d      = d
        self.domain = (-10.0, 10.0)
        self.f_min  = 0.0

    def generate(self):
        return {'objective': _Alpine()}

class _LunacekBiRastrigin:
    """Lunacek bi-Rastrigin: two funnels, the deeper one the smaller.

        f(x) = min( sum (x_i - mu0)^2,  d + s * sum (x_i - mu1)^2 )
               + 10 * sum (1 - cos(2 pi (x_i - mu0)))

    The point of this function is that a method which identifies the dominant
    funnel and follows it lands in the WRONG one. The mu1 funnel is wider and
    captures far more of the domain, while the global optimum sits in the
    narrower mu0 funnel, and a Rastrigin ripple on top keeps either from being
    descended locally.

    That is a structural argument, not a tuned one: it penalises adapting a
    single search distribution to the apparent global trend, which is exactly
    what covariance adaptation does.

    References:
      Lunacek, Whitley & Sutton, "The Impact of Global Structure on Search",
      PPSN X (2008), LNCS 5199, pp. 498-507 -- introduces the construction and
      the argument that exploration misleads when local optima are not
      structured predictably.

      Hansen, Finck, Ros & Auger, "Real-Parameter Black-Box Optimization
      Benchmarking 2009: Noiseless Functions Definitions", INRIA RR-6829 --
      adopts it as BBOB f24, in the group "multi-modal functions with weak
      global structure" (f20-f24).

    Note the function is due to Lunacek et al., NOT to Hansen; Hansen et al.
    adopted it into BBOB and defined the grouping. Using it is characterisation
    against an external taxonomy rather than cherry-picking, which is the whole
    reason to prefer it over a function chosen for the result.
    """

    def __init__(self, d):
        self.mu0 = 2.5
        self.s = 1.0 - 1.0 / (2.0 * np.sqrt(d + 20.0) - 8.2)
        self.mu1 = -np.sqrt((self.mu0 ** 2 - 1.0) / self.s)

    def __call__(self, x):
        d = x.shape[-1]
        a = np.sum((x - self.mu0) ** 2, axis=-1)
        b = d + self.s * np.sum((x - self.mu1) ** 2, axis=-1)
        ripple = 10.0 * np.sum(1.0 - np.cos(2.0 * np.pi * (x - self.mu0)), axis=-1)
        return np.minimum(a, b) + ripple

class LunacekProblem(Problem):
    def __init__(self, d=10):
        self.d      = d
        self.domain = (-5.0, 5.0)
        self.f_min  = 0.0
        # the optimum is at mu0 * ones, which is already off the origin
        self.x_min  = 2.5

    def generate(self):
        return {'objective': _LunacekBiRastrigin(self.d)}

class _Katsuura:
    """Katsuura (BBOB f23): rugged at every scale, no exploitable smoothness.

        f(x) = 10/d^2 * prod_i (1 + i * sum_{j=1..32} |2^j x_i - [2^j x_i]| / 2^j)
               ^(10 / d^1.2)  -  10/d^2

    The inner sum measures how far 2^j x_i sits from an integer at 32 dyadic
    scales at once, so the landscape is self-similar and nowhere differentiable
    in the limit. There is no local gradient worth estimating and no covariance
    worth adapting -- which is the structural claim, and it is a different one
    from Lunacek's (deceptive trend) or Styblinski-Tang's (degenerate minima).

    Computed via logs: the product runs over d factors each raised to 10/d^1.2,
    which at d=100 reaches exp(15) before the 10/d^2 prefactor pulls it back.
    """

    def __call__(self, x):
        d = x.shape[-1]
        j = 2.0 ** np.arange(1, 33)
        z = x[..., None] * j                       # (..., d, 32)
        s = np.sum(np.abs(z - np.round(z)) / j, axis=-1)        # (..., d)
        i = np.arange(1, d + 1, dtype=float)
        # log-space: the direct product overflows well before d=100
        logp = (10.0 / d ** 1.2) * np.sum(np.log1p(i * s), axis=-1)
        return (10.0 / d ** 2) * np.exp(logp) - 10.0 / d ** 2

class KatsuuraProblem(Problem):
    def __init__(self, d=10):
        self.d      = d
        self.domain = (-5.0, 5.0)
        self.f_min  = 0.0
        self.x_min  = 0.0

    def generate(self):
        return {'objective': _Katsuura()}

def _mlp(**kwargs):
    """Lazy: keeps torch out of the import path for every other problem."""
    from nn_problems import MLPProblem
    return MLPProblem(**kwargs)

PROBLEMS = {
    'mlp':            _mlp,
    'rastrigin':      RastriginProblem,
    'ackley':         AckleyProblem,
    'styblinski_tang': StyblinskiTangProblem,
    'alpine':         AlpineProblem,
    'lunacek':        LunacekProblem,
    'katsuura':       KatsuuraProblem,
}
