"""
Relative-error minimax piecewise-linear approximation of g(x) = x**r,
0 < r < 1, on an interval [a, b] with 0 < a < b.

Minimises  E = max |log(p(x)/x**r)|: p stays inside the multiplicative band
exp(-E) <= p/g <= exp(+E).

minimax_ratio_pwl() is the single entry point; it returns a Fit that carries
the coefficients, a sympy Piecewise, evaluation and verification.  Its
`integer` flag selects between two genuinely different problems.

CONTINUOUS (integer=False)
--------------------------
p is a continuous piecewise-linear function of a real x.  Everything follows
from scale covariance: x -> lam*x, y -> lam**r * y maps lines to lines and
leaves log(p/g) alone, so a segment's optimality condition depends only on
the RATIO of its endpoints, never on where it sits.  The alternation forces
log(p/g) = +E at every free knot, so each segment type has one characteristic
ratio fixed by E alone -- rho(E) for a generic segment, sig_L(E) / sig_R(E)
when an endpoint is pinned exact -- and the problem collapses to a span
equation

    sig_L^[pin_left] * rho^(N - pins) * sig_R^[pin_right]  =  b / a

solved for E by bisection.  Unpinned it is closed form: rho = (b/a)**(1/N).
Knots come out geometric, cost depends on b/a only through log(b/a)/N
(decades of dynamic range, not absolute width), and the answer is invariant
under r <-> 1-r.

INTEGER (integer=True)
----------------------
x takes only integer values, so continuity between segments is a constraint
with no payoff and p becomes N independent affine blocks over consecutive
runs of integers -- a table of (slope, intercept) selected by index.  This
is strictly better than the continuous fit and qualitatively different: a
block of 1 or 2 integers fits exactly, so E = 0 as soon as the integer count
is at most 2N.  The gain over the continuous fit is large for short ranges
(4x at [1, 9]) and fades as blocks come to hold many integers.

The integer count never enters the running time, so b may be astronomically
large.  Two facts about g do that: exp(E)*k**r - s*k is concave in k, so its
minimum over a block is at an ENDPOINT and the upper-band constraints
collapse from N to two; and exp(-E)*k**r is concave, so the binding lower
constraint is a tangency located by root-finding rather than scanning.
Block feasibility is then O(1) plus one root solve, the greedy takes maximal
blocks (feasibility is monotone in length, so greedy minimises the count),
and E comes from bisection.

Both modes return a Fit.  Beware that an integer Fit is discontinuous at
block boundaries by design: evaluating it at a non-integer inside [a, b] is
meaningless, so treat it as a table indexed by x.
"""

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np


# ruff: noqa: E731

# ---------------------------------------------------------------------------
# continuous internals
# ---------------------------------------------------------------------------


def _cont_tools(r):
    C = (1.0 - r) ** (1.0 - r) * r**r

    def min_log_ratio(lam_u, t, lam_v):
        """
        log of  min_{x in [1,t]} line(x)/x**r  for the segment joining
        (1, exp(lam_u)) to (t, exp(lam_v) * t**r).

        min_{x>0} (a0 + s*x)/x**r = a0**(1-r) * s**r / C when a0, s > 0,
        attained at x* = r*a0/((1-r)*s); otherwise the min is at an endpoint,
        so clamp rather than trusting the closed form.
        """
        yu, yv = np.exp(lam_u), np.exp(lam_v) * t**r
        s = (yv - yu) / (t - 1.0)
        a0 = yu - s
        best = min(lam_u, lam_v)
        if a0 > 0.0 and s > 0.0:
            xstar = r * a0 / ((1.0 - r) * s)
            if 1.0 < xstar < t:
                best = min(best, (1.0 - r) * np.log(a0) + r * np.log(s) - np.log(C))
        return best

    def seg_ratio(E, lam_u, lam_v):
        """Endpoint ratio t > 1 of a segment whose dip is exactly -E."""
        g = lambda v: min_log_ratio(lam_u, np.exp(v), lam_v) + E
        hi = 1.0
        while g(hi) > 0.0:
            hi *= 2.0
            if hi > 700.0:
                raise RuntimeError("E too large to realise")
        lo = 0.0
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if g(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return np.exp(0.5 * (lo + hi))

    return C, min_log_ratio, seg_ratio


def _solve_cont(r, N, a, b, pin_left, pin_right, iters=300):
    """Continuous solve; returns (xs, ys, E) with xs, ys of length N+1."""
    if not 0.0 < r < 1.0:
        raise ValueError("need 0 < r < 1")
    if not 0.0 < a < b:
        raise ValueError("need 0 < a < b")
    if N < 1:
        raise ValueError("need N >= 1")
    n_pin = int(pin_left) + int(pin_right)
    if N < n_pin:
        raise ValueError("need N >= number of pinned endpoints")

    C, min_log_ratio, seg_ratio = _cont_tools(r)
    R = b / a
    logR = np.log(R)
    n_free = N - n_pin

    # --- fully unpinned: closed form, no root finding ---------------------
    if n_pin == 0:
        rho = R ** (1.0 / N)
        # both ends lifted by k=exp(E) scales the line by k, so the dip is
        # k*Psi(rho) and the condition k*Psi = 1/k gives E = -log(Psi)/2
        E = -0.5 * min_log_ratio(0.0, rho, 0.0)
        xs = a * rho ** np.arange(N + 1)
        xs[-1] = b
        return xs, np.exp(E) * xs**r, E

    # --- N segments, at least one pinned end: shoot on E ------------------
    def log_span(E):
        tot = n_free * np.log(seg_ratio(E, E, E)) if n_free else 0.0
        if pin_left:
            tot += np.log(seg_ratio(E, 0.0, E))
        if pin_right:
            tot += np.log(seg_ratio(E, E, 0.0))
        return tot

    lo, hi = 0.0, 1.0
    while log_span(hi) < logR:  # span grows with E
        hi *= 2.0
        if hi > 500.0:
            raise RuntimeError("failed to bracket E")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if log_span(mid) < logR:
            lo = mid
        else:
            hi = mid
    E = 0.5 * (lo + hi)

    ratios = []
    if pin_left:
        ratios.append(seg_ratio(E, 0.0, E))
    ratios += [seg_ratio(E, E, E)] * n_free
    if pin_right:
        ratios.append(seg_ratio(E, E, 0.0))

    xs = a * np.concatenate(([1.0], np.cumprod(ratios)))
    xs[-1] = b
    ys = np.exp(E) * xs**r
    if pin_left:
        ys[0] = a**r
    if pin_right:
        ys[-1] = b**r
    return xs, ys, E


# -------------------------------------------------------------------------
# integer internals
# -------------------------------------------------------------------------


def _int_tools(r):
    def gp(k):
        return float(k) ** r

    def logratio(c, s, k):
        """log(line(k) / k**r)."""
        return math.log(c + s * k) - r * math.log(k)

    def _real_root(h, klo, khi, decreasing):
        """Bisect a monotone h on [klo, khi]; returns the real crossing."""
        lo, hi = float(klo), float(khi)
        if (h(lo) <= 0.0) == decreasing:
            return lo
        if (h(hi) >= 0.0) == decreasing:
            return hi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if (h(mid) > 0.0) == decreasing:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _best_int(f, klo, khi, root, maximize):
        """Extremum of a unimodal integer sequence given the real argext."""
        cands = {klo, khi}
        for k in (math.floor(root), math.ceil(root)):
            if klo <= k <= khi:
                cands.add(k)
        vals = [f(k) for k in cands]
        return max(vals) if maximize else min(vals)

    def far_max(C, i, klo, khi, el):
        """
        max over integers k in [klo, khi] of (el*k**r - C)/(k - i).

        The sequence is unimodal, but a binary search on f(m) vs f(m+1) is
        useless for large k: consecutive values differ by far less than the
        float resolution of f, so the comparison is noise and the search
        walks off in a random direction.  Locate the extremum in the
        continuum instead -- phi'(k)*(k-i) = phi(k) - C is monotone and
        cancellation-free -- then round to the neighbouring integers.
        """
        h = lambda k: el * r * k ** (r - 1.0) * (k - i) - (el * k**r - C)
        root = _real_root(h, klo, khi, True)
        f = lambda k: (el * gp(k) - C) / (k - i)
        return _best_int(f, klo, khi, root, True)

    def near_min(C, j, klo, khi, el):
        """min over integers k in [klo, khi] of (C - el*k**r)/(j - k)."""
        h = lambda k: -el * r * k ** (r - 1.0) * (j - k) + (C - el * k**r)
        root = _real_root(h, klo, khi, False)
        f = lambda k: (C - el * gp(k)) / (j - k)
        return _best_int(f, klo, khi, root, False)

    def block_line(i, j, E, pin_l, pin_r):
        """
        A line fitting x**r within exp(+-E) at every integer of [i, j], or
        None.  Returns (c, s, p, yp): line(x) = c + s*x, pivoting exactly
        through (p, yp).  The caller can re-form c = yp - s*p at higher
        precision, which matters because c is about (1-r)*yp, so the
        subtraction sheds roughly log10(1/(1-r)) digits in doubles.
        """
        eu, el = math.exp(E), math.exp(-E)
        gi, gj = gp(i), gp(j)
        s_nat = (gj - gi) / (j - i) if j > i else 0.0

        def pick(s_lo, s_hi):
            """
            Choose a slope from the feasible interval [s_lo, s_hi].

            Every slope in the interval satisfies the band, so this is free
            to optimise for conditioning -- and it must.  For a short block
            at large i the interval is enormous (hundreds wide, while the
            sensible slope is ~1e-7), so a midpoint would give an intercept
            of ~1e12 and then c + s*x cancels catastrophically, leaving the
            line's actual values dominated by rounding.  Clamping g's own
            chord slope into the interval keeps |c| ~ (1-r)*y instead.
            """
            return min(max(s_nat, s_lo), s_hi)

        def ok(cand):
            """
            Accept a candidate only if it really does fit.

            The unimodal searches below merely PROPOSE a slope; over huge
            ranges the differences they evaluate are cancellation-dominated
            (for k near j, float(k) and float(j) can coincide outright), and
            unimodality then fails numerically and a bogus slope comes back.
            _block_err is exact and O(1), so validating here turns any such
            failure into a conservative rejection instead of a wrong answer.
            """
            if cand is None:
                return None
            c, s = cand[0], cand[1]
            if c + s * i <= 0.0 or c + s * j <= 0.0:
                return None
            # absolute floor: at E near machine epsilon the recomputed
            # error is itself only good to ~1e-16, so a purely relative
            # test would reject exact 2-point fits.  Genuine violations
            # are orders of magnitude larger, not marginal.
            return cand if _block_err(r, c, s, i, j) <= E * (1 + 1e-9) + 1e-14 else None

        if i == j:
            return (gi, 0.0, i, gi)

        if pin_l and pin_r:  # two points fix the line
            s = (gj - gi) / (j - i)
            c = gi - s * i
            return ok((c, s, i, gi))

        if pin_l or pin_r:
            # Line through one exact endpoint; the free slope is boxed in by
            # a lower and an upper bound over the block's integers.  Care is
            # needed about WHICH extremum each is: (concave - const)/(k - p)
            # is unimodal with a MAXIMUM, so its minimum sits at an endpoint
            # of the k-range, and vice versa.  Getting this backwards makes a
            # binary search return a non-extremal point and silently loosens
            # the box.
            if pin_l:
                p, gp_ = i, gi
                lof = lambda k: (el * gp(k) - gp_) / (k - p)  # unimodal max
                upf = lambda k: (eu * gp(k) - gp_) / (k - p)  # unimodal max
                klo, khi = i + 1, j
                s_lo = far_max(gp_, p, klo, khi, el)
                s_hi = min(upf(klo), upf(khi))
            else:
                p, gp_ = j, gj
                lof = lambda k: (gp_ - eu * gp(k)) / (p - k)  # unimodal min
                upf = lambda k: (gp_ - el * gp(k)) / (p - k)  # unimodal min
                klo, khi = i, j - 1
                s_lo = max(lof(klo), lof(khi))
                s_hi = near_min(gp_, p, klo, khi, el)
            if s_lo > s_hi:
                return None
            s = pick(s_lo, s_hi)
            return ok((gp_ - s * p, s, p, gp_))

        # free block.  Raising the line only helps the lower band, so an
        # optimal line touches the upper band at i or at j (fact 1 above).
        upi, upj = eu * gi, eu * gj
        S2 = (upj - upi) / (j - i)
        S1 = far_max(upi, i, i + 1, j, el)
        if S1 <= S2:
            s = pick(S1, S2)
            got = ok((upi - s * i, s, i, upi))
            if got is not None:
                return got
        T1 = near_min(upj, j, i, j - 1, el)
        if S2 <= T1:
            s = pick(S2, T1)
            return ok((upj - s * j, s, j, upj))
        return None

    return gp, logratio, block_line


def _block_err(r, c, s, i, j):
    """
    max |log(line/g)| over the integers of [i, j], in O(1).

    log(c + s*x) - r*log(x) has one interior stationary point, a minimum at
    x* = r*c/((1-r)*s), so the maximum is at an endpoint and the minimum at
    an integer neighbouring x*.
    """

    def lr(k):
        v = c + s * k
        return -math.inf if v <= 0.0 else math.log(v) - r * math.log(k)

    cand = [lr(i), lr(j)]
    if c > 0.0 and s > 0.0:
        xs = r * c / ((1.0 - r) * s)
        for k in (math.floor(xs), math.ceil(xs)):
            if i <= k <= j:
                cand.append(lr(k))
    return max(abs(v) for v in cand)


def _refine_intercept(line, dps=40):
    """Re-form c = yp - s*p at extended precision, then round once."""
    import mpmath as mp

    c, s, p, yp = line
    with mp.workdps(dps):
        return float(mp.mpf(yp) - mp.mpf(s) * mp.mpf(p)), s


def _solve_int(r, N, a, b, pin_left, pin_right, iters=200, E_hi=None):
    """Integer solve; returns (blocks, coeffs, E), blocks inclusive (i, j)."""
    if not 0.0 < r < 1.0:
        raise ValueError("need 0 < r < 1")
    if N < 1:
        raise ValueError("need N >= 1")
    i0, i1 = math.ceil(a), math.floor(b)
    if i0 < 1:
        raise ValueError("need a >= 1 so that log(x) is defined")
    if i0 >= i1:
        raise ValueError("need at least two integers in [a, b]")

    _, _, block_line = _int_tools(r)

    def cover(E):
        """Greedy maximal blocks; returns the block list or None."""
        out, cur = [], i0
        while True:
            if cur > i1:
                return out
            used = len(out)
            if used >= N:
                return None
            pl = pin_left and used == 0
            # can we close out here with the (possibly pinned) final block?
            if block_line(cur, i1, E, pl, pin_right) is not None:
                out.append((cur, i1))
                return out
            if used == N - 1:
                return None
            # Otherwise take the longest feasible block, capped at i1 - 1:
            # the last integer must be covered by a block that has been
            # tested WITH pin_right, which is stricter.  Letting a generic
            # block reach i1 would make cover() non-monotone in E.
            limit = i1 - 1 - cur
            if limit < 1 or block_line(cur, cur + 1, E, pl, False) is None:
                return None
            step = 1
            while (
                2 * step <= limit
                and block_line(cur, cur + 2 * step, E, pl, False) is not None
            ):
                step *= 2
            lo, hi = step, min(2 * step, limit)
            while lo < hi:
                m = (lo + hi + 1) // 2
                if block_line(cur, cur + m, E, pl, False) is not None:
                    lo = m
                else:
                    hi = m - 1
            out.append((cur, cur + lo))
            cur += lo + 1

    if E_hi is None:
        E_hi = math.log(1.0 + 1.0)  # a very loose start
        while cover(E_hi) is None:
            E_hi *= 2.0
            if E_hi > 700.0:
                raise RuntimeError("no feasible E found")
    lo, hi = 0.0, E_hi
    if cover(lo) is not None:
        hi = 0.0  # M <= 2N: exact fit
    else:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if cover(mid) is None:
                lo = mid
            else:
                hi = mid
    blocks = cover(hi)

    # tighten each block on its own: the binding block sets E, the others
    # can usually do better than the level they were merely feasible at
    coeffs, errs = [], []
    for k, (i, j) in enumerate(blocks):
        pl = pin_left and k == 0
        pr = pin_right and k == len(blocks) - 1
        blo, bhi = 0.0, hi
        best = block_line(i, j, bhi, pl, pr)
        for _ in range(iters):
            mid = 0.5 * (blo + bhi)
            if mid <= blo or mid >= bhi:
                break
            got = block_line(i, j, mid, pl, pr)
            if got is None:
                blo = mid
            else:
                bhi, best = mid, got
        c, s = _refine_intercept(best)
        coeffs.append((c, s))
        errs.append(_block_err(r, c, s, i, j))
    return blocks, coeffs, max(errs)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fit:
    """
    A solved approximant.  `bounds` and `coeffs` are the representation both
    modes share: segment k covers bounds[k] and evaluates to c + s*x.  For
    the continuous mode `knots` and `values` additionally give the shared
    segment endpoints (values[k] = p(knots[k])); for the integer mode they
    are None, because adjacent blocks do not meet.
    """

    r: float
    E: float
    integer: bool
    bounds: tuple  # per segment, (lo, hi)
    coeffs: tuple  # per segment, (c, s)
    knots: tuple = None  # continuous only, length N+1
    values: tuple = None  # continuous only, length N+1

    @property
    def n_segments(self):
        return len(self.coeffs)

    @property
    def band(self):
        """(lower, upper) multiplicative bounds on p/g."""
        return math.exp(-self.E), math.exp(self.E)

    def __call__(self, x):
        """
        Evaluate.  Returns nan outside [a, b].  For an integer Fit this is
        only meaningful at integers -- nothing here rounds for you.
        """
        xa = np.asarray(x, dtype=float)
        edges = np.array([hi for _, hi in self.bounds])
        c = np.array([cc for cc, _ in self.coeffs])
        s = np.array([ss for _, ss in self.coeffs])
        k = np.clip(np.searchsorted(edges, xa, side="left"), 0, len(c) - 1)
        out = c[k] + s[k] * xa
        lo, hi = self.bounds[0][0], self.bounds[-1][1]
        out = np.where((xa < lo) | (xa > hi), np.nan, out)
        return out if np.ndim(x) else float(out)

    def extrema(self):
        """
        (sup, peaks, dips) of log(p/g), evaluated at the exact extrema.

        On each segment log(c + s*x) - r*log(x) has a single interior
        stationary point, a minimum at x* = r*c/((1-r)*s), so the maxima sit
        at segment ends and the minima at x* -- or, in integer mode, at the
        integers neighbouring it.  Sampling instead would need absurdly many
        points to resolve the dips and quietly reports the sup too low.
        """
        peaks, dips = [], []
        for (lo, hi), (c, s) in zip(self.bounds, self.coeffs):
            for xv in (lo, hi):
                peaks.append(math.log(c + s * xv) - self.r * math.log(xv))
            if c > 0.0 and s > 0.0:
                xstar = self.r * c / ((1.0 - self.r) * s)
                cands = (
                    [math.floor(xstar), math.ceil(xstar)] if self.integer else [xstar]
                )
                for xv in cands:
                    if lo <= xv <= hi:
                        dips.append(math.log(c + s * xv) - self.r * math.log(xv))
        sup = max(max(abs(v) for v in peaks), max(abs(v) for v in dips))
        return sup, peaks, dips

    @cached_property
    def piecewise(self):
        """
        The approximant as a sympy Piecewise, with default symbol and nan
        outside [a, b].  Call sympy() instead to choose those.

        Lazy rather than eager: solving does not otherwise need sympy, and
        segments_for_tolerance() solves repeatedly in a loop, so building an
        expression every time would be wasted work.
        """
        return self.sympy()

    def sympy(self, x=None, outside="nan"):
        """
        Build a sympy Piecewise.  Coefficients and breakpoints are plain
        double-precision Floats.

        x        symbol to use; defaults to Symbol('x', real=True), or
                 Symbol('x', integer=True) for an integer Fit
        outside  'nan'    -> nan outside [a, b]
                 'clamp'  -> held at the endpoint values
                 'extend' -> end segments extrapolated

        An integer Fit emits integer comparisons (x <= j) and, being a table
        of independent blocks, is discontinuous at the boundaries.
        """
        import sympy as sp

        if outside not in ("nan", "clamp", "extend"):
            raise ValueError("outside must be 'nan', 'clamp' or 'extend'")
        if x is None:
            x = (
                sp.Symbol("x", integer=True)
                if self.integer
                else sp.Symbol("x", real=True)
            )

        F = sp.Float
        n = self.n_segments
        piece = lambda k: F(self.coeffs[k][0]) + F(self.coeffs[k][1]) * x
        # integer blocks are inclusive, so the natural split is `x <= j`;
        # continuous segments share a knot, so it is `x < x_{k+1}`
        if self.integer:
            cut = [x <= self.bounds[k][1] for k in range(n)]
        else:
            cut = [x < self.bounds[k][1] for k in range(n - 1)] + [
                x <= self.bounds[-1][1]
            ]

        pieces = []
        if outside == "extend":
            pieces += [(piece(k), cut[k]) for k in range(n - 1)]
            pieces.append((piece(n - 1), sp.true))
        else:
            lo, hi = self.bounds[0][0], self.bounds[-1][1]
            if outside == "nan":
                out_lo = out_hi = sp.nan
            else:
                out_lo, out_hi = F(self(lo)), F(self(hi))
            pieces.append((out_lo, x < lo))
            pieces += [(piece(k), cut[k]) for k in range(n)]
            pieces.append((out_hi, sp.true))
        return sp.Piecewise(*pieces)


def minimax_ratio_pwl(r, N, a, b, pin_left=True, pin_right=True, integer=False):
    """
    Solve with at most N segments and return a Fit.

    pin_left / pin_right force p exact at that endpoint (log-error 0 there
    instead of +E); both default to True, which needs N >= 2.  Unpinning
    is worth a couple of percent in E and, in continuous mode, makes the
    solution closed form.

    integer=False  continuous p, minimax over all real x in [a, b]
    integer=True   minimax over the integers of [a, b] only; see the module
                   docstring for why the result is then discontinuous

    This is the only constructor.  The returned Fit carries everything
    derived from the solution: fit.piecewise (or fit.sympy(...) to choose
    the symbol and out-of-domain behaviour), fit(x) to evaluate, and
    fit.extrema() to verify.
    """
    if integer:
        blocks, coeffs, E = _solve_int(r, N, a, b, pin_left, pin_right)
        return Fit(
            r=float(r),
            E=float(E),
            integer=True,
            bounds=tuple((int(i), int(j)) for i, j in blocks),
            coeffs=tuple((float(c), float(s)) for c, s in coeffs),
        )

    xs, ys, E = _solve_cont(r, N, a, b, pin_left, pin_right)
    return Fit(
        r=float(r),
        E=float(E),
        integer=False,
        bounds=tuple(zip(map(float, xs[:-1]), map(float, xs[1:]))),
        coeffs=_cont_coeffs(xs, ys),
        knots=tuple(map(float, xs)),
        values=tuple(map(float, ys)),
    )


def segments_for_tolerance(
    r,
    tol,
    a,
    b,
    pin_left=True,
    pin_right=True,
    integer=False,
    relative="log",
    nmax=100000,
):
    """
    Smallest N meeting a tolerance.  relative='log' reads tol as a bound on
    |log(p/g)|; 'band' reads it as a bound on |p/g - 1|.
    """
    if relative not in ("log", "band"):
        raise ValueError("relative must be 'log' or 'band'")
    target = tol if relative == "log" else math.log1p(tol)
    n = max(1, int(pin_left) + int(pin_right))
    while n <= nmax:
        if minimax_ratio_pwl(r, n, a, b, pin_left, pin_right, integer).E <= target:
            return n
        n += 1
    raise RuntimeError("tolerance not reachable within nmax")


def _cont_coeffs(xs, ys):
    """
    Per-segment (c, s) for a continuous solution, formed in mpmath.

    A piece is written c + s*x rather than y0 + s*(x - x0): that is the
    better form to evaluate, since c and s are both positive on [a, b] so
    the sum never cancels.  Forming c is the delicate part --

        c = (y0*x1 - y1*x0) / (x1 - x0)

    is a difference of nearly equal products, and c works out to roughly
    (1-r)*y0, so in doubles it sheds about log10(1/(1-r)) digits, two of
    them at r = 0.99.  Extended precision here and a single rounding gives
    correctly rounded doubles; the solve itself stays in double.
    """
    import mpmath as mp

    out = []
    with mp.workdps(40):
        for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
            X0, X1 = mp.mpf(float(x0)), mp.mpf(float(x1))
            Y0, Y1 = mp.mpf(float(y0)), mp.mpf(float(y1))
            out.append(
                (float((Y0 * X1 - Y1 * X0) / (X1 - X0)), float((Y1 - Y0) / (X1 - X0)))
            )
    return tuple(out)


def verify_expr(expr, x, r, dps=40):
    """
    Check a built expression at its exact extrema rather than on a grid.

    For a piece a0 + s*x the log-ratio log(p/g) has a single interior
    stationary point, at x* = r*a0/((1-r)*s), so every peak and dip can be
    hit exactly.  Grid sampling would need absurdly many points to resolve
    the dips, and quietly reports the sup too low.

    Returns (sup, peaks, dips) as floats.  Needs outside='nan' or 'clamp' so
    the domain edges can be recovered from the expression.
    """
    import mpmath as mp
    import sympy as sp

    lin = [(e, c) for e, c in expr.args if e.has(x)]
    bnds = [(float(c.rhs) if c is not sp.true else None) for e, c in lin]
    edge = [c.rhs for e, c in expr.args if not e.has(x) and c is not sp.true]
    if not edge:
        raise ValueError("need outside='nan' or 'clamp' to locate the domain")

    with mp.workdps(dps):
        rm = mp.mpf(float(r))
        left = mp.mpf(float(edge[0]))
        peaks, dips = [], []
        for k, (e, _) in enumerate(lin):
            s, a0 = (mp.mpf(float(c)) for c in sp.Poly(e, x).all_coeffs())
            u = left if k == 0 else mp.mpf(bnds[k - 1])
            v = mp.mpf(bnds[k]) if bnds[k] is not None else mp.inf
            for xv in (u, v):
                if xv != mp.inf:
                    peaks.append(mp.log(a0 + s * xv) - rm * mp.log(xv))
            xstar = rm * a0 / ((1 - rm) * s)
            if u < xstar < v:
                dips.append(mp.log(a0 + s * xstar) - rm * mp.log(xstar))
        sup = max(max(abs(p) for p in peaks), max(abs(d) for d in dips))
    return float(sup), [float(p) for p in peaks], [float(d) for d in dips]


if __name__ == "__main__":
    import sympy as sp

    r, a, b = 0.4, 3.0, 4096.0
    print(f"g(x) = x**{r} on [{a}, {b}]   ({math.log10(b / a):.2f} decades)\n")
    print(f"{'N':>3} {'continuous':>12} {'unpinned':>12} {'integer':>12} {'band':>9}")
    for N in (2, 4, 8, 16, 32):
        Ec = minimax_ratio_pwl(r, N, a, b).E
        Ef = minimax_ratio_pwl(r, N, a, b, False, False).E
        Ei = minimax_ratio_pwl(r, N, a, b, integer=True).E
        print(f"{N:>3} {Ec:>12.5e} {Ef:>12.5e} {Ei:>12.5e} {math.expm1(Ec):>8.3%}")

    n = segments_for_tolerance(r, 1e-3, a, b, relative="band")
    ni = segments_for_tolerance(r, 1e-3, a, b, integer=True, relative="band")
    print(f"\nN for a 0.1% band:  continuous {n}, integer {ni}")

    print("\nsmall integer ranges (r=0.4, N=4):")
    for hi in (8, 9, 13, 64):
        f = minimax_ratio_pwl(r, 4, 1, hi, integer=True)
        c = minimax_ratio_pwl(r, 4, 1.0, float(hi))
        print(
            f"  [1, {hi}]  integer {f.E:.5e}   continuous {c.E:.5e}"
            f"   gain {c.E / f.E if f.E > 1e-14 else float('inf'):>7.2f}"
        )

    fit = minimax_ratio_pwl(r, 3, a, b)
    print("\ncontinuous, N=3 -- fit.piecewise:")
    sp.pprint(sp.N(fit.piecewise, 6))
    sup, peaks, dips = fit.extrema()
    print(f"  E = {fit.E:.17g}   sup at exact extrema = {sup:.17g}")
    print(f"  dip spread = {max(dips) - min(dips):.2e}")
    print(
        f"  fit(100.0) = {fit(100.0):.10f}   band = "
        f"({fit.band[0]:.6f}, {fit.band[1]:.6f})"
    )

    fit = minimax_ratio_pwl(r, 3, 1, 100, integer=True)
    print("\ninteger, N=3 on [1, 100] -- fit.piecewise:")
    sp.pprint(sp.N(fit.piecewise, 6))
    print(f"  blocks {fit.bounds}")
    print(f"  E = {fit.E:.6e}   sup at exact extrema = {fit.extrema()[0]:.6e}")
