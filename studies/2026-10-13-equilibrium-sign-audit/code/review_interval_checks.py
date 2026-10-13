"""Independent dummy rational checks of enclosure components. No model calls."""
import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import platform
import numpy as np
import intervals as iv


def frac(x):
    return F.from_float(float(x))


def exact_map(h, w, a, removed):
    return [F(0) if i == removed else max(F(0), a[i] + sum(w[i][j] * h[j] for j in range(2))) for i in range(2)]


def exact_equilibrium(w, a, removed):
    """Enumerate the four active sets with a closed-form two-by-two solve."""
    found = set()
    for active in itertools.product((False, True), repeat=2):
        if removed >= 0 and active[removed]:
            continue
        if active == (False, False):
            h = (F(0), F(0))
        elif active == (True, False):
            h = (a[0] / (1 - w[0][0]), F(0))
        elif active == (False, True):
            h = (F(0), a[1] / (1 - w[1][1]))
        else:
            a00, a01, a10, a11 = 1-w[0][0], -w[0][1], -w[1][0], 1-w[1][1]
            determinant = a00*a11-a01*a10
            h = ((a[0]*a11-a01*a[1])/determinant, (a00*a[1]-a[0]*a10)/determinant)
        if tuple(exact_map(h, w, a, removed)) == h:
            found.add(h)
    assert len(found) == 1
    return next(iter(found))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    def check(ok, label):
        counts[label] = counts.get(label, 0) + 1
        assert bool(ok), label
    def enclosure(lo, hi, value, label):
        check(frac(lo) <= value <= frac(hi), label)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    record = dict(status='running', scientific_model_outputs_collected=False,
                  interval_code_sha256=sha(iv.__file__), checker_code_sha256=sha(__file__),
                  python=platform.python_version(), numpy=np.__version__)
    try:
        tiny = float(np.nextafter(0., 1.))
        intervals = [(-tiny, tiny), (0., 0.), (-2., 3.), (.1, np.nextafter(.1, np.inf)),
                     (-2.**-100, 2.**-100), (-2.**100, 2.**100)]
        for a, b in itertools.product(intervals, repeat=2):
            for func, operation, label in [(iv.add, lambda x,y:x+y, 'interval_add'),
                                          (iv.sub, lambda x,y:x-y, 'interval_subtract'),
                                          (iv.mul, lambda x,y:x*y, 'interval_multiply')]:
                bounds = func(a, b)
                for x,y in itertools.product(a,b):
                    enclosure(*bounds, operation(frac(x),frac(y)), label)
        for a in intervals:
            bounds = iv.relu(a)
            for x in a:
                enclosure(*bounds, max(F(0),frac(x)), 'relu')
            check(frac(iv.abs_upper(a)) >= max(abs(frac(x)) for x in a), 'absolute_upper')
        rng = np.random.Generator(np.random.PCG64(2026101602))
        for trial in range(32):
            lo = rng.normal(size=7)
            hi = lo + np.abs(rng.normal(size=7))
            bounds = iv.sum_interval((lo,hi))
            enclosure(*bounds, sum(map(frac,lo)), 'ordered_sum_lower')
            enclosure(*bounds, sum(map(frac,hi)), 'ordered_sum_upper')
            values = np.abs(lo)
            check(frac(iv.nonnegative_sum_upper(values)) >= sum(map(frac,values)), 'nonnegative_sum_upper')
        max_q = 0.
        for trial in range(40):
            w = rng.integers(-8, 9, size=(2,2)).astype(float) / 32.
            if trial == 0:
                w = np.diag([1.-2.**-30, .5])
            x = rng.normal(size=(1,3))
            B = rng.normal(size=(2,3))
            bias = rng.normal(size=2)
            readout = rng.normal(size=(3,2))
            if trial == 1:
                readout[0] = [2.**52, -2.**52]
                readout[1] = [np.nextafter(2.**52,np.inf), np.nextafter(-2.**52,np.inf)]
            readbias = rng.normal(size=3)
            injection = iv.affine_point(x,B,bias)
            wr = [[frac(v) for v in row] for row in w]
            ar = [frac(bias[i])+sum(frac(B[i,j])*frac(x[0,j]) for j in range(3)) for i in range(2)]
            q = iv.contraction_upper(w)
            max_q = max(max_q,q)
            exact_q = max(sum(abs(v) for v in row) for row in wr)
            check(frac(q) >= exact_q, 'contraction_norm')
            denominator = iv.down(1.-q)
            check(F(0) < frac(denominator) <= 1-frac(q), 'denominator_lower')
            base_target = exact_equilibrium(wr,ar,-1)
            base = rng.normal(size=(1,2))
            for removed in range(2):
                edit_target = exact_equilibrium(wr,ar,removed)
                edit = rng.normal(size=(1,2))
                values = iv.effect_interval(base,edit,w,injection,np.array([removed]),readout,
                                            np.array([0]),np.array([1]),q)
                c = [frac(readout[0,j])-frac(readout[1,j]) for j in range(2)]
                exact_effect = sum(c[j]*(base_target[j]-edit_target[j]) for j in range(2))
                enclosure(values['lower'][0],values['upper'][0],exact_effect,'equilibrium_effect')
                center = sum(c[j]*(frac(base[0,j])-frac(edit[0,j])) for j in range(2))
                enclosure(values['center_lower'][0],values['center_upper'][0],center,'effect_center')
                norm = sum(map(abs,c))
                check(frac(values['coefficient_norm_upper'][0]) >= norm,'coefficient_norm')
                residuals = []
                for state,target,k,rem in [(base,base_target,'residual_base_upper',-1),
                                            (edit,edit_target,'residual_edit_upper',removed)]:
                    hs = list(map(frac,state[0]))
                    mh = exact_map(hs,wr,ar,rem)
                    residual = max(abs(a-b) for a,b in zip(mh,hs))
                    residuals.append(residual)
                    check(frac(values[k][0]) >= residual,'residual_norm')
                    er = iv.state_error_upper(values[k][0],q)
                    check(frac(er) >= max(abs(a-b) for a,b in zip(hs,target)),'state_error')
                check(frac(values['radius'][0]) >= norm*sum(residuals)/(1-exact_q),'effect_radius')
                bounds,rr,er = iv.logits_interval(base,w,injection,readout,readbias,q)
                for label in range(3):
                    exact_logit = sum(frac(readout[label,j])*base_target[j] for j in range(2))+frac(readbias[label])
                    enclosure(bounds[0][0,label],bounds[1][0,label],exact_logit,'equilibrium_logits')
        for name, operation in [
            ('reject_noncontractive', lambda:iv.contraction_upper(np.eye(2))),
            ('reject_zero_denominator',lambda:iv.positive_division_upper(1.,0.)),
            ('reject_infinite_input',lambda:iv.point(np.inf)),
            ('reject_reversed_interval',lambda:iv.valid((1.,0.))),
            ('reject_upper_overflow',lambda:iv.up(np.finfo(float).max))]:
            caught = False
            try:
                with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                    operation()
            except (AssertionError,FloatingPointError):
                caught = True
            check(caught,name)
        check(max_q > .999999,'near_one_contraction_exercised')
        record.update(status='passed',maximum_q_upper=max_q)
    except BaseException as e:
        record.update(status='failed',error_type=type(e).__name__,error=str(e))
        raise
    finally:
        record['checks_by_type'] = counts
        record['check_count'] = sum(counts.values())
        args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
