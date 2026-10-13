"""Binary64 enclosures for a contractive ReLU state equation.

Trusted base: IEEE-754 round-to-nearest elementary NumPy float64 operations
and nextafter, with gradual underflow and no fused operations across calls.
The exact target uses the real values of the stored binary64 coefficients.
No BLAS dot product or floating-point reduction produces a certificate.
"""
import numpy as np


def finite(*values):
    for value in values:
        assert np.isfinite(value).all(), 'Nonfinite interval arithmetic.'


def down(value):
    value = np.asarray(value, dtype=np.float64)
    finite(value)
    with np.errstate(under='ignore'):
        result = np.nextafter(value, -np.inf)
    finite(result)
    return result


def up(value):
    value = np.asarray(value, dtype=np.float64)
    finite(value)
    with np.errstate(under='ignore'):
        result = np.nextafter(value, np.inf)
    finite(result)
    return result


def valid(interval):
    lo, hi = interval
    finite(lo, hi)
    assert np.all(lo <= hi), 'Invalid endpoint order.'
    return lo, hi


def point(value):
    value = np.asarray(value, dtype=np.float64)
    finite(value)
    return value.copy(), value.copy()


def add(a, b):
    a, b = valid(a), valid(b)
    return valid((down(a[0]+b[0]), up(a[1]+b[1])))


def sub(a, b):
    a, b = valid(a), valid(b)
    return valid((down(a[0]-b[1]), up(a[1]-b[0])))


def mul(a, b):
    a, b = valid(a), valid(b)
    products = [a[i]*b[j] for i,j in ((0,0),(0,1),(1,0),(1,1))]
    lower = np.minimum.reduce([down(value) for value in products])
    upper = np.maximum.reduce([up(value) for value in products])
    return valid((lower, upper))


def point_mul(a, b):
    # A single separately rounded product has two adjacent outward endpoints.
    finite(a,b)
    product = np.asarray(a,dtype=np.float64)*np.asarray(b,dtype=np.float64)
    return valid((down(product),up(product)))


def relu(a):
    a = valid(a)
    return np.maximum(a[0],0.),np.maximum(a[1],0.)


def sum_interval(a):
    a = valid(a)
    result = point(np.zeros(a[0].shape[:-1], dtype=np.float64))
    for k in range(a[0].shape[-1]):
        result = add(result, (a[0][...,k],a[1][...,k]))
    return result


def abs_upper(a):
    a = valid(a)
    return np.maximum(np.abs(a[0]),np.abs(a[1]))


def nonnegative_sum_upper(values):
    values = np.asarray(values,dtype=np.float64)
    finite(values)
    assert (values>=0).all()
    total = np.zeros(values.shape[:-1],dtype=np.float64)
    for k in range(values.shape[-1]):
        total = up(total+values[...,k])
    return total


def positive_division_upper(numerator_upper, denominator_lower):
    finite(numerator_upper,denominator_lower)
    assert np.all(numerator_upper>=0) and np.all(denominator_lower>0)
    return up(numerator_upper/denominator_lower)


def affine_point(x, weight, bias):
    """Enclose x @ weight.T + bias, with exact stored point operands."""
    x,weight,bias = [np.asarray(v,dtype=np.float64) for v in (x,weight,bias)]
    finite(x,weight,bias)
    assert x.ndim==weight.ndim==2 and bias.shape==(weight.shape[0],)
    assert x.shape[1]==weight.shape[1]
    result = point(np.broadcast_to(bias,(len(x),len(bias))))
    for k in range(x.shape[1]):
        result = add(result,point_mul(x[:,k,None],weight[None,:,k]))
    return result


def contraction_upper(weight):
    value = float(np.max(nonnegative_sum_upper(np.abs(weight))))
    assert value<1., 'Global contraction was not established.'
    return value


def map_interval(state, weight, injection, removed):
    """removed is one index per row; -1 means the intact map."""
    result = relu(add(affine_point(state,weight,np.zeros(len(weight))),injection))
    removed = np.asarray(removed,dtype=np.int64)
    assert removed.shape==(len(state),) and ((removed>=-1)&(removed<len(weight))).all()
    rows = np.flatnonzero(removed>=0)
    for endpoint in result:
        endpoint[rows,removed[rows]]=0.
    return valid(result)


def residual_upper(state, weight, injection, removed):
    residual = sub(map_interval(state,weight,injection,removed),point(state))
    return np.max(abs_upper(residual),axis=1)


def state_error_upper(residual, q_upper):
    denominator = down(1.-q_upper)
    return positive_division_upper(residual,denominator)


def effect_interval(base, edit, weight, injection, removed, readout, winner, runner, q_upper):
    """Enclose the real equilibrium margin drop for each paired state."""
    base_residual = residual_upper(base,weight,injection,np.full(len(base),-1))
    edit_residual = residual_upper(edit,weight,injection,removed)
    coefficient = sub(point(readout[winner]),point(readout[runner]))
    norm = nonnegative_sum_upper(abs_upper(coefficient))
    state_radius = state_error_upper(up(base_residual+edit_residual),q_upper)
    radius = up(norm*state_radius)
    center = sum_interval(mul(coefficient,sub(point(base),point(edit))))
    interval = valid((down(center[0]-radius),up(center[1]+radius)))
    return dict(lower=interval[0],upper=interval[1],radius=radius,
        center_lower=center[0],center_upper=center[1],coefficient_norm_upper=norm,
        residual_base_upper=base_residual,residual_edit_upper=edit_residual)


def logits_interval(state, weight, injection, readout, bias, q_upper):
    residual = residual_upper(state,weight,injection,np.full(len(state),-1))
    radius = state_error_upper(residual,q_upper)
    norms = nonnegative_sum_upper(np.abs(readout))
    scalar = up(radius[:,None]*norms[None,:])
    center = affine_point(state,readout,bias)
    return valid((down(center[0]-scalar),up(center[1]+scalar))),residual,radius
