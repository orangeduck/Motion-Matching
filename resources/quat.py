import numpy as np

def _fast_cross(a, b):
    o = np.empty(np.broadcast(a, b).shape)
    o[...,0] = a[...,1]*b[...,2] - a[...,2]*b[...,1]
    o[...,1] = a[...,2]*b[...,0] - a[...,0]*b[...,2]
    o[...,2] = a[...,0]*b[...,1] - a[...,1]*b[...,0]
    return o

def eye(shape):
    return np.ones(list(shape) + [4], dtype=np.float32) * np.asarray([1, 0, 0, 0], dtype=np.float32)

def length(x):
    return np.sqrt(np.sum(x * x, axis=-1))

def normalize(x, eps=1e-8):
    return x / (length(x)[...,np.newaxis] + eps)

def abs(x):
    return np.where(x[...,0:1] > 0.0, x, -x)

def from_angle_axis(angle, axis):
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q

def to_xform(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)[...,np.newaxis,:],
        np.concatenate([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)[...,np.newaxis,:],
        np.concatenate([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)[...,np.newaxis,:],
    ], axis=-2)

def from_euler(e, order='zyx'):
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = from_angle_axis(e[..., 0], axis[order[0]])
    q1 = from_angle_axis(e[..., 1], axis[order[1]])
    q2 = from_angle_axis(e[..., 2], axis[order[2]])

    return mul(q0, mul(q1, q2))

def from_xform(ts, eps=1e-10):

    qs = np.empty_like(ts[...,:1,0].repeat(4, axis=-1))

    t = ts[...,0,0] + ts[...,1,1] + ts[...,2,2]
    
    s = 0.5 / np.sqrt(np.maximum(t + 1, eps))
    qs = np.where((t > 0)[...,np.newaxis].repeat(4, axis=-1), np.concatenate([
        (0.25 / s)[...,np.newaxis],
        (s * (ts[...,2,1] - ts[...,1,2]))[...,np.newaxis],
        (s * (ts[...,0,2] - ts[...,2,0]))[...,np.newaxis],
        (s * (ts[...,1,0] - ts[...,0,1]))[...,np.newaxis]
    ], axis=-1), qs)
    
    c0 = (ts[...,0,0] > ts[...,1,1]) & (ts[...,0,0] > ts[...,2,2])
    s0 = 2.0 * np.sqrt(np.maximum(1.0 + ts[...,0,0] - ts[...,1,1] - ts[...,2,2], eps))
    qs = np.where(((t <= 0) & c0)[...,np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[...,2,1] - ts[...,1,2]) / s0)[...,np.newaxis],
        (s0 * 0.25)[...,np.newaxis],
        ((ts[...,0,1] + ts[...,1,0]) / s0)[...,np.newaxis],
        ((ts[...,0,2] + ts[...,2,0]) / s0)[...,np.newaxis]
    ], axis=-1), qs)
    
    c1 = (~c0) & (ts[...,1,1] > ts[...,2,2])
    s1 = 2.0 * np.sqrt(np.maximum(1.0 + ts[...,1,1] - ts[...,0,0] - ts[...,2,2], eps))
    qs = np.where(((t <= 0) & c1)[...,np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[...,0,2] - ts[...,2,0]) / s1)[...,np.newaxis],
        ((ts[...,0,1] + ts[...,1,0]) / s1)[...,np.newaxis],
        (s1 * 0.25)[...,np.newaxis],
        ((ts[...,1,2] + ts[...,2,1]) / s1)[...,np.newaxis]
    ], axis=-1), qs)
    
    c2 = (~c0) & (~c1)
    s2 = 2.0 * np.sqrt(np.maximum(1.0 + ts[...,2,2] - ts[...,0,0] - ts[...,1,1], eps))
    qs = np.where(((t <= 0) & c2)[...,np.newaxis].repeat(4, axis=-1), np.concatenate([
        ((ts[...,1,0] - ts[...,0,1]) / s2)[...,np.newaxis],
        ((ts[...,0,2] + ts[...,2,0]) / s2)[...,np.newaxis],
        ((ts[...,1,2] + ts[...,2,1]) / s2)[...,np.newaxis],
        (s2 * 0.25)[...,np.newaxis]
    ], axis=-1), qs)
    
    return qs

def inv(q):
    return np.asarray([1, -1, -1, -1], dtype=np.float32) * q

def mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

def inv_mul(x, y):
    return mul(inv(x), y)

def mul_inv(x, y):
    return mul(x, inv(y))

def mul_vec(q, x):
    t = 2.0 * _fast_cross(q[..., 1:], x)
    return x + q[..., 0][..., np.newaxis] * t + _fast_cross(q[..., 1:], t)

def inv_mul_vec(q, x):
    return mul_vec(inv(q), x)

def unroll(x):
    y = x.copy()
    for i in range(1, len(x)):
        d0 = np.sum( y[i] * y[i-1], axis=-1)
        d1 = np.sum(-y[i] * y[i-1], axis=-1)
        y[i][d0 < d1] = -y[i][d0 < d1]
    return y

def between(x, y):
    return np.concatenate([
        np.sqrt(np.sum(x*x, axis=-1) * np.sum(y*y, axis=-1))[...,np.newaxis] + 
        np.sum(x * y, axis=-1)[...,np.newaxis], 
        _fast_cross(x, y)], axis=-1)
        
def log(x, eps=1e-5):
    length = np.sqrt(np.sum(np.square(x[...,1:]), axis=-1))[...,np.newaxis]
    halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[...,0:1]) / length)
    return halfangle * x[...,1:]
    
def exp(x, eps=1e-5):
    halfangle = np.sqrt(np.sum(np.square(x), axis=-1))[...,np.newaxis]
    c = np.where(halfangle < eps, np.ones_like(halfangle), np.cos(halfangle))
    s = np.where(halfangle < eps, np.ones_like(halfangle), np.sinc(halfangle / np.pi))
    return np.concatenate([c, s * x], axis=-1)
    
def to_scaled_angle_axis(x, eps=1e-5):
    return 2.0 * log(x, eps)
    
def from_scaled_angle_axis(x, eps=1e-5):
    return exp(x / 2.0, eps)

def fk(lrot, lpos, parents):
    
    gp, gr = [lpos[...,:1,:]], [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        
    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    
def ik(grot, gpos, parents):
    
    return (
        np.concatenate([
            grot[...,:1,:],
            mul(inv(grot[...,parents[1:],:]), grot[...,1:,:]),
        ], axis=-2),
        np.concatenate([
            gpos[...,:1,:],
            mul_vec(
                inv(grot[...,parents[1:],:]),
                gpos[...,1:,:] - gpos[...,parents[1:],:]),
        ], axis=-2))
    
def fk_vel(lrot, lpos, lvel, lang, parents):
    
    gp, gr, gv, ga = [lpos[...,:1,:]], [lrot[...,:1,:]], [lvel[...,:1,:]], [lang[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        gv.append(mul_vec(gr[parents[i]], lvel[...,i:i+1,:]) + 
            _fast_cross(ga[parents[i]], mul_vec(gr[parents[i]], lpos[...,i:i+1,:])) +
            gv[parents[i]])
        ga.append(mul_vec(gr[parents[i]], lang[...,i:i+1,:]) + ga[parents[i]])
        
    return (
        np.concatenate(gr, axis=-2), 
        np.concatenate(gp, axis=-2),
        np.concatenate(gv, axis=-2),
        np.concatenate(ga, axis=-2))
        