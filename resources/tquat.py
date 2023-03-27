import torch
import numpy as np

"""
def eye(shape):
    return torch.as_tensor([1, 0, 0, 0], dtype=torch.float32).repeat(shape + [1])
"""

def _fast_cross(a, b):
    return torch.cat([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]], dim=-1)

def length(x):
    return torch.sqrt(torch.sum(torch.square(x), dim=-1))

def normalize(x, eps=1e-8):
    return x / (length(x)[...,None] + eps)

def abs(x):
    return torch.where(x[...,0:1] > 0.0, x, -x)

def from_angle_axis(angle, axis):
    c = torch.cos(angle / 2.0)[...,None]
    s = torch.sin(angle / 2.0)[...,None]
    q = torch.cat([c, s * axis], dim=-1)
    return q

def to_xform(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return torch.cat([
        torch.cat([1.0 - (yy + zz), xy - wz, xz + wy], dim=-1)[...,None,:],
        torch.cat([xy + wz, 1.0 - (xx + zz), yz - wx], dim=-1)[...,None,:],
        torch.cat([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=-1)[...,None,:],
    ], dim=-2)
    
def to_xform_xy(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return torch.cat([
        torch.cat([1.0 - (yy + zz), xy - wz], dim=-1)[...,None,:],
        torch.cat([xy + wz, 1.0 - (xx + zz)], dim=-1)[...,None,:],
        torch.cat([xz - wy, yz + wx], dim=-1)[...,None,:],
    ], dim=-2)
    

def from_euler(e, order='zyx'):

    axis = {
        'x': torch.as_tensor([1, 0, 0], dtype=torch.float32),
        'y': torch.as_tensor([0, 1, 0], dtype=torch.float32),
        'z': torch.as_tensor([0, 0, 1], dtype=torch.float32)}

    q0 = from_angle_axis(e[..., 0], axis[order[0]])
    q1 = from_angle_axis(e[..., 1], axis[order[1]])
    q2 = from_angle_axis(e[..., 2], axis[order[2]])

    return mul(q0, mul(q1, q2))

def from_xform(ts):

    return normalize(
        torch.where((ts[...,2,2] < 0.0)[...,None],
            torch.where((ts[...,0,0] >  ts[...,1,1])[...,None],
                torch.cat([
                    (ts[...,2,1]-ts[...,1,2])[...,None], 
                    (1.0 + ts[...,0,0] - ts[...,1,1] - ts[...,2,2])[...,None], 
                    (ts[...,1,0]+ts[...,0,1])[...,None], 
                    (ts[...,0,2]+ts[...,2,0])[...,None]], dim=-1),
                torch.cat([
                    (ts[...,0,2]-ts[...,2,0])[...,None], 
                    (ts[...,1,0]+ts[...,0,1])[...,None], 
                    (1.0 - ts[...,0,0] + ts[...,1,1] - ts[...,2,2])[...,None], 
                    (ts[...,2,1]+ts[...,1,2])[...,None]], dim=-1)),
            torch.where((ts[...,0,0] < -ts[...,1,1])[...,None],
                torch.cat([
                    (ts[...,1,0]-ts[...,0,1])[...,None], 
                    (ts[...,0,2]+ts[...,2,0])[...,None], 
                    (ts[...,2,1]+ts[...,1,2])[...,None], 
                    (1.0 - ts[...,0,0] - ts[...,1,1] + ts[...,2,2])[...,None]], dim=-1),
                torch.cat([
                    (1.0 + ts[...,0,0] + ts[...,1,1] + ts[...,2,2])[...,None], 
                    (ts[...,2,1]-ts[...,1,2])[...,None], 
                    (ts[...,0,2]-ts[...,2,0])[...,None], 
                    (ts[...,1,0]-ts[...,0,1])[...,None]], dim=-1))))


def from_xform_xy(x):

    c2 = _fast_cross(x[...,0], x[...,1])
    c2 = c2 / torch.sqrt(torch.sum(torch.square(c2), dim=-1))[...,None]
    c1 = _fast_cross(c2, x[...,0])
    c1 = c1 / torch.sqrt(torch.sum(torch.square(c1), dim=-1))[...,None]
    c0 = x[...,0]
    
    return from_xform(torch.cat([
        c0[...,None], 
        c1[...,None], 
        c2[...,None]], axis=-1))

def inv(q):
    return torch.as_tensor([1, -1, -1, -1], dtype=torch.float32) * q

def mul(x, y):
    x0, x1, x2, x3 = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    y0, y1, y2, y3 = y[...,0:1], y[...,1:2], y[...,2:3], y[...,3:4]
    
    return torch.cat([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], dim=-1)

def inv_mul(x, y):
    return mul(inv(x), y)

def mul_inv(x, y):
    return mul(x, inv(y))

def mul_vec(q, x):
    t = 2.0 * _fast_cross(q[..., 1:], x)
    return x + q[...,0:1] * t + _fast_cross(q[...,1:], t)

def inv_mul_vec(q, x):
    return mul_vec(inv(q), x)

def between(x, y):
    return torch.concatenate([
        torch.sqrt(np.sum(torch.square(x), dim=-1) * np.sum(torch.square(y), dim=-1))[...,None] + 
        torch.sum(x * y, dim=-1)[...,None], 
        _fast_cross(x, y)], dim=-1)
        
def log(x, eps=1e-5):
    length = torch.sqrt(torch.sum(torch.square(x[...,1:]), dim=-1))[...,None]
    halfangle = torch.where(length < eps, torch.ones_like(length), torch.arctan2(length, x[...,0:1]) / length)
    return halfangle * x[...,1:]
    
def exp(x, eps=1e-5):
    halfangle = torch.sqrt(torch.sum(torch.square(x), dim=-1))[...,None]
    c = torch.where(halfangle < eps, torch.ones_like(halfangle), torch.cos(halfangle))
    s = torch.where(halfangle < eps, torch.ones_like(halfangle), torch.sinc(halfangle / np.pi))
    return torch.cat([c, s * x], dim=-1)
    
def to_scaled_angle_axis(x, eps=1e-5):
    return 2.0 * log(x, eps)
    
def from_scaled_angle_axis(x, eps=1e-5):
    return exp(x / 2.0, eps)

def fk(lrot, lpos, parents):
    
    gp, gr = [lpos[...,:1,:]], [lrot[...,:1,:]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
        gr.append(mul    (gr[parents[i]], lrot[...,i:i+1,:]))
        
    return torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)
    
def ik(grot, gpos, parents):
    
    return (
        torch.cat([
            grot[...,:1,:],
            mul(inv(grot[...,parents[1:],:]), grot[...,1:,:]),
        ], dim=-2),
        torch.cat([
            gpos[...,:1,:],
            mul_vec(
                inv(grot[...,parents[1:],:]),
                gpos[...,1:,:] - gpos[...,parents[1:],:]),
        ], dim=-2))
    
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
        torch.cat(gr, dim=-2), 
        torch.cat(gp, dim=-2),
        torch.cat(gv, dim=-2),
        torch.cat(ga, dim=-2))
