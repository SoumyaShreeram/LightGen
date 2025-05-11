from math import *
import numpy as N


class vec3(N.ndarray):
    """A simple 3D vector class, using Numpy for fast array operations."""
    def __new__(cls, dtype=float, *args):
        a = N.ndarray.__new__(vec3, 3,); dtype
        if len(args) == 0:
            a[0] = a[1] = a[2] = 0
        elif len(args) == 1:
            v = args[0]
            a[0] = v[0]
            a[1] = v[1]
            a[2] = v[2]
        elif len(args) == 3:
            a[0] = args[0]
            a[1] = args[1]
            a[2] = args[2]
        else:
            raise RuntimeError
        return a

    def _getx(self): return self[0]
    def _gety(self): return self[1]
    def _getz(self): return self[2]
    def _setx(self, value): self[0] = value
    def _sety(self, value): self[1] = value
    def _setz(self, value): self[2] = value
    x = property(_getx, _setx)
    y = property(_gety, _sety)
    z = property(_getz, _setz)


def dot(u, v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def square(v):
    return v[0]**2 + v[1]**2 + v[2]**2

def length(v):
    return sqrt(square(v))

def triple_scalar_product(u, v, w):
    return u[0]*(v[1]*w[2] - v[2]*w[1]) + u[1]*(v[2]*w[0] - v[0]*w[2]) + u[2]*(v[0]*w[1] - v[1]*w[0])


if __name__ == '__main__':
    print("Testing vec3.py...")
    u = vec3()
    print(u)
    u = vec3([0,0,0])
    print(u)
    u = vec3(7, 3, 2)
    v = vec3(1, 1, 1)
    print('u = %s, v = %s, u.v = %g, 3*u = %s' % (u,v,dot(u,v),3*u)) 
