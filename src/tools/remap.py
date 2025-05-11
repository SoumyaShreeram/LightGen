#!/usr/bin/python
"""
remap.py

Cuboid remapping implementation.

See http://mwhite.berkeley.edu/BoxRemap/ for details.

"Embedding realistic surveys in simulations through volume remapping", J. Carlson and M. White, ApJS 190(2010)311.
(preprint available at arxiv:1003.3178)
"""
import sys
from math import *

verbose = False


try:
    # Use fast vec3 implementation if Numpy is available
    import numpy as N
    class vec3(N.ndarray):
        """A simple 3D vector class, using Numpy for fast array operations."""
        def __new__(cls, *args):
            a = N.ndarray.__new__(vec3, (3,), float)
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

except:
    # Fall back to non-Numpy implementation
    class vec3:
        """A simple 3D vector class."""
        def __init__(self, x, y=None, z=None, dtype=float):
            if y is None and z is None:
                self.x = dtype(x[0])
                self.y = dtype(x[1])
                self.z = dtype(x[2])
            else:
                self.x = dtype(x)
                self.y = dtype(y)
                self.z = dtype(z)

        def __pos__(self):
            return self

        def __neg__(self):
            return vec3(-self.x, -self.y, -self.z)

        def __add__(self, other):
            return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

        def __sub__(self, other):
            return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

        def __rmul__(self, scalar):
            return vec3(self.x*scalar, self.y*scalar, self.z*scalar)

        def __mul__(self, scalar):
            return vec3(self.x*scalar, self.y*scalar, self.z*scalar)

        def __div__(self, scalar):
            return vec3(self.x/scalar, self.y/scalar, self.z/scalar)

        def __repr__(self):
            return "(%g, %g, %g)" % (self.x, self.y, self.z)

        def __getitem__(self, i):
            if   i == 0: return self.x
            elif i == 1: return self.y
            elif i == 2: return self.z


def dot(u, v):
    return u.x*v.x + u.y*v.y + u.z*v.z

def square(v):
    return v.x**2 + v.y**2 + v.z**2

def length(v):
    return sqrt(square(v))

def triple_scalar_product(u, v, w):
    return u.x*(v.y*w.z - v.z*w.y) + u.y*(v.z*w.x - v.x*w.z) + u.z*(v.x*w.y - v.y*w.x)


class Plane:
    def __init__(self, p, n):
        self.a = n.x
        self.b = n.y
        self.c = n.z
        self.d = -dot(p,n)

    def normal(self):
        ell = sqrt(self.a**2 + self.b**2 + self.c**2)
        return vec3(self.a/ell, self.b/ell, self.c/ell)

    def test(self, x, y, z):
        """Compare a point to a plane.  Return value is positive, negative, or
        zero depending on whether the point lies above, below, or on the plane."""
        return self.a*x + self.b*y + self.c*z + self.d


class Cell:
    def __init__(self, ix=0, iy=0, iz=0):
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.faces = []

    def contains(self, x, y, z):
        for f in self.faces:
            if f.test(x,y,z) < 0:
                return False
        return True

    
def UnitCubeTest(P):
    """Return +1, 0, or -1 if the unit cube is above, below, or intersecting the plane."""
    above = 0
    below = 0
    for (a,b,c) in [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]:
        s = P.test(a, b, c)
        if s > 0:
            above = 1
        elif s < 0:
            below = 1
    return above - below


class Cuboid:
    """Cuboid remapping class."""

    def __init__(self, u1=(1,0,0), u2=(0,1,0), u3=(0,0,1)):
        """Initialize by passing a 3x3 invertible integer matrix."""
        u1 = vec3(u1)
        u2 = vec3(u2)
        u3 = vec3(u3)

        if triple_scalar_product(u1, u2, u3) != 1:
            print >> sys.stderr, "!! Invalid lattice vectors: u1 = %s, u2 = %s, u3 = %s" % (u1,u2,u3)
            self.e1 = vec3(1,0,0)
            self.e2 = vec3(0,1,0)
            self.e3 = vec3(0,0,1)
        else:
            s1 = square(u1)
            s2 = square(u2)
            d12 = dot(u1, u2)
            d23 = dot(u2, u3)
            d13 = dot(u1, u3)
            alpha = -d12/s1
            gamma = -(alpha*d13 + d23)/(alpha*d12 + s2)
            beta = -(d13 + gamma*d12)/s1
            self.e1 = u1
            self.e2 = u2 + alpha*u1
            self.e3 = u3 + beta*u1 + gamma*u2

        if verbose:
            print( "e1 = %s" % self.e1)
            print( "e2 = %s" % self.e2)
            print( "e3 = %s" % self.e3
)
        self.L1 = length(self.e1)
        self.L2 = length(self.e2)
        self.L3 = length(self.e3)
        self.n1 = self.e1/self.L1
        self.n2 = self.e2/self.L2
        self.n3 = self.e3/self.L3
        self.cells = []

        v0 = vec3(0,0,0)
        self.v = [v0,
                  v0 + self.e3,
                  v0 + self.e2,
                  v0 + self.e2 + self.e3,
                  v0 + self.e1,
                  v0 + self.e1 + self.e3,
                  v0 + self.e1 + self.e2,
                  v0 + self.e1 + self.e2 + self.e3]

        # Compute bounding box of cuboid
        xs = [vk.x for vk in self.v]
        ys = [vk.y for vk in self.v]
        zs = [vk.z for vk in self.v]
        vmin = vec3(min(xs), min(ys), min(zs))
        vmax = vec3(max(xs), max(ys), max(zs))

        # Extend to nearest integer coordinates
        ixmin = int(floor(vmin.x))
        ixmax = int(ceil(vmax.x))
        iymin = int(floor(vmin.y))
        iymax = int(ceil(vmax.y))
        izmin = int(floor(vmin.z))
        izmax = int(ceil(vmax.z))
        if verbose:
            print( "ixmin, ixmax = %d, %d" % (ixmin,ixmax))
            print( "iymin, iymax = %d, %d" % (iymin,iymax))
            print( "izmin, izmax = %d, %d" % (izmin,izmax))

        # Determine which cells (and which faces within those cells) are non-trivial
        for ix in range(ixmin, ixmax):
            for iy in range(iymin, iymax):
                for iz in range(izmin, izmax):
                    shift = vec3(-ix, -iy, -iz)
                    faces = [Plane(self.v[0] + shift, +self.n1),
                             Plane(self.v[4] + shift, -self.n1),
                             Plane(self.v[0] + shift, +self.n2),
                             Plane(self.v[2] + shift, -self.n2),
                             Plane(self.v[0] + shift, +self.n3),
                             Plane(self.v[1] + shift, -self.n3)]

                    c = Cell(ix, iy, iz)
                    skipcell = False
                    for f in faces:
                        r = UnitCubeTest(f)
                        if r == +1:
                            # Unit cube is completely above this plane; this cell is empty
                            continue
                        elif r == 0:
                            # Unit cube intersects this plane; keep track of it
                            c.faces.append(f)
                        elif r == -1:
                            skipcell = True
                            break

                    if skipcell or len(c.faces) == 0:
                        if verbose:
                            print("Skipping cell at (%d,%d,%d)" % (ix,iy,iz))
                        continue
                    else:
                        self.cells.append(c)
                        if verbose:
                            print( "Adding cell at (%d,%d,%d)" % (ix,iy,iz))

        # For the identity remapping, use exactly one cell
        if len(self.cells) == 0:
            self.cells.append(Cell())

        # Print the full list of cells
        if verbose:
            print( "%d non-empty cells" % len(self.cells))
            for c in self.cells:
                print( "Cell at (%d,%d,%d) has %d non-trivial planes" % (c.ix, c.iy, c.iz, len(c.faces)))

    def Transform(self, x, y, z):
        for c in self.cells:
            if c.contains(x,y,z):
                x += c.ix
                y += c.iy
                z += c.iz
                p = vec3(x,y,z)
                return (dot(p, self.n1), dot(p, self.n2), dot(p, self.n3))
        raise RuntimeError( "(%g, %g, %g) not contained in any cell" % (x,y,z))

    def InverseTransform(self, r1, r2, r3):
        p = r1*self.n1 + r2*self.n2 + r3*self.n3
        x1 = fmod(p[0], 1) + (p[0] < 0)
        x2 = fmod(p[1], 1) + (p[1] < 0)
        x3 = fmod(p[2], 1) + (p[2] < 0)
        return vec3(x1, x2, x3)


def abort(msg=None, code=1):
    if msg:
        print >> sys.stderr, msg
    sys.exit(code)

if __name__ == '__main__':
    # Parse command line arguments
    params = {}
    for arg in sys.argv[1:]:
        pair = arg.split('=', 1)
        if len(pair) == 2:
            name, val = pair
            if   name == "m": params['m'] = int(val)
            elif name == "n": params['n'] = int(val)
            elif name == "u": params['u'] = [int(f) for f in val.strip("[()]").replace(',', ' ').split()]
            elif name == "u1": params['u1'] = [int(f) for f in val.strip("[()]").replace(',', ' ').split()]
            elif name == "u2": params['u2'] = [int(f) for f in val.strip("[()]").replace(',', ' ').split()]
            elif name == "u3": params['u3'] = [int(f) for f in val.strip("[()]").replace(',', ' ').split()]
            elif name == "in": params['in'] = str(val)
            elif name == "out": params['out'] = str(val)
            else: abort("Unrecognized parameter '%s'" % name)
        else:
            if arg == "-v" or arg == "--verbose":
                verbose = True
            elif arg == "-h" or arg == "--help":
                print("Usage: python remap.py [OPTIONS] PARAMS")
            else:
                abort("Unrecognized option '%s'" % arg)

    # Open input and output files
    if 'in' not in params or params['in'] == "stdin":
        fin = sys.stdin
    else:
        fin = open(params['in'], "r")
        if not fin: abort("Could not open input file '%s'" % params['in'])
    if 'out' not in params or params['out'] == "stdout":
        fout = sys.stdout
    else:
        fout = open(params['out'], "w")
        if not fout: abort("!! Could not open output file '%s'" % params['out'])

    # Initialize remapping
    if 'm' in params and 'n' in params:
        m = params['m']
        n = params['n']
        u1 = (1,m,n)
        u2 = (0,1,0)
        u3 = (0,0,1)
    elif 'u' in params:
        u = params['u']
        if len(u) != 9: abort("!! Input matrix 'u' should have 9 components, not %d" % len(u))
        u1 = (u[0], u[1], u[2])
        u2 = (u[3], u[4], u[5])
        u3 = (u[6], u[7], u[8])
    elif 'u1' in params and 'u2' in params and 'u3' in params:
        u1 = params['u1']
        u2 = params['u2']
        u3 = params['u3']
    else:
        print >> sys.stderr, "?? Cuboid geometry not specified, assuming trivial remapping"
        u1 = (1,0,0)
        u2 = (0,1,0)
        u3 = (0,0,1)

    if verbose:
        print( "u1 = %s, u2 = %s, u3 = %s" % (u1,u2,u3))
    C = Cuboid(u1, u2, u3)

    for line in fin:
        line = line.strip()
        if len(line) == 0 or line.startswith('#'):
            continue
        coords = line.replace(',', ' ').split()
        if len(coords) != 3:
            print >> sys.stderr, "?? Expecting 3 coordinates per line, not '%s'" % line
            continue
        (xin,yin,zin) = float(coords[0]), float(coords[1]), float(coords[2])
        (xout,yout,zout) = C.Transform(xin, yin, zin)
        print >> fout, "%e %e %e" % (xout,yout,zout)

    fin.close()
    fout.close()
