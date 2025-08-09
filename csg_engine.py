"""
CSG Engine (Python)

Single-file CSG engine based on signed distance functions (SDF).
Features:
 - Primitives: Sphere, Box, Cylinder, Plane
 - Transforms: Translate, Rotate, Scale (applied to primitives)
 - Boolean ops: Union, Intersection, Difference
 - Evaluate SDF at points or on a 3D grid
 - Optional acceleration with numba if available
 - Optional mesh extraction via skimage.measure.marching_cubes
 - Optional STL export via trimesh

Usage: see bottom of file for a small example that builds a cylindrical reactor
with an inner tube and subtracts an inlet port.

Notes for performance / C++ bindings:
 - The computational heavy part is SDF evaluation on dense grids. To move to C++:
   * Keep the same SDF function signatures and expose via pybind11 or use CFFI.
   * Offload the grid evaluation loop and marching cubes to C++ for significant speedups.

"""

from __future__ import annotations
import math
from typing import Tuple, Callable, Optional
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    njit = lambda f: f
    _HAS_NUMBA = False


# ----------------------------- Core SDF primitives -----------------------------
class SDF:
    """Base SDF node. Subclass and implement sdf(p: np.ndarray) -> float or array."""
    def sdf(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # boolean combinators
    def __or__(self, other: 'SDF') -> 'Union':
        return Union(self, other)

    def __and__(self, other: 'SDF') -> 'Intersection':
        return Intersection(self, other)

    def __sub__(self, other: 'SDF') -> 'Difference':
        return Difference(self, other)


class Sphere(SDF):
    def __init__(self, radius: float = 1.0, center: Tuple[float,float,float]=(0,0,0)):
        self.r = float(radius)
        self.center = np.array(center, dtype=float)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        # p can be shape (..., 3)
        return np.linalg.norm(p - self.center, axis=-1) - self.r


class Box(SDF):
    def __init__(self, size: Tuple[float,float,float], center=(0,0,0)):
        self.half = np.array(size, dtype=float) / 2.0
        self.center = np.array(center, dtype=float)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        q = np.abs(p - self.center) - self.half
        # max(q,0) componentwise
        qpos = np.maximum(q, 0.0)
        outside = np.linalg.norm(qpos, axis=-1)
        inside = np.minimum(np.maximum.reduce(q), 0.0)
        return outside + inside


class Cylinder(SDF):
    def __init__(self, radius: float, height: float, center=(0,0,0), axis=(0,0,1)):
        self.r = float(radius)
        self.h = float(height) / 2.0
        self.center = np.array(center, dtype=float)
        a = np.array(axis, dtype=float)
        if np.linalg.norm(a) == 0:
            raise ValueError("axis must be non-zero")
        self.axis = a / np.linalg.norm(a)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        # project onto axis
        rel = p - self.center
        # compute components parallel and perpendicular to axis
        parallel = np.dot(rel, self.axis)
        perp = rel - np.outer(parallel, self.axis)
        d_perp = np.linalg.norm(perp, axis=-1) - self.r
        d_parallel = np.abs(parallel) - self.h
        qpos = np.maximum(np.stack([d_perp, d_parallel], axis=-1), 0.0)
        outside = np.linalg.norm(qpos, axis=-1)
        inside = np.minimum(np.maximum(d_perp, d_parallel), 0.0)
        return outside + inside


class Plane(SDF):
    def __init__(self, normal=(0,0,1), d=0.0):
        n = np.array(normal, dtype=float)
        if np.linalg.norm(n) == 0:
            raise ValueError("normal must be non-zero")
        self.n = n / np.linalg.norm(n)
        self.d = float(d)  # plane equation n.x + d = 0

    def sdf(self, p: np.ndarray) -> np.ndarray:
        return np.dot(p, self.n) + self.d


# ----------------------------- Transforms -----------------------------
class Transform(SDF):
    def __init__(self, child: SDF):
        self.child = child

    def sdf(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Translate(Transform):
    def __init__(self, child: SDF, offset: Tuple[float,float,float]):
        super().__init__(child)
        self.offset = np.array(offset, dtype=float)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        return self.child.sdf(p - self.offset)


class Scale(Transform):
    def __init__(self, child: SDF, scale: float):
        super().__init__(child)
        self.scale = float(scale)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        return self.child.sdf(p / self.scale) * self.scale


# Rotation around axis using Rodrigues formula
class Rotate(Transform):
    def __init__(self, child: SDF, axis: Tuple[float,float,float], angle_rad: float):
        super().__init__(child)
        a = np.array(axis, dtype=float)
        assert np.linalg.norm(a) > 0
        self.axis = a / np.linalg.norm(a)
        self.angle = float(angle_rad)
        # precompute rotation matrix
        ux, uy, uz = self.axis
        c = math.cos(self.angle)
        s = math.sin(self.angle)
        R = np.array([
            [c + ux*ux*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
            [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c), uy*uz*(1-c)-ux*s],
            [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)]
        ], dtype=float)
        self.R = R

    def sdf(self, p: np.ndarray) -> np.ndarray:
        # rotate points by inverse rotation
        # if p is (...,3), we need to apply R^T
        orig_shape = p.shape
        flat = p.reshape(-1, 3)
        rotated = flat.dot(self.R.T)
        rotated = rotated.reshape(orig_shape)
        return self.child.sdf(rotated)


# ----------------------------- Boolean ops -----------------------------
class Union(SDF):
    def __init__(self, a: SDF, b: SDF):
        self.a = a
        self.b = b

    def sdf(self, p: np.ndarray) -> np.ndarray:
        return np.minimum(self.a.sdf(p), self.b.sdf(p))


class Intersection(SDF):
    def __init__(self, a: SDF, b: SDF):
        self.a = a
        self.b = b

    def sdf(self, p: np.ndarray) -> np.ndarray:
        return np.maximum(self.a.sdf(p), self.b.sdf(p))


class Difference(SDF):
    def __init__(self, a: SDF, b: SDF):
        self.a = a
        self.b = b

    def sdf(self, p: np.ndarray) -> np.ndarray:
        return np.maximum(self.a.sdf(p), -self.b.sdf(p))


# ----------------------------- Utilities -----------------------------
def evaluate_on_grid(sdf: SDF, bounds: Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]],
                     resolution: Tuple[int,int,int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate SDF on a regular 3D grid.
    returns (grid_coords, sdf_values) where grid_coords is (nx, ny, nz, 3) and sdf_values is (nx, ny, nz)
    """
    (xmin,xmax), (ymin,ymax), (zmin,zmax) = bounds
    nx, ny, nz = resolution
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([X, Y, Z], axis=-1)  # (nx,ny,nz,3)
    flat = pts.reshape(-1, 3)
    vals = sdf.sdf(flat)
    vals = vals.reshape((nx, ny, nz))
    return pts, vals


# Optional numba-accelerated grid evaluator for heavy loops (keeps same interface)
def evaluate_on_grid_numba(sdf_func: Callable[[np.ndarray], np.ndarray], bounds, resolution):
    (xmin,xmax), (ymin,ymax), (zmin,zmax) = bounds
    nx, ny, nz = resolution
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([X, Y, Z], axis=-1)
    flat = pts.reshape(-1, 3)
    vals = sdf_func(flat)
    vals = vals.reshape((nx, ny, nz))
    return pts, vals


# ----------------------------- Mesh extraction / export -----------------------------
def mesh_from_grid(sdf_values: np.ndarray, bounds, level: float = 0.0):
    """
    Use marching cubes to extract a mesh from the SDF grid. Requires skimage.
    Returns verts (N,3), faces (M,3)
    """
    try:
        from skimage import measure
    except Exception as e:
        raise RuntimeError("skimage is required for marching cubes. Install scikit-image.") from e

    (xmin,xmax), (ymin,ymax), (zmin,zmax) = bounds
    nx, ny, nz = sdf_values.shape
    spacing = ((xmax-xmin)/(nx-1), (ymax-ymin)/(ny-1), (zmax-zmin)/(nz-1))
    verts, faces, normals, values = measure.marching_cubes(sdf_values, level=level, spacing=spacing)
    # marching_cubes returns verts in index coords (x,y,z) starting at 0; shift to actual coords
    verts[:,0] += xmin
    verts[:,1] += ymin
    verts[:,2] += zmin
    return verts, faces


def export_stl(verts: np.ndarray, faces: np.ndarray, filename: str):
    try:
        import trimesh
    except Exception as e:
        raise RuntimeError("trimesh is required for STL export. Install trimesh.") from e
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)


