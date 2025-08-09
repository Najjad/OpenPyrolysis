"""
pyro_reactor_sim.py

Extends the CSG engine (csg_engine.py) to couple reactor geometry with simple
pyrolysis kinetics. The file provides:

- Kinetics class: user-provided rate law (dalpha/dt = f(alpha, T, params))
- ReactorGeometry helper: build composite geometries (stacked cylinders, etc.)
  using the CSG primitives and boolean ops from csg_engine.py
- ReactorSimulator: sample the interior of any SDF reactor geometry on a grid
  and integrate alpha(t) at every grid cell using a simple explicit RK4 ODE
  integrator. Two modes are provided:
    * batch mode: every cell evolves with the same time-temperature history
    * plug-flow (axial) approximation: space is treated along an axis and
      residence time is computed by velocity; cross-sections are averaged

- Example at bottom shows how to build two stacked cylinders (top feeds into bottom)
  and compute alpha on the interior grid.

Notes & limitations:
- This is a physics-kernel proof-of-concept. For accurate reactor modelling
  you'll need to include advection, diffusion, heat transfer, and possibly a
  CFD solver. This module focuses on coupling geometry -> voxelization -> local
  kinetic integration so you can prototype kinetics on complex shapes.
- No external dependencies beyond numpy (and the earlier csg_engine file).

"""
from __future__ import annotations
import math
from typing import Callable, Tuple, List, Optional
import numpy as np

# import everything exported by the csg_engine file created earlier.
# Ensure that csg_engine.py is in the same directory or on PYTHONPATH.
from csg_engine import (
    SDF, Sphere, Box, Cylinder, Plane, Translate, Rotate, Scale,
    Union, Intersection, Difference, evaluate_on_grid
)


# ----------------------------- Kinetics -----------------------------
class Kinetics:
    """Simple kinetics wrapper.

    User provides a rate function r(alpha, T, params) -> dalpha/dt.
    We store params (dict) and provide helper methods.
    """
    def __init__(self, rate_func: Callable[[float, float, dict], float], params: dict):
        self.rate_func = rate_func
        self.params = params

    def rate(self, alpha: float, T: float) -> float:
        return self.rate_func(alpha, T, self.params)


# Example Arrhenius first-order rate: dalpha/dt = k(T) * (1 - alpha)
def arrhenius_first_order(alpha: float, T: float, params: dict) -> float:
    A = params.get('A', 1e3)
    Ea = params.get('Ea', 1e5)
    R = params.get('R', 8.314)
    n = params.get('n', 1.0)
    k = A * math.exp(-Ea / (R * T))
    return k * ((1.0 - alpha) ** n)


# ----------------------------- Geometry helpers -----------------------------
class ReactorGeometry:
    """Helper to build reactor geometries from primitives.

    Includes factory methods for:
      - stacked_cylinders: create two (or many) cylinders stacked along z
      - general_union: union of arbitrary SDFs

    Uses CSG operations (Union, Difference) from csg_engine.
    """
    @staticmethod
    def stacked_cylinders(params: List[dict], center_z: float = 0.0, spacing: float = 0.0) -> SDF:
        """
        Build stacked cylinders.

        params: list of dict each containing keys: radius, height, center=(x,y,z) optional, axis
        Example: [{'radius':0.2,'height':0.5}, {'radius':0.4,'height':1.0}]

        The cylinders are stacked along z. center_z is the central z coordinate of the stack.
        spacing adds extra gap between cylinders (use negative spacing to overlap/merge).
        Returns an SDF representing the union of cylinders.
        """
        # compute total height for centering
        heights = [p.get('height', 1.0) for p in params]
        total_h = sum(heights) + spacing * (len(heights)-1)
        z_top = center_z + total_h/2.0

        sdf_nodes = []
        cur_top = z_top
        for p in params:
            h = p.get('height', 1.0)
            r = p.get('radius', 0.5)
            axis = p.get('axis', (0,0,1))
            # center of this cylinder
            center = p.get('center', (0.0, 0.0, cur_top - h/2.0))
            cyl = Cylinder(radius=r, height=h, center=center, axis=axis)
            sdf_nodes.append(cyl)
            cur_top = cur_top - h - spacing

        # union them all
        geom = sdf_nodes[0]
        for node in sdf_nodes[1:]:
            geom = Union(geom, node)
        return geom

    @staticmethod
    def apply_port(geometry: SDF, port_radius: float, port_length: float, port_offset: Tuple[float,float,float], port_axis=(1,0,0)) -> SDF:
        """
        Subtract a cylindrical port (hole) from `geometry` by placing a cylinder then
        rotating it to match axis and subtracting.
        """
        port = Cylinder(radius=port_radius, height=port_length, center=(0,0,0), axis=port_axis)
        # translate port to offset
        port = Translate(port, offset=port_offset)
        return Difference(geometry, port)


# ----------------------------- Reactor simulator -----------------------------
class ReactorSimulator:
    """Simulate alpha on a voxelized reactor interior.

    Workflow:
      1. Build or receive an SDF representing the reactor solid (reactor walls) or the reactor volume.
         By convention here we treat the SDF as the reactor solid where sdf<0 inside the solid. If you
         instead have SDF that is negative inside the reactor volume, adjust the sign test when masking.
      2. Use evaluate_on_grid to sample SDF values on a regular grid.
      3. Identify interior voxels (sdf < 0 if interior is negative)
      4. Integrate alpha(t) for each interior voxel using a supplied kinetics model and temperature field.

    Limitations: This performs pointwise integration only. No advection or diffusion coupling.
    """
    def __init__(self, sdf: SDF):
        self.sdf = sdf
        self.grid_coords = None
        self.sdf_values = None
        self.interior_mask = None
        self.alpha = None
        self.T_field = None

    def sample_grid(self, bounds: Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]],
                    resolution: Tuple[int,int,int]):
        pts, vals = evaluate_on_grid(self.sdf, bounds, resolution)
        self.grid_coords = pts
        self.sdf_values = vals
        # convention: negative inside (typical signed distance). Adjust if different.
        self.interior_mask = (vals < 0.0)
        self.alpha = np.zeros_like(vals)
        return pts, vals

    # simple RK4 integrator for scalar ODE dalpha/dt = f(alpha, t, T)
    @staticmethod
    def _rk4_step(alpha, t, dt, deriv):
        k1 = deriv(alpha, t)
        k2 = deriv(alpha + 0.5*dt*k1, t + 0.5*dt)
        k3 = deriv(alpha + 0.5*dt*k2, t + 0.5*dt)
        k4 = deriv(alpha + dt*k3, t + dt)
        return alpha + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def integrate_alpha_batch(self, kinetics: Kinetics, T_func: Callable[[float, np.ndarray], np.ndarray],
                              t_span: Tuple[float,float], dt: float):
        """
        Batch mode: every interior voxel evolves with the same global time history but
        may experience a different temperature T(x,t) provided by T_func.

        T_func(t, coords_flat) -> shape (N,) temperatures for each coordinate

        Returns alpha array of shape grid matching sampled grid.
        """
        if self.grid_coords is None:
            raise RuntimeError("Grid not sampled. Call sample_grid() first.")
        nx, ny, nz, _ = self.grid_coords.shape
        mask = self.interior_mask

        # flatten interior points
        flat_pts = self.grid_coords.reshape(-1,3)
        flat_mask = mask.ravel()
        interior_idx = np.where(flat_mask)[0]
        N = interior_idx.size
        alpha_flat = np.zeros(flat_pts.shape[0], dtype=float)

        t0, tf = t_span
        t = t0
        steps = int(math.ceil((tf - t0)/dt))

        # integration loop
        for step in range(steps):
            if t >= tf:
                break
            current_dt = min(dt, tf - t)
            # get temperatures at time t for interior points
            T_vals = T_func(t, flat_pts[interior_idx])  # shape (N,)

            def deriv_vec(a_vec, time):
                # a_vec is array of alphas for interior points
                # compute rate elementwise
                out = np.empty_like(a_vec)
                for i in range(a_vec.size):
                    out[i] = kinetics.rate(float(a_vec[i]), float(T_vals[i]))
                return out

            # step alpha for interior points
            a_prev = alpha_flat[interior_idx]
            a_next = self._rk4_step(a_prev, t, current_dt, deriv_vec)
            # clamp
            a_next = np.clip(a_next, 0.0, 1.0)
            alpha_flat[interior_idx] = a_next
            t += current_dt

        # reshape back
        self.alpha = alpha_flat.reshape(nx, ny, nz)
        return self.alpha

    def integrate_alpha_plugflow(self, kinetics: Kinetics, T_profile: Callable[[float], float],
                                  axis: int = 2, velocity: float = 0.01, dz_sample: Optional[float]=None):
        """
        Simple plug-flow (axial) approximation along `axis` (0=x,1=y,2=z).
        We average cross-sections and compute a residence time for slices along the axis.

        T_profile(t) -> temperature as function of residence time. velocity in units of axis/second.
        dz_sample overrides grid spacing along axis; if None, uses grid spacing.
        """
        if self.grid_coords is None:
            raise RuntimeError("Grid not sampled. Call sample_grid() first.")
        coords = self.grid_coords
        sdfv = self.sdf_values
        mask = self.interior_mask
        nx, ny, nz, _ = coords.shape

        # determine axis coords and ordering
        axis_coords = coords[..., axis]
        axis_min = axis_coords.min()
        axis_max = axis_coords.max()

        # create slices along axis using existing grid points
        if axis == 0:
            num_slices = nx
            slice_centers = np.linspace(axis_min, axis_max, num_slices)
            slice_mask = mask.reshape(nx, ny*nz)
        elif axis == 1:
            num_slices = ny
            slice_centers = np.linspace(axis_min, axis_max, num_slices)
            slice_mask = mask.transpose(1,0,2).reshape(ny, nx*nz)
        else:
            num_slices = nz
            slice_centers = np.linspace(axis_min, axis_max, num_slices)
            slice_mask = mask.transpose(2,0,1).reshape(nz, nx*ny)

        # compute cross section area (in voxel count) for each slice
        cross_counts = np.sum(slice_mask, axis=1)
        # treat empty slices as zero area

        # residence time per slice: dt_slice = dz / velocity, where dz = slice spacing
        dz = (axis_max - axis_min) / (num_slices - 1)
        dt_slice = dz / velocity

        # integrate along plug flow: each slice sees a time history from t=0 to t=total_res_time
        # We'll march slices from inlet to outlet, carrying alpha with them.

        # initial alpha in all interior voxels = 0
        alpha = np.zeros_like(sdfv)

        # for simplicity assume inlet at axis_min
        # we compute alpha for representative voxel of each slice by integrating ODE over dt_slice
        # using temperature from T_profile at local residence time
        # This is a heavy simplification but useful as an approximation.

        rep_alpha = np.zeros(num_slices, dtype=float)
        residence = 0.0
        for i in range(num_slices):
            # local temperature for slice (could be function of residence/time)
            T_local = T_profile(residence)
            # integrate dalpha/dt for dt_slice starting from rep_alpha[i]
            def deriv_scalar(a, t_ignored):
                return kinetics.rate(a, T_local)

            a0 = rep_alpha[i-1] if i>0 else 0.0
            a_next = self._rk4_step(a0, residence, dt_slice, lambda a, t: np.array([deriv_scalar(float(a), t)])[0])[0]
            rep_alpha[i] = float(np.clip(a_next, 0.0, 1.0))
            residence += dt_slice

        # now broadcast rep_alpha back into 3D alpha field for all voxels in a slice
        if axis == 0:
            for ix in range(nx):
                slice_a = rep_alpha[ix]
                for j in range(ny):
                    for k in range(nz):
                        if mask[ix,j,k]:
                            alpha[ix,j,k] = slice_a
        elif axis == 1:
            for iy in range(ny):
                slice_a = rep_alpha[iy]
                for i in range(nx):
                    for k in range(nz):
                        if mask[i,iy,k]:
                            alpha[i,iy,k] = slice_a
        else:
            for iz in range(nz):
                slice_a = rep_alpha[iz]
                for i in range(nx):
                    for j in range(ny):
                        if mask[i,j,iz]:
                            alpha[i,j,iz] = slice_a

        self.alpha = alpha
        return alpha

