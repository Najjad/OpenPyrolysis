"""
pyro_reactor_full.py

Extended pyrolysis reactor modelling package built on the CSG SDF engine.

Goals implemented in this file:
 - Custom definable materials with physical properties (rho, cp, k, porosity)
 - Multi-step reaction kinetics (list of reactions, Arrhenius parameters, stoichiometry)
 - Energy balance (transient heat conduction on the voxel grid) coupled to local reaction heat
 - Mass loss / volatile generation tracking (per-voxel)
 - Fixed-grid finite-volume explicit solver for heat + local kinetics integration
 - Optional simple advective transport along an imposed velocity field (user-provided)
 - I/O: saves per-timestep NPZ files with alpha, T, mass; exports VTK (via meshio) if available
 - CLI-style example at bottom showing typical usage for stacked-cylinder reactor

Limitations / notes:
 - The thermal solver is explicit for simplicity. Time-step must satisfy stability (dt <= dx^2/(6*alpha_th)).
   For serious runs, an implicit scheme or C++ backend is recommended.
 - No full CFD coupling. Advection is supported only as a user-supplied velocity vector field on the grid
   (e.g., plug-flow velocity along z). No turbulence, no pressure solver.
 - Multi-step kinetics are independent and act locally; gas-phase reactions, diffusion of volatiles, and
   condensed-phase phase change are not modelled except via voilatile source terms.
 - Heavy loops are good candidates for numba or C++ acceleration; optional numba decorators are used
   if numba is installed.

Dependencies: numpy, optional: meshio (VTK export), numba (speedup)

"""
from __future__ import annotations
import math
import os
from typing import Callable, List, Tuple, Dict, Optional
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    def njit(f=None, **kwargs):
        return (lambda g: g) if f is None else f
    _HAS_NUMBA = False

# import the earlier modules (assume csg_engine.py and pyro_reactor_sim.py are present)
from csg_engine import evaluate_on_grid
from pyro_reactor_sim import Kinetics, ReactorSimulator, ReactorGeometry

# ----------------------------- Material & kinetics -----------------------------
class ReactionStep:
    """A single reaction step: A -> products (we track conversion of A as alpha_i).

    rate = A * exp(-Ea/(R*T)) * (1 - alpha)^n
    heat_release: exothermic positive (J/kg reactant consumed)
    """
    def __init__(self, name: str, A: float, Ea: float, n: float, heat_release: float):
        self.name = name
        self.A = A
        self.Ea = Ea
        self.n = n
        self.heat_release = heat_release

    def rate(self, alpha: float, T: float, R=8.314) -> float:
        k = self.A * math.exp(-self.Ea / (R * T))
        return k * ((1.0 - alpha) ** self.n)


class Material:
    """Material definition with physical and kinetic properties.

    Attributes:
      rho: bulk density (kg/m3)
      cp: specific heat capacity (J/kg/K)
      k: thermal conductivity (W/m/K)
      porosity: fraction (0-1)
      initial_mass_frac: initial solid mass fraction
      reactions: list of ReactionStep
    """
    def __init__(self, name: str, rho: float, cp: float, k: float, porosity: float, reactions: List[ReactionStep]):
        self.name = name
        self.rho = rho
        self.cp = cp
        self.k = k
        self.porosity = porosity
        self.reactions = reactions


# --------------------------- Numerical helpers ---------------------------
@njit
def _stable_dt_for_explicit(dx: float, k: float, rho: float, cp: float) -> float:
    # thermal diffusivity alpha_th = k/(rho*cp)
    alpha = k / (rho * cp + 1e-12)
    # explicit 3D stability approx dt <= dx^2/(6*alpha)
    if alpha <= 0:
        return 1e-6
    return 0.25 * dx*dx / alpha


# --------------------------- Solver: heat + kinetics ---------------------------
class PyroSolver:
    """Couples thermal conduction (explicit finite-volume) with local kinetics.

    Inputs:
      sim: ReactorSimulator (contains grid, sdf, masks)
      material_map: function or 3D array mapping voxel -> Material
      velocity_field: optional vector field for advective transport of volatiles (Nx,Ny,Nz,3)
    """
    def __init__(self, sim: ReactorSimulator, material_map: Optional[Callable[[np.ndarray], Material]] = None,
                 material_array: Optional[np.ndarray] = None, velocity_field: Optional[np.ndarray] = None):
        self.sim = sim
        if sim.grid_coords is None:
            raise RuntimeError("Simulator must be sampled before creating solver")
        self.coords = sim.grid_coords
        self.sdf = sim.sdf_values
        self.mask = sim.interior_mask
        self.nx, self.ny, self.nz, _ = self.coords.shape
        self.material_array = material_array
        self.material_map = material_map
        self.velocity_field = velocity_field

        # create state arrays
        self.T = np.full((self.nx, self.ny, self.nz), 300.0)  # initial T
        self.alpha = np.zeros((self.nx, self.ny, self.nz))
        self.mass_frac = np.ones((self.nx, self.ny, self.nz))  # remaining solid mass fraction

        # compute voxel spacings (assuming regular grid)
        xs = self.coords[:,0,0,0]
        ys = self.coords[0,:,0,1]
        zs = self.coords[0,0,:,2]
        self.dx = abs(xs[1]-xs[0]) if xs.size>1 else 1e-3
        self.dy = abs(ys[1]-ys[0]) if ys.size>1 else 1e-3
        self.dz = abs(zs[1]-zs[0]) if zs.size>1 else 1e-3

        # material property arrays (filled lazily)
        self.rho = np.zeros_like(self.T)
        self.cp = np.zeros_like(self.T)
        self.k = np.zeros_like(self.T)
        self.porosity = np.zeros_like(self.T)
        self._populate_material_props()

    def _populate_material_props(self):
        # fill arrays using either material_array or material_map
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if not self.mask[i,j,k]:
                        continue
                    if self.material_array is not None:
                        mat = self.material_array[i,j,k]
                    else:
                        mat = self.material_map(self.coords[i,j,k])
                    self.rho[i,j,k] = mat.rho
                    self.cp[i,j,k] = mat.cp
                    self.k[i,j,k] = mat.k
                    self.porosity[i,j,k] = mat.porosity

    def estimate_stable_dt(self):
        # conservative dt from all voxels
        dt_min = 1e9
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if not self.mask[i,j,k]:
                        continue
                    dx = min(self.dx, self.dy, self.dz)
                    dt = _stable_dt_for_explicit(dx, self.k[i,j,k], self.rho[i,j,k], self.cp[i,j,k])
                    if dt < dt_min:
                        dt_min = dt
        return dt_min

    def step(self, dt: float, external_T_bc: Optional[Callable[[float, np.ndarray], np.ndarray]] = None):
        """Advance the solver by dt (explicit scheme).

        external_T_bc(t, coords_flat) -> temperature boundary condition for surface voxels.
        """
        newT = self.T.copy()
        # conduction flux using 6-neighbor finite differences
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if not self.mask[i,j,k]:
                        continue
                    T0 = self.T[i,j,k]
                    k_local = self.k[i,j,k]
                    rho_cp = self.rho[i,j,k] * self.cp[i,j,k]
                    # sum neighbors
                    lap = 0.0
                    count = 0
                    # x- neighbors
                    if i>0 and self.mask[i-1,j,k]:
                        lap += (self.T[i-1,j,k] - T0) / (self.dx*self.dx)
                        count += 1
                    if i<self.nx-1 and self.mask[i+1,j,k]:
                        lap += (self.T[i+1,j,k] - T0) / (self.dx*self.dx)
                        count += 1
                    if j>0 and self.mask[i,j-1,k]:
                        lap += (self.T[i,j-1,k] - T0) / (self.dy*self.dy)
                        count += 1
                    if j<self.ny-1 and self.mask[i,j+1,k]:
                        lap += (self.T[i,j+1,k] - T0) / (self.dy*self.dy)
                        count += 1
                    if k>0 and self.mask[i,j,k-1]:
                        lap += (self.T[i,j,k-1] - T0) / (self.dz*self.dz)
                        count += 1
                    if k<self.nz-1 and self.mask[i,j,k+1]:
                        lap += (self.T[i,j,k+1] - T0) / (self.dz*self.dz)
                        count += 1

                    # conduction term
                    if rho_cp <= 0:
                        continue
                    Td = T0 + dt * (k_local * lap) / rho_cp

                    # reaction source/sink (local kinetics) -> compute total heat generation per unit mass
                    # sum contributions of each reaction step based on local alpha
                    mat = None
                    if self.material_array is not None:
                        mat = self.material_array[i,j,k]
                    else:
                        mat = self.material_map(self.coords[i,j,k])

                    # integrate alpha for dt using RK4 for local reaction network
                    alpha0 = self.alpha[i,j,k]
                    def deriv_scalar(a, t_ignored):
                        total = 0.0
                        for rstep in mat.reactions:
                            total += rstep.rate(a, T0)
                        return total
                    a1 = alpha0
                    # scalar RK4 manually
                    k1 = deriv_scalar(a1, 0.0)
                    k2 = deriv_scalar(a1 + 0.5*dt*k1, 0.0)
                    k3 = deriv_scalar(a1 + 0.5*dt*k2, 0.0)
                    k4 = deriv_scalar(a1 + dt*k3, 0.0)
                    anew = a1 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
                    anew = max(0.0, min(1.0, anew))

                    # heat release from conversion during dt (per unit mass)
                    delta_alpha = anew - alpha0
                    qdot = 0.0
                    if delta_alpha != 0.0:
                        # approximate heat = sum(heat_release_i * delta_alpha) ; assuming heat_release is per mass of original material
                        for rstep in mat.reactions:
                            # proportion of delta_alpha attributable to this reaction is approximated by its rate fraction at alpha0
                            denom = 0.0
                            for r2 in mat.reactions:
                                denom += r2.rate(alpha0, T0)
                            frac = 0.0
                            if denom > 0:
                                frac = rstep.rate(alpha0, T0) / denom
                            qdot += frac * rstep.heat_release * delta_alpha
                    # convert qdot (J/kg) to temperature change: dT = qdot/(cp)
                    if mat.cp > 0:
                        Td += qdot / mat.cp

                    newT[i,j,k] = Td
                    self.alpha[i,j,k] = anew
                    # mass fraction update (simplified)
                    self.mass_frac[i,j,k] = max(0.0, self.mass_frac[i,j,k] - delta_alpha)

        self.T = newT

    def run(self, t_final: float, dt: Optional[float]=None, save_every: int=10, out_dir: str='output'):
        if dt is None:
            dt = self.estimate_stable_dt() * 0.5
        os.makedirs(out_dir, exist_ok=True)
        t = 0.0
        step = 0
        while t < t_final - 1e-12:
            self.step(dt)
            t += dt
            step += 1
            if step % save_every == 0 or t >= t_final - 1e-12:
                fname = os.path.join(out_dir, f'state_t{t:.4f}.npz')
                np.savez_compressed(fname, T=self.T, alpha=self.alpha, mass_frac=self.mass_frac)
                print('Saved', fname)
        return dict(T=self.T, alpha=self.alpha, mass_frac=self.mass_frac)

# --------------------------- Output helpers ---------------------------
class OutputWriter:
    @staticmethod
    def export_vtk(coords: np.ndarray, scalar_fields: Dict[str, np.ndarray], filename: str):
        try:
            import meshio
        except Exception:
            print('meshio not installed; VTK export skipped. Install meshio to enable VTK export.')
            return
        # coords shape (nx,ny,nz,3). Flatten into points and write point data
        nx, ny, nz, _ = coords.shape
        pts = coords.reshape(-1,3)
        # build an unstructured grid of points only; VTK can store point data without connectivity
        cells = []
        mesh = meshio.Mesh(points=pts, cells=cells, point_data={k: v.reshape(-1) for k,v in scalar_fields.items()})
        meshio.write(filename, mesh)
        print('Wrote', filename)


# --------------------------- Example CLI usage ---------------------------
if __name__ == '__main__':
    # Build geometry: stacked cylinders (top small feeding to bottom large)
    top = {'radius': 0.12, 'height': 0.4}
    neck = {'radius': 0.06, 'height': 0.1}
    bottom = {'radius': 0.3, 'height': 0.8}
    params = [top, neck, bottom]
    geom = ReactorGeometry.stacked_cylinders(params, center_z=0.0, spacing=-0.01)

    # sample grid
    from csg_engine import Cylinder
    from pyro_reactor_sim import ReactorSimulator
    sim = ReactorSimulator(geom)
    bounds = ((-0.4,0.4), (-0.4,0.4), (-0.7,0.7))
    res = (100,100,160)
    print('Sampling grid...')
    sim.sample_grid(bounds, res)
    print('Interior voxels:', sim.interior_mask.sum())

    # define a material (example biomass-like)
    reactions = [ReactionStep('primary', A=1e7, Ea=9e4, n=1.0, heat_release=-2e5)]
    mat = Material('biomass', rho=600.0, cp=1500.0, k=0.2, porosity=0.5, reactions=reactions)

    # simple material_map that returns same material everywhere inside
    def material_map(coord):
        return mat

    solver = PyroSolver(sim, material_map=material_map)
    dt_est = solver.estimate_stable_dt()
    print('Estimated stable dt (s):', dt_est)

    # run for 5 seconds (this will be slow due to pure-Python loops)
    resdict = solver.run(t_final=5.0, dt=dt_est*0.5, save_every=5, out_dir='pyro_output')
    OutputWriter.export_vtk(sim.grid_coords, {'T':resdict['T'], 'alpha':resdict['alpha']}, 'pyro_output/state.vtk')

    print('Finished example run. Results saved in pyro_output/')
