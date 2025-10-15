# src/PyParticle/analysis/population/factory/dNdlnD.py
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from ...distributions import (
    make_edges,
    density1d_from_samples,
    density1d_cdf_map,
    kde1d_in_measure,
)

# import numpy as np
# from scipy.interpolate import PchipInterpolator  # shape-preserving

# def dndlnd_from_samples(Ds, weights, D_grid, tol=0.0, clip_nonneg=True):
#     """
#     Compute dN/dlnD on D_grid from (Ds, weights).
#     - Ds, D_grid: diameters (>0)
#     - weights: counts/weights (>=0)
#     - tol: merge duplicates within |ΔlnD| <= tol (0 means exact duplicates only)
#     """
#     Ds   = np.asarray(Ds, dtype=float)
#     w    = np.asarray(weights, dtype=float)
#     Dg   = np.asarray(D_grid, dtype=float)

#     # basic sanity
#     mask = (Ds > 0) & (w >= 0)
#     Ds, w = Ds[mask], w[mask]
#     if Ds.size == 0:
#         return np.zeros_like(Dg, dtype=float)

#     x  = np.log(Ds)      # work in ln D
#     xg = np.log(Dg)

#     # sort by x
#     idx = np.argsort(x)
#     x, w = x[idx], w[idx]

#     # coalesce duplicates in x (within tol in ln-space)
#     if tol > 0:
#         # starts of new groups where the gap exceeds tol
#         starts = np.r_[0, 1 + np.nonzero(np.diff(x) > tol)[0]]
#     else:
#         # exact duplicates only
#         _, starts = np.unique(x, return_index=True)
#     starts = np.sort(starts)
#     x_unique = x[starts]
#     w_unique = np.add.reduceat(w, starts)

#     # cumulative number vs lnD (this is the CDF)
#     cdf = np.cumsum(w_unique)

#     # monotone spline of CDF in lnD, then differentiate → dN/dlnD
#     F = PchipInterpolator(x_unique, cdf, extrapolate=True)
#     dens = F.derivative(1)(xg)

#     if clip_nonneg:
#         dens = np.clip(dens, 0.0, None)

#     return dens

@register_variable("dNdlnD")
class DNdlnDVar(PopulationVariable):
    meta = VariableMeta(
        name="dNdlnD",
        axis_names=("D",),
        description="Size distribution dN/dlnD",
        units="m$^{-3}$",
        scale="linear",
        long_label="Number size distribution",
        short_label="$dN/d\\ln D$",
    )
    
    def compute(self, population, as_dict=False):
        cfg = self.cfg
        method  = cfg.get("method", "hist")      # "hist"|"kde"|"provided"|"cdf_interp"|"direct"
        measure = "ln"                            # this variable is per dlnD

        # values & weights
        Ds = np.array([
            (population.get_particle(pid).get_Dwet()
             if cfg.get("wetsize", True)
             else population.get_particle(pid).get_Ddry())
            for pid in population.ids
        ])
        weights = np.asarray(getattr(population, "num_concs", np.ones_like(Ds)), dtype=float)

        # target grid (prefer edges if provided)
        edges = cfg.get("edges")
        D_grid = cfg.get("diam_grid")
        if edges is None:
            if D_grid is None:
                D_min = cfg.get("D_min", 1e-9); D_max = cfg.get("D_max", 2e-6)
                N_bins = cfg.get("N_bins", 50)
                edges, D_grid = make_edges(D_min, D_max, N_bins, scale="log")
            else:
                # infer geometric edges from centers
                r = np.sqrt(D_grid[1:] / D_grid[:-1])
                edges = np.empty(D_grid.size + 1)
                edges[1:-1] = D_grid[:-1] * r
                edges[0]     = D_grid[0] / r[0]
                edges[-1]    = D_grid[-1] * r[-1]

        # fixme: move to helpers
        if method == "hist":
            # fixme: move this out, but wanted to clear it up first!
            dens,_ = np.histogram(np.log(Ds), bins=np.log(edges), weights=weights)  # for checking
        elif method == "kde":
            dens = kde1d_in_measure(
                Ds, weights, D_grid, measure=measure, normalize=cfg.get("normalize", False)
            )
        # elif method == "provided":
        #     centers = np.asarray(cfg["src_D"])
        #     dens    = np.asarray(cfg["src_dNdlnD"], dtype=float)
        # elif method == "interp":
        #     idx = np.argsort(Ds)
        #     Ds_sorted = Ds[idx]
        #     weights_sorted = weights[idx]
        #     centers, dens, edges_tgt = density1d_cdf_map(Ds_sorted, weights_sorted, edges, measure=measure)
            #dens = dndlnd_from_samples(Ds, weights, D_grid, tol=0.0, clip_nonneg=True)
            
            # use provided source density if given; otherwise prebin particles on a fine log grid
            # idx = np.argsort(Ds)
            # Ds_sorted = Ds[idx]
            # Ns_sorted = weights[idx]
            # from scipy.interpolate import CubicSpline
            # dens = CubicSpline(x=np.log(Ds_sorted), y=np.cumsum(Ns_sorted), extrapolate=True).derivative(1)(np.log(D_grid))
        #     dNdlnD = np.interp(Ds_sorted, np.cumsum(Ns_sorted))

        #         centers, dens, _ = density1d_cdf_map(
        #             x_src_centers=np.asarray(cfg["src_D"]),
        #             dens_src=np.asarray(cfg["src_dNdlnD"], dtype=float),
        #             edges_tgt=edges, measure=measure,
        #         )
        #     else:
        #         fine_edges, fine_centers = make_edges(Ds.min()*0.9, Ds.max()*1.1, max(200, cfg.get("N_bins", 50)*4), "log")
        #         fine_centers, fine_dens, _ = density1d_from_samples(Ds, weights, fine_edges, measure=measure, normalize=False)
        #         centers, dens, _ = density1d_cdf_map(
        #             x_src_centers=fine_centers, dens_src=fine_dens, edges_tgt=edges, measure=measure
        #         )
        # elif method == "direct":
        #     centers = D_grid if D_grid is not None else Ds
        #     dens = np.zeros_like(centers)
        #     for d, w in zip(Ds, weights):
        #         dens[np.argmin(np.abs(centers - d))] += w
        else:
            raise ValueError(f"Unknown method {method}")
        
        out = {"D": D_grid, "dNdlnD": dens, "edges": edges}
        return out if as_dict else dens

def build(cfg=None):
    var = DNdlnDVar(cfg or {})
    if var.cfg.get("normalize"):
        var.meta.units = ""  # becomes probability density in lnD
    if not var.cfg.get("wetsize", True):
        var.meta.long_label = "Dry number size distribution"
        var.meta.short_label = "$dN/d\\ln D_{dry}$"
    return var
