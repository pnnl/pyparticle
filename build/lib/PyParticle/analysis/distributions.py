from __future__ import annotations
import numpy as np

try:
    from scipy.interpolate import PchipInterpolator as _PCHIP
    from scipy.interpolate import RegularGridInterpolator as _RGI
except Exception:
    _PCHIP = None
    _RGI = None

# ---------- Grid helpers ----------

def make_edges(xmin: float, xmax: float, n_bins: int, scale: str = "log"):
    """Return (edges, centers) on linear or logarithmic scale."""
    if scale == "log":
        edges = np.geomspace(xmin, xmax, n_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    elif scale == "linear":
        edges = np.linspace(xmin, xmax, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        raise ValueError("scale must be 'log' or 'linear'")
    return edges, centers

def bin_widths(edges: np.ndarray, measure: str = "ln"):
    """Widths in the measure of integration: 'ln' -> dlnx, 'linear' -> dx."""
    if measure == "ln":
        return np.log(edges[1:]) - np.log(edges[:-1])
    elif measure == "linear":
        return edges[1:] - edges[:-1]
    else:
        raise ValueError("measure must be 'ln' or 'linear'")

def _u_from_x(x: np.ndarray, measure: str):
    """Change of variable to the integration coordinate u."""
    if measure == "ln":
        return np.log(x)
    elif measure == "linear":
        return np.asarray(x)
    else:
        raise ValueError("measure must be 'ln' or 'linear'")

# ---------- 1D distributions ----------

def density1d_from_samples(
    x: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
    measure: str = "ln",
    normalize: bool = False,
):
    """
    Conservative histogram of samples into a density wrt the chosen measure.
    Returns (centers, density, edges).
    """
    x = np.asarray(x)
    w = np.asarray(weights, dtype=float)
    H, _ = np.histogram(x, bins=edges, weights=w)
    widths = bin_widths(edges, measure)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(widths > 0, H / widths, 0.0)
    if normalize:
        total = (dens * widths).sum()
        if total > 0:
            dens = dens / total
    centers = 0.5 * (edges[:-1] + edges[1:]) if measure == "linear" else np.sqrt(edges[:-1] * edges[1:])
    return centers, dens, edges

def density1d_cdf_map(
    x_src_centers: np.ndarray,
    dens_src: np.ndarray,
    edges_tgt: np.ndarray,
    measure: str = "ln",
):
    """
    Conservative mapping of a tabulated 1D density (per d{measure}x) onto target edges.
    Integrates the source to a CDF in u-space, interpolates (monotone if SciPy present),
    then differences to recover bin-mean densities on the target.
    """
    x_src = np.asarray(x_src_centers)
    y_src = np.asarray(dens_src, dtype=float)
    # Build edges around centers
    if measure == "ln":
        r = np.sqrt(x_src[1:] / x_src[:-1])
        src_edges = np.empty(x_src.size + 1)
        src_edges[1:-1] = x_src[:-1] * r
        src_edges[0] = x_src[0] / r[0]
        src_edges[-1] = x_src[-1] * r[-1]
    else:
        d = 0.5 * (x_src[1:] - x_src[:-1])
        src_edges = np.empty(x_src.size + 1)
        src_edges[1:-1] = 0.5 * (x_src[:-1] + x_src[1:])
        src_edges[0] = x_src[0] - d[0]
        src_edges[-1] = x_src[-1] + d[-1]

    # Integrate to CDF in u-space
    u_src = _u_from_x(x_src, measure)
    du_src = np.diff(_u_from_x(src_edges, measure))
    cell_N = y_src * du_src  # numbers per bin
    N_src = np.concatenate([[0.0], np.cumsum(cell_N)])  # CDF at src_edges

    u_edges_tgt = _u_from_x(edges_tgt, measure)
    if _PCHIP is not None and N_src.size >= 2:
        N_of_u = _PCHIP(_u_from_x(src_edges, measure), N_src, extrapolate=True)
        N_edges = N_of_u(u_edges_tgt)
    else:
        N_edges = np.interp(u_edges_tgt, _u_from_x(src_edges, measure), N_src, left=0.0, right=N_src[-1])

    widths = bin_widths(edges_tgt, measure)
    dN = np.maximum(0.0, np.diff(N_edges))
    with np.errstate(divide="ignore", invalid="ignore"):
        dens_tgt = np.where(widths > 0, dN / widths, 0.0)
    centers = 0.5 * (edges_tgt[:-1] + edges_tgt[1:]) if measure == "linear" else np.sqrt(edges_tgt[:-1] * edges_tgt[1:])
    return centers, dens_tgt, edges_tgt

def kde1d_in_measure(
    x: np.ndarray,
    weights: np.ndarray,
    xq: np.ndarray,
    measure: str = "ln",
    normalize: bool = False,
):
    """
    Smooth estimate of density wrt the chosen measure using a KDE performed in u-space.
    Returns values at query points xq (not bin-averaged).
    """
    try:
        from scipy.stats import gaussian_kde
    except Exception as e:
        raise RuntimeError("scipy is required for KDE") from e
    u = _u_from_x(np.asarray(x), measure)
    w = np.asarray(weights, dtype=float)
    kde = gaussian_kde(u, weights=w)
    dens = kde(_u_from_x(np.asarray(xq), measure))
    if normalize:
        # gaussian_kde yields a PDF that integrates to 1 over du; weights already handled.
        # If user normalized their weights, dens is already normalized.
        pass
    return dens

# ---------- 2D distributions ----------

def density2d_from_samples(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    edges_x: np.ndarray,
    edges_y: np.ndarray,
    measure_x: str = "ln",
    measure_y: str = "ln",
    normalize: bool = False,
):
    """
    Conservative 2D histogram -> density per d{measure_x}x d{measure_y}y.
    Returns (centers_x, centers_y, density, edges_x, edges_y).
    """
    H, ex, ey = np.histogram2d(x, y, bins=[edges_x, edges_y], weights=np.asarray(weights, dtype=float))
    wx = bin_widths(ex, measure_x)[:, None]
    wy = bin_widths(ey, measure_y)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where((wx > 0) & (wy > 0), H / (wx * wy), 0.0)
    if normalize:
        total = (dens * wx * wy).sum()
        if total > 0:
            dens = dens / total
    cx = 0.5 * (ex[:-1] + ex[1:]) if measure_x == "linear" else np.sqrt(ex[:-1] * ex[1:])
    cy = 0.5 * (ey[:-1] + ey[1:]) if measure_y == "linear" else np.sqrt(ey[:-1] * ey[1:])
    return cx, cy, dens, ex, ey

def density2d_cdf_map(
    src_edges_x: np.ndarray,
    src_edges_y: np.ndarray,
    dens_src: np.ndarray,  # per d{measure_x}x d{measure_y}y on src cells
    tgt_edges_x: np.ndarray,
    tgt_edges_y: np.ndarray,
    measure_x: str = "ln",
    measure_y: str = "ln",
):
    """
    Conservative mapping of a 2D density on a rectilinear source grid onto target edges.
    Uses the 2D CDF in (u_x, u_y), then inclusion-exclusion per target cell.
    """
    if _RGI is None:
        raise RuntimeError("scipy RegularGridInterpolator is required for 2D CDF mapping")

    ux_e_src = _u_from_x(src_edges_x, measure_x)
    uy_e_src = _u_from_x(src_edges_y, measure_y)
    dux = np.diff(ux_e_src)
    duy = np.diff(uy_e_src)

    # integrate density over source cells to get counts
    cell_N = dens_src * (dux[:, None] * duy[None, :])

    # Build CDF on edge grid (nx+1, ny+1)
    N = np.zeros((cell_N.shape[0] + 1, cell_N.shape[1] + 1))
    N[1:, 1:] = cell_N.cumsum(axis=0).cumsum(axis=1)

    # Interpolate CDF to target edge grid
    ux_e_tgt = _u_from_x(tgt_edges_x, measure_x)
    uy_e_tgt = _u_from_x(tgt_edges_y, measure_y)
    rgi = _RGI((ux_e_src, uy_e_src), N, bounds_error=False, fill_value=(N[-1, -1]))
    Ux, Uy = np.meshgrid(ux_e_tgt, uy_e_tgt, indexing="ij")
    Nt = rgi(np.stack([Ux, Uy], axis=-1))  # shape (Nx+1, Ny+1)

    # Inclusion-exclusion to recover counts per target cell
    dN = Nt[1:, 1:] - Nt[:-1, 1:] - Nt[1:, :-1] + Nt[:-1, :-1]
    dux_t = np.diff(ux_e_tgt)[:, None]
    duy_t = np.diff(uy_e_tgt)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        dens_tgt = np.where((dux_t > 0) & (duy_t > 0), dN / (dux_t * duy_t), 0.0)

    cx = 0.5 * (tgt_edges_x[:-1] + tgt_edges_x[1:]) if measure_x == "linear" else np.sqrt(tgt_edges_x[:-1] * tgt_edges_x[1:])
    cy = 0.5 * (tgt_edges_y[:-1] + tgt_edges_y[1:]) if measure_y == "linear" else np.sqrt(tgt_edges_y[:-1] * tgt_edges_y[1:])
    return cx, cy, dens_tgt, tgt_edges_x, tgt_edges_y

