from __future__ import annotations
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from scipy.optimize import minimize

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri"],
        "font.size": 8,
        "text.usetex": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Calibri",
        "mathtext.it": "Calibri:italic",
        "mathtext.bf": "Calibri:bold",
        "mathtext.sf": "Calibri",
        "mathtext.tt": "Calibri",
        "mathtext.default": "it",
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

PI = np.pi
XMIN = -2.0 * PI
XMAX = 2.0 * PI
YMIN = -2.0 * PI
YMAX = 2.0 * PI
I_BIAS_NORM = 0.0
BETA_L_EPS = 1e-3
GRID_N = 121
CMAP = cm.get_cmap("jet")
ROOT_OUTPUT_DIR = Path(
    r"C:\Users\achintya\OneDrive - Chalmers\Documents\1. Project\6. Python Scripts\Plots & Analysis"
)

SURFACE_MODES = (
    "single_sector",
    "seamless",
)
DEFAULT_SURFACE_MODE = "single_sector"

DISPLAY_POTENTIAL_MODES = (
    "no_magnetic",
    "magnetic_clipped",
    "magnetic_full",
)
DEFAULT_DISPLAY_POTENTIAL_MODE = "magnetic_full"

MAGNETIC_DISPLAY_CAP = 3.0
DISPLAY_ZMAX = 8.0

# z-axis display compression
Z_VISUAL_SCALE = 0.78
VIEW_ELEV = 75.0
VIEW_AZIM = -85.0
DRAW_MARKER_STEMS = True
MARKER_SORT_ZPOS = DISPLAY_ZMAX + 10.0

MINIMA_MARKER_FONTSIZE = 8
TRACK_MARKER_FONTSIZE = 8
MINIMA_MARKER_HOVER = 0.1
TRACK_MARKER_HOVER = 0.15
MARKER_OUTLINE_WIDTH = 0.5

SYMM_SWITCH_DPHI_TOL = 0.35 * PI
SYMM_SWITCH_DVAR_TOL = 0.70 * PI


def output_dir_for_render_mode(display_mode: str, surface_mode: str) -> Path:
    if display_mode == "no_magnetic":
        base = ROOT_OUTPUT_DIR / "U_no_magnetic"
    elif display_mode == "magnetic_clipped":
        base = ROOT_OUTPUT_DIR / "U_magnetic_clipped"
    elif display_mode == "magnetic_full":
        base = ROOT_OUTPUT_DIR / "U_magnetic_full"
    else:
        raise ValueError(
            "display_mode must be 'no_magnetic', 'magnetic_clipped', or 'magnetic_full'"
        )

    if surface_mode not in {"single_sector", "seamless"}:
        raise ValueError("surface_mode must be 'single_sector' or 'seamless'")

    return base / surface_mode


@dataclass(frozen=True)
class StablePoint:
    phi_l: float
    varphi: float
    u: float
    m: int
    det_h: float
    evals: tuple[float, float]


@dataclass(frozen=True)
class DisplaySurface:
    Z_plot: np.ndarray
    z_shift: float
    color_norm: Normalize


def beta_eff(beta_l: float) -> float:
    return beta_l if beta_l > 0.0 else BETA_L_EPS


def wrap_periodic(
    x: float, period: float = 2.0 * PI, lo: float = -2.0 * PI, hi: float = 2.0 * PI
) -> float:
    w = ((x - lo) % period) + lo
    if w > hi:
        w -= period
    return w


def wrapped_delta(a: float, b: float, period: float = 2.0 * PI) -> float:
    d = a - b
    return ((d + 0.5 * period) % period) - 0.5 * period


def point_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    dphi = wrapped_delta(p1[0], p2[0])
    dvar = p1[1] - p2[1]
    return float(np.hypot(dphi, dvar))


def equivalent_copies(point: StablePoint, n_shift: int = 4) -> list[StablePoint]:
    copies: list[StablePoint] = []
    seen: set[tuple[float, float, int]] = set()

    for a in range(-n_shift, n_shift + 1):
        for b in range(-n_shift, n_shift + 1):
            phi_l = point.phi_l + 2.0 * PI * a + PI * b
            varphi = point.varphi + PI * b
            m = point.m + b

            if not (YMIN - 1e-12 <= phi_l <= YMAX + 1e-12):
                continue
            if not (XMIN - 1e-12 <= varphi <= XMAX + 1e-12):
                continue

            key = (round(phi_l, 12), round(varphi, 12), int(m))
            if key in seen:
                continue
            seen.add(key)

            copies.append(
                StablePoint(
                    phi_l=float(phi_l),
                    varphi=float(varphi),
                    u=point.u,
                    m=int(m),
                    det_h=point.det_h,
                    evals=point.evals,
                )
            )

    if not copies:
        copies = [point]

    return copies


def physical_distance(p1: StablePoint, p2: StablePoint) -> float:
    best = np.inf
    for q in equivalent_copies(p2):
        d = float(np.hypot(p1.phi_l - q.phi_l, p1.varphi - q.varphi))
        if d < best:
            best = d
    return best


def representative_near(
    point: StablePoint,
    ref_phi_l: float,
    ref_varphi: float,
) -> StablePoint:
    return min(
        equivalent_copies(point),
        key=lambda q: (
            abs(q.varphi - ref_varphi),
            abs(q.phi_l - ref_phi_l),
            abs(q.m),
            abs(q.phi_l),
            abs(q.varphi),
        ),
    )


def continuation_cost(
    current: StablePoint, previous: StablePoint
) -> tuple[float, float, float, float, float]:
    return (
        abs(current.varphi - previous.varphi),
        abs(current.phi_l - previous.phi_l),
        abs(current.u - previous.u),
        abs(current.m - previous.m),
        abs(current.phi_l),
    )


def display_representative(point: StablePoint, phi_e: float) -> StablePoint:
    return representative_near(point, ref_phi_l=0.0, ref_varphi=PI * phi_e)


def choose_anchor_state(alpha: float, beta_l: float) -> StablePoint:
    stable0 = stable_points_for_parameters(alpha, beta_l, 0.0)
    anchored = [display_representative(p, 0.0) for p in stable0]
    anchored = deduplicate_points(anchored, tol=5e-2)
    return min(
        anchored,
        key=lambda p: (p.u, abs(p.varphi), abs(p.phi_l), abs(p.m)),
    )


def equivalent_values_in_window(
    value: float,
    period: float = 2.0 * PI,
    lo: float = YMIN,
    hi: float = YMAX,
) -> list[float]:
    vals: list[float] = []
    n_min = math.floor((lo - value) / period) - 1
    n_max = math.ceil((hi - value) / period) + 1
    for n in range(n_min, n_max + 1):
        v = value + n * period
        if lo - 1e-12 <= v <= hi + 1e-12:
            vals.append(float(v))
    if not vals:
        vals = [float(np.clip(value, lo, hi))]
    vals = sorted(set(round(v, 12) for v in vals))
    return [float(v) for v in vals]


def closest_equivalent_in_window(
    value: float,
    reference: float,
    period: float = 2.0 * PI,
    lo: float = YMIN,
    hi: float = YMAX,
) -> float:
    vals = equivalent_values_in_window(value, period=period, lo=lo, hi=hi)
    return min(vals, key=lambda v: abs(v - reference))


def remap_point_phi_l_near_reference(
    point: StablePoint,
    reference_phi_l: float,
) -> StablePoint:
    phi_adj = closest_equivalent_in_window(point.phi_l, reference_phi_l)
    return StablePoint(
        phi_l=phi_adj,
        varphi=point.varphi,
        u=point.u,
        m=point.m,
        det_h=point.det_h,
        evals=point.evals,
    )


def expanded_phi_l_copies_for_plot(point: StablePoint) -> list[StablePoint]:
    copies: list[StablePoint] = []
    for phi_copy in equivalent_values_in_window(point.phi_l, lo=YMIN, hi=YMAX):
        copies.append(
            StablePoint(
                phi_l=phi_copy,
                varphi=point.varphi,
                u=point.u,
                m=point.m,
                det_h=point.det_h,
                evals=point.evals,
            )
        )
    return copies


def sector_m_values(phi_e: float) -> list[int]:
    lo = math.floor(-2.5 - phi_e)
    hi = math.ceil(2.5 - phi_e)
    return list(range(lo, hi + 1))


def sector_strip_mask(varphi: np.ndarray, phi_e: float, m_plot: int) -> np.ndarray:
    sector_coord = varphi / PI - phi_e
    return (sector_coord >= m_plot - 0.5) & (sector_coord < m_plot + 0.5)


def josephson_potential(
    phi_l: np.ndarray | float,
    varphi: np.ndarray | float,
    alpha: float,
    i_bias_norm: float = I_BIAS_NORM,
) -> np.ndarray | float:
    josephson = -(1.0 + alpha) * np.cos(phi_l + varphi) - (1.0 - alpha) * np.cos(
        phi_l - varphi
    )
    tilt = -i_bias_norm * phi_l
    return josephson + tilt


def magnetic_potential(
    varphi: np.ndarray | float, beta_l: float, phi_e: float, m: int
) -> np.ndarray | float:
    be = beta_eff(beta_l)
    return (2.0 * PI / be) * (varphi / PI - phi_e - m) ** 2


def squid_potential_sector(
    phi_l: np.ndarray | float,
    varphi: np.ndarray | float,
    alpha: float,
    beta_l: float,
    phi_e: float,
    m: int,
    i_bias_norm: float = I_BIAS_NORM,
) -> np.ndarray | float:
    return josephson_potential(
        phi_l, varphi, alpha, i_bias_norm=i_bias_norm
    ) + magnetic_potential(varphi, beta_l, phi_e, m)


def squid_potential_combo(
    phi_l: np.ndarray,
    varphi: np.ndarray,
    alpha: float,
    beta_l: float,
    phi_e: float,
    m_values: Iterable[int],
) -> np.ndarray:
    z_stack = []
    for m in m_values:
        z_stack.append(squid_potential_sector(phi_l, varphi, alpha, beta_l, phi_e, m))
    return np.min(np.stack(z_stack, axis=0), axis=0)


def displayed_sector_potential(
    phi_l: np.ndarray | float,
    varphi: np.ndarray | float,
    alpha: float,
    beta_l: float,
    phi_e: float,
    m: int,
    mode: str = DEFAULT_DISPLAY_POTENTIAL_MODE,
) -> np.ndarray | float:
    u_j = josephson_potential(phi_l, varphi, alpha)

    # beta_l = 0: show Josephson term only
    if beta_l <= 0.0:
        return u_j

    if mode == "no_magnetic":
        return u_j

    u_mag = magnetic_potential(varphi, beta_l, phi_e, m)

    if mode == "magnetic_clipped":
        return u_j + np.minimum(u_mag, MAGNETIC_DISPLAY_CAP)

    if mode == "magnetic_full":
        return u_j + u_mag

    raise ValueError(
        "display_mode must be 'no_magnetic', 'magnetic_clipped', or 'magnetic_full'"
    )


def displayed_combo_potential(
    phi_l: np.ndarray,
    varphi: np.ndarray,
    alpha: float,
    beta_l: float,
    phi_e: float,
    m_values: Iterable[int],
    mode: str = DEFAULT_DISPLAY_POTENTIAL_MODE,
) -> np.ndarray:
    z_stack = []
    for m in m_values:
        z_stack.append(
            displayed_sector_potential(phi_l, varphi, alpha, beta_l, phi_e, m, mode=mode)
        )
    return np.min(np.stack(z_stack, axis=0), axis=0)


def grad_sector(
    x: np.ndarray,
    alpha: float,
    beta_l: float,
    phi_e: float,
    m: int,
    i_bias_norm: float = I_BIAS_NORM,
) -> np.ndarray:
    phi_l, varphi = x
    be = beta_eff(beta_l)
    g_phi = (
        (1.0 + alpha) * np.sin(phi_l + varphi)
        + (1.0 - alpha) * np.sin(phi_l - varphi)
        - i_bias_norm
    )
    g_var = (
        (1.0 + alpha) * np.sin(phi_l + varphi)
        - (1.0 - alpha) * np.sin(phi_l - varphi)
        + (4.0 / be) * (varphi / PI - phi_e - m)
    )
    return np.array([g_phi, g_var], dtype=float)


def hessian_sector(
    phi_l: float, varphi: float, alpha: float, beta_l: float, phi_e: float, m: int
) -> np.ndarray:
    be = beta_eff(beta_l)
    c1 = (1.0 + alpha) * np.cos(phi_l + varphi)
    c2 = (1.0 - alpha) * np.cos(phi_l - varphi)
    u_phiphi = c1 + c2
    u_phivar = c1 - c2
    u_varvar = c1 + c2 + 4.0 / (PI * be)
    return np.array([[u_phiphi, u_phivar], [u_phivar, u_varvar]], dtype=float)


def psi0(varphi: float, alpha: float) -> float:
    return float(np.arctan2(alpha * np.sin(varphi), np.cos(varphi)))


def initial_seeds(
    alpha: float, beta_l: float, phi_e: float, m_values: Iterable[int]
) -> list[tuple[float, float, int]]:
    seeds: list[tuple[float, float, int]] = []

    for m in m_values:
        varphi0 = PI * (phi_e + m)
        if beta_l > 0.0:
            varphi_guesses = [varphi0 - 0.35 * PI, varphi0, varphi0 + 0.35 * PI]
        else:
            varphi_guesses = [varphi0]

        p0 = -psi0(varphi0, alpha)
        phi_candidates: list[float] = []
        for k in range(-2, 3):
            for candidate in (p0 + 2.0 * PI * k, p0 + PI + 2.0 * PI * k):
                if YMIN - 1e-12 <= candidate <= YMAX + 1e-12:
                    phi_candidates.append(float(candidate))

        for vg in varphi_guesses:
            if not (XMIN - 0.75 * PI <= vg <= XMAX + 0.75 * PI):
                continue
            for pg in phi_candidates:
                seeds.append((pg, vg, m))

    coarse_phi = np.linspace(YMIN, YMAX, 5)
    coarse_var = np.linspace(XMIN, XMAX, 5)
    for m in m_values:
        for vg in coarse_var:
            for pg in coarse_phi:
                seeds.append((float(pg), float(vg), int(m)))

    return seeds


def optimize_sector(
    seed_phi_l: float,
    seed_varphi: float,
    alpha: float,
    beta_l: float,
    phi_e: float,
    m: int,
) -> StablePoint | None:
    bounds = [(YMIN, YMAX), (XMIN, XMAX)]
    x0 = np.array([seed_phi_l, seed_varphi], dtype=float)

    fun = lambda x: float(squid_potential_sector(x[0], x[1], alpha, beta_l, phi_e, m))
    jac = lambda x: grad_sector(x, alpha, beta_l, phi_e, m)

    res = minimize(fun, x0, method="L-BFGS-B", jac=jac, bounds=bounds)
    if not res.success:
        return None

    phi_l = float(res.x[0])
    varphi = float(res.x[1])

    if not (YMIN <= phi_l <= YMAX and XMIN <= varphi <= XMAX):
        return None

    h = hessian_sector(phi_l, varphi, alpha, beta_l, phi_e, m)
    evals = np.linalg.eigvalsh(h)
    det_h = float(np.linalg.det(h))

    if not (evals[0] > 1e-6 and evals[1] > 1e-6 and det_h > 1e-6):
        return None

    u = float(squid_potential_sector(phi_l, varphi, alpha, beta_l, phi_e, m))

    return StablePoint(
        phi_l=phi_l,
        varphi=varphi,
        u=u,
        m=m,
        det_h=det_h,
        evals=(float(evals[0]), float(evals[1])),
    )


def deduplicate_points(points: list[StablePoint], tol: float = 5e-2) -> list[StablePoint]:
    unique: list[StablePoint] = []
    for p in sorted(
        points, key=lambda item: (item.u, abs(item.varphi), abs(item.phi_l), abs(item.m))
    ):
        duplicate = False
        for q in unique:
            if physical_distance(p, q) < tol:
                duplicate = True
                break
        if not duplicate:
            unique.append(p)
    return unique


def stable_points_for_parameters(
    alpha: float, beta_l: float, phi_e: float
) -> list[StablePoint]:
    points: list[StablePoint] = []

    if beta_l == 0.0:
        for m in sector_m_values(phi_e):
            varphi = PI * (phi_e + m)
            if not (XMIN <= varphi <= XMAX):
                continue

            phi0 = -psi0(varphi, alpha)
            for k in range(-2, 3):
                phi_l = phi0 + 2.0 * PI * k
                if not (YMIN <= phi_l <= YMAX):
                    continue

                u = float(squid_potential_sector(phi_l, varphi, alpha, beta_l, phi_e, m))
                points.append(
                    StablePoint(
                        phi_l=float(phi_l),
                        varphi=float(varphi),
                        u=u,
                        m=int(m),
                        det_h=np.inf,
                        evals=(np.inf, np.inf),
                    )
                )
    else:
        seeds = initial_seeds(alpha, beta_l, phi_e, sector_m_values(phi_e))
        for seed_phi_l, seed_varphi, m in seeds:
            p = optimize_sector(seed_phi_l, seed_varphi, alpha, beta_l, phi_e, m)
            if p is not None:
                points.append(p)

    return deduplicate_points(points, tol=5e-2)


def continue_state(
    previous: StablePoint,
    alpha: float,
    beta_l: float,
    phi_e: float,
    candidates: list[StablePoint],
) -> StablePoint:
    if beta_l > 0.0:
        trial = optimize_sector(
            previous.phi_l, previous.varphi, alpha, beta_l, phi_e, previous.m
        )
        if trial is not None:
            trial = representative_near(trial, previous.phi_l, previous.varphi)
            return trial

    same_m = [
        representative_near(p, previous.phi_l, previous.varphi)
        for p in candidates
        if p.m == previous.m
    ]
    all_m = [representative_near(p, previous.phi_l, previous.varphi) for p in candidates]

    pool = same_m if same_m else all_m
    pool = deduplicate_points(pool, tol=5e-2)

    return min(pool, key=lambda p: continuation_cost(p, previous))


def setup_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_xlim(2.0, -2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(0.0, DISPLAY_ZMAX)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_zticks([0, 8])
    ax.set_xlabel(r"$\varphi/\pi$", labelpad=-8, fontsize=8)
    ax.set_ylabel(r"$\phi_l/\pi$", labelpad=-8, fontsize=8)
    ax.set_zlabel("")
    ax.text2D(
        0.97,
        0.8,
        r"$U/E_0$",
        transform=ax.transAxes,
        fontsize=8,
        ha="center",
        va="center",
        rotation=90,
    )
    ax.tick_params(
        axis="both",
        which="major",
        pad=-4,
        labelsize=8,
        width=0.5,
    )
    ax.tick_params(
        axis="z",
        which="major",
        pad=-4,
        labelsize=8,
        width=0.5,
    )
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    fw, fh = ax.get_figure().get_size_inches()
    ax.set_box_aspect((1.0 * fw / fh, 1.0, Z_VISUAL_SCALE))

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo["grid"]["color"] = (0.68, 0.68, 0.68, 1.0)
        axis._axinfo["grid"]["linestyle"] = "-"
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.1

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


def interpolated_surface_height(
    xp: float, yp: float, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray
) -> float:
    xp = float(np.clip(xp, x_grid[0], x_grid[-1]))
    yp = float(np.clip(yp, y_grid[0], y_grid[-1]))

    ix1 = int(np.searchsorted(x_grid, xp, side="right"))
    iy1 = int(np.searchsorted(y_grid, yp, side="right"))
    ix1 = min(max(ix1, 1), len(x_grid) - 1)
    iy1 = min(max(iy1, 1), len(y_grid) - 1)
    ix0 = ix1 - 1
    iy0 = iy1 - 1

    x0 = float(x_grid[ix0])
    x1 = float(x_grid[ix1])
    y0 = float(y_grid[iy0])
    y1 = float(y_grid[iy1])

    tx = 0.0 if abs(x1 - x0) < 1e-15 else (xp - x0) / (x1 - x0)
    ty = 0.0 if abs(y1 - y0) < 1e-15 else (yp - y0) / (y1 - y0)

    z00 = float(z_grid[iy0, ix0])
    z10 = float(z_grid[iy0, ix1])
    z01 = float(z_grid[iy1, ix0])
    z11 = float(z_grid[iy1, ix1])

    z0 = (1.0 - tx) * z00 + tx * z10
    z1 = (1.0 - tx) * z01 + tx * z11
    return float((1.0 - ty) * z0 + ty * z1)


def overlay_projected_marker(
    ax: plt.Axes, x3d: float, y3d: float, z3d: float, color: str, fontsize: float
) -> None:
    x2, y2, _ = proj3d.proj_transform(x3d, y3d, z3d, ax.get_proj())
    txt = ax.annotate(
        "●",
        xy=(x2, y2),
        xycoords="data",
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        zorder=10000,
        annotation_clip=False,
    )
    txt.set_path_effects([pe.withStroke(linewidth=MARKER_OUTLINE_WIDTH, foreground="k")])


def overlay_projected_text(
    ax: plt.Axes,
    x3d: float,
    y3d: float,
    z3d: float,
    text: str,
    color: str,
    fontsize: float,
) -> None:
    x2, y2, _ = proj3d.proj_transform(x3d, y3d, z3d, ax.get_proj())
    txt = ax.annotate(
        text,
        xy=(x2, y2),
        xycoords="data",
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        zorder=10001,
        annotation_clip=False,
    )
    txt.set_path_effects([pe.withStroke(linewidth=MARKER_OUTLINE_WIDTH, foreground="k")])


def overlay_projected_arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    color: str = "#ffd400",
    linewidth: float = 1.5,
) -> None:
    x2a, y2a, _ = proj3d.proj_transform(x0, y0, z0, ax.get_proj())
    x2b, y2b, _ = proj3d.proj_transform(x1, y1, z1, ax.get_proj())

    ann = ax.annotate(
        "",
        xy=(x2b, y2b),
        xytext=(x2a, y2a),
        xycoords="data",
        textcoords="data",
        zorder=10000,
        annotation_clip=False,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=linewidth,
            mutation_scale=5,
            shrinkA=2,
            shrinkB=2,
            alpha=0.98,
        ),
    )
    ann.arrow_patch.set_path_effects(
        [pe.withStroke(linewidth=linewidth + 0.5, foreground="k")]
    )


def equivalent_phi_l_copies_in_window(phi_l: float) -> list[float]:
    copies = []
    for k in range(-2, 3):
        val = phi_l + 2.0 * PI * k
        if YMIN - 1e-12 <= val <= YMAX + 1e-12:
            copies.append(float(val))
    copies = sorted(set(round(v, 12) for v in copies))
    return [float(v) for v in copies]


def is_symmetric_branch_switch(
    previous_state: StablePoint | None,
    current_state: StablePoint,
    alpha: float,
    alpha_tol: float = 1e-12,
    dvar_tol: float = 0.35 * PI,
    dphi_tol: float = 0.30 * PI,
) -> bool:
    if previous_state is None:
        return False
    if abs(alpha) > alpha_tol:
        return False

    dvar = abs(current_state.varphi - previous_state.varphi)
    dphi_mod = abs(wrapped_delta(current_state.phi_l, previous_state.phi_l))

    return (dvar <= dvar_tol) and (abs(dphi_mod - PI) <= dphi_tol)


def branch_switch_display_targets(
    previous_state: StablePoint,
    switched_state: StablePoint,
    dphi_tol: float = SYMM_SWITCH_DPHI_TOL,
) -> list[StablePoint]:
    candidates: list[tuple[float, StablePoint]] = []

    for phi_copy in equivalent_values_in_window(switched_state.phi_l, lo=YMIN, hi=YMAX):
        dphi = phi_copy - previous_state.phi_l
        if abs(abs(dphi) - PI) <= dphi_tol:
            candidates.append(
                (
                    dphi,
                    StablePoint(
                        phi_l=float(phi_copy),
                        varphi=switched_state.varphi,
                        u=switched_state.u,
                        m=switched_state.m,
                        det_h=switched_state.det_h,
                        evals=switched_state.evals,
                    ),
                )
            )

    neg = sorted(
        [item for item in candidates if item[0] < 0.0],
        key=lambda item: abs(abs(item[0]) - PI),
    )
    pos = sorted(
        [item for item in candidates if item[0] > 0.0],
        key=lambda item: abs(abs(item[0]) - PI),
    )

    out: list[StablePoint] = []
    if neg:
        out.append(neg[0][1])
    if pos:
        out.append(pos[0][1])

    if len(out) < 2:
        leftovers = sorted(candidates, key=lambda item: abs(abs(item[0]) - PI))
        for _, p in leftovers:
            if all(abs(p.phi_l - q.phi_l) > 1e-9 for q in out):
                out.append(p)
            if len(out) == 2:
                break

    return out[:2]


def symmetric_switch_display_state(
    previous_state: StablePoint | None,
    current_state: StablePoint,
    alpha: float,
    dphi_tol: float = SYMM_SWITCH_DPHI_TOL,
    dvar_tol: float = SYMM_SWITCH_DVAR_TOL,
) -> StablePoint | None:
    if previous_state is None:
        return None
    if abs(alpha) > 1e-12:
        return None

    candidates = [
        representative_near(q, previous_state.phi_l, previous_state.varphi)
        for q in equivalent_copies(current_state)
    ]

    filtered: list[StablePoint] = []
    for q in candidates:
        dphi = abs(wrapped_delta(q.phi_l, previous_state.phi_l))
        dvar = abs(q.varphi - previous_state.varphi)
        if abs(dphi - PI) <= dphi_tol and dvar <= dvar_tol:
            filtered.append(q)

    if not filtered:
        return None

    return min(
        filtered,
        key=lambda q: (
            abs(q.varphi - previous_state.varphi),
            abs(abs(wrapped_delta(q.phi_l, previous_state.phi_l)) - PI),
            abs(q.phi_l),
            abs(q.m),
        ),
    )


def make_display_surface(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    beta_l: float,
    phi_e: float,
    tracked_state: StablePoint,
    surface_mode: str = DEFAULT_SURFACE_MODE,
    display_mode: str = DEFAULT_DISPLAY_POTENTIAL_MODE,
) -> tuple[DisplaySurface, list[StablePoint]]:
    if surface_mode == "single_sector":
        m_plot = tracked_state.m
        Z_raw = displayed_sector_potential(
            Y, X, alpha, beta_l, phi_e, m_plot, mode=display_mode
        )
        visible_points = [
            p for p in stable_points_for_parameters(alpha, beta_l, phi_e) if p.m == m_plot
        ]
    elif surface_mode == "seamless":
        Z_raw = displayed_combo_potential(
            Y, X, alpha, beta_l, phi_e, sector_m_values(phi_e), mode=display_mode
        )
        visible_points = stable_points_for_parameters(alpha, beta_l, phi_e)
    else:
        raise ValueError("surface_mode must be 'single_sector' or 'seamless'")

    z_min = float(np.min(Z_raw))
    Z_shifted = Z_raw - z_min
    z_max = float(np.max(Z_shifted))
    if z_max < 1e-12:
        z_max = 1.0
    Z_plot = np.clip(Z_shifted, 0.0, DISPLAY_ZMAX)
    color_norm = Normalize(vmin=0.0, vmax=float(np.max(Z_plot)))
    return (
        DisplaySurface(Z_plot=Z_plot, z_shift=z_min, color_norm=color_norm),
        visible_points,
    )


def displayed_point_height(
    phi_l: float,
    varphi: float,
    alpha: float,
    beta_l: float,
    phi_e: float,
    m: int,
    z_shift: float,
    display_mode: str = DEFAULT_DISPLAY_POTENTIAL_MODE,
) -> float:
    z = float(
        displayed_sector_potential(
            phi_l, varphi, alpha, beta_l, phi_e, m, mode=display_mode
        )
        - z_shift
    )
    return float(np.clip(z, 0.0, DISPLAY_ZMAX))


def interpolated_surface_height(
    xp: float, yp: float, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray
) -> float:
    xp = float(np.clip(xp, x_grid[0], x_grid[-1]))
    yp = float(np.clip(yp, y_grid[0], y_grid[-1]))

    ix1 = int(np.searchsorted(x_grid, xp, side="right"))
    iy1 = int(np.searchsorted(y_grid, yp, side="right"))
    ix1 = min(max(ix1, 1), len(x_grid) - 1)
    iy1 = min(max(iy1, 1), len(y_grid) - 1)
    ix0 = ix1 - 1
    iy0 = iy1 - 1

    x0 = float(x_grid[ix0])
    x1 = float(x_grid[ix1])
    y0 = float(y_grid[iy0])
    y1 = float(y_grid[iy1])

    tx = 0.0 if abs(x1 - x0) < 1e-15 else (xp - x0) / (x1 - x0)
    ty = 0.0 if abs(y1 - y0) < 1e-15 else (yp - y0) / (y1 - y0)

    z00 = float(z_grid[iy0, ix0])
    z10 = float(z_grid[iy0, ix1])
    z01 = float(z_grid[iy1, ix0])
    z11 = float(z_grid[iy1, ix1])

    z0 = (1.0 - tx) * z00 + tx * z10
    z1 = (1.0 - tx) * z01 + tx * z11
    return float((1.0 - ty) * z0 + ty * z1)


def scatter_hover_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    z_surface: float,
    hover: float,
    size: float,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    stem_color: str | None = None,
):
    z_marker = min(float(z_surface) + hover, DISPLAY_ZMAX - 1e-3)

    if DRAW_MARKER_STEMS and stem_color is not None:
        ax.plot([x, x], [y, y], [z_surface, z_marker], color=stem_color, linewidth=0.5)

    artist = ax.scatter(
        [x],
        [y],
        [z_marker],
        s=size,
        c=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        depthshade=False,
        clip_on=False,
    )

    try:
        artist.set_sort_zpos(MARKER_SORT_ZPOS)
    except Exception:
        pass

    return artist


def plot_single_landscape(
    alpha: float,
    beta_l: float,
    phi_e: float,
    tracked_state: StablePoint,
    output_path: Path,
    grid_n: int = GRID_N,
    surface_mode: str = DEFAULT_SURFACE_MODE,
    display_mode: str = DEFAULT_DISPLAY_POTENTIAL_MODE,
    previous_state: StablePoint | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(XMIN, XMAX, grid_n)
    y = np.linspace(YMIN, YMAX, grid_n)
    X, Y = np.meshgrid(x, y, indexing="xy")

    if surface_mode == "single_sector":
        m_plot = tracked_state.m
        Z_raw = displayed_sector_potential(
            Y, X, alpha, beta_l, phi_e, m_plot, mode=display_mode
        )

        # beta_l = 0: mask to selected sector for display
        if beta_l <= 0.0:
            keep = sector_strip_mask(X, phi_e, m_plot)
            Z_raw = np.ma.masked_where(~keep, Z_raw)

    elif surface_mode == "seamless":
        Z_raw = displayed_combo_potential(
            Y, X, alpha, beta_l, phi_e, sector_m_values(phi_e), mode=display_mode
        )
    else:
        raise ValueError("surface_mode must be 'single_sector' or 'seamless'")

    if np.ma.isMaskedArray(Z_raw):
        raw_vals = Z_raw.compressed()
    else:
        raw_vals = np.asarray(Z_raw).ravel()

    raw_vals = raw_vals[np.isfinite(raw_vals)]
    z_shift = float(np.min(raw_vals)) if raw_vals.size > 0 else 0.0
    Z_shifted = Z_raw - z_shift

    Z_for_markers = np.clip(Z_shifted, 0.0, DISPLAY_ZMAX)

    if display_mode == "magnetic_full" and beta_l > 0.0:
        Z_for_render = np.ma.masked_where(Z_shifted >= DISPLAY_ZMAX, Z_shifted)
    else:
        Z_for_render = np.clip(Z_shifted, 0.0, DISPLAY_ZMAX)

    if np.ma.isMaskedArray(Z_for_render):
        visible_vals = Z_for_render.compressed()
    else:
        visible_vals = np.asarray(Z_for_render).ravel()

    visible_vals = visible_vals[np.isfinite(visible_vals)]
    visible_max = float(np.max(visible_vals)) if visible_vals.size > 0 else 1.0
    visible_max = max(visible_max, 1e-12)

    color_norm = Normalize(vmin=0.0, vmax=visible_max)

    x_grid_plot = x / PI
    y_grid_plot = y / PI

    fig = plt.figure(figsize=(2.0, 1.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    setup_axes(ax)

    surf = ax.plot_surface(
        X / PI,
        Y / PI,
        Z_for_render,
        cmap=CMAP,
        norm=color_norm,
        rcount=grid_n,
        ccount=grid_n,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=False,
        rasterized=True,
    )
    Z_wire = np.array(Z_for_render, dtype=float)
    if np.ma.isMaskedArray(Z_for_render):
        Z_wire[Z_for_render.mask] = np.nan
    ax.plot_wireframe(
        X / PI,
        Y / PI,
        Z_wire,
        rstride=20,
        cstride=20,
        linewidth=0.025,
        color=(0, 0, 0, 0.5),
        rasterized=True,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.04, fraction=0.055)

    integer_ticks = np.arange(0, int(np.floor(visible_max)) + 1, 1)
    if integer_ticks.size == 0:
        integer_ticks = np.array([0])

    cbar.set_ticks(integer_ticks)
    cbar.set_ticklabels([str(int(t)) for t in integer_ticks])
    cbar.ax.tick_params(labelsize=8, width=0.5)
    cbar.ax.set_title(
        r"$U/E_0$",
        fontsize=8,
        pad=2,
    )

    fig.subplots_adjust(
        left=0,
        bottom=0,
        right=0.85,
        top=1,
    )
    fig.canvas.draw()

    switched_state = symmetric_switch_display_state(
        previous_state=previous_state,
        current_state=tracked_state,
        alpha=alpha,
    )

    if switched_state is not None and previous_state is not None:
        x0 = previous_state.varphi / PI
        y0 = previous_state.phi_l / PI
        z0m = interpolated_surface_height(x0, y0, x_grid_plot, y_grid_plot, Z_for_markers)
        z0m = min(z0m + TRACK_MARKER_HOVER, DISPLAY_ZMAX - 0.05)

        overlay_projected_marker(
            ax=ax,
            x3d=x0,
            y3d=y0,
            z3d=z0m,
            color="white",
            fontsize=TRACK_MARKER_FONTSIZE,
        )

        split_targets = branch_switch_display_targets(previous_state, switched_state)

        for idx, t in enumerate(split_targets, start=1):
            xt = t.varphi / PI
            yt = t.phi_l / PI
            zt = interpolated_surface_height(
                xt, yt, x_grid_plot, y_grid_plot, Z_for_markers
            )
            zt = min(zt + TRACK_MARKER_HOVER, DISPLAY_ZMAX - 0.05)

            overlay_projected_arrow(ax, x0, y0, z0m, xt, yt, zt)
            overlay_projected_marker(
                ax=ax,
                x3d=xt,
                y3d=yt,
                z3d=zt,
                color="white",
                fontsize=TRACK_MARKER_FONTSIZE,
            )
            overlay_projected_text(
                ax=ax,
                x3d=xt,
                y3d=yt,
                z3d=min(zt + 0.8, DISPLAY_ZMAX - 0.02),
                text=str(idx),
                color="#ffd400",
                fontsize=8,
            )
    else:
        xt = tracked_state.varphi / PI
        yt = tracked_state.phi_l / PI
        zt = interpolated_surface_height(xt, yt, x_grid_plot, y_grid_plot, Z_for_markers)
        zt = min(zt + TRACK_MARKER_HOVER, DISPLAY_ZMAX - 0.05)

        overlay_projected_marker(
            ax=ax,
            x3d=xt,
            y3d=yt,
            z3d=zt,
            color="white",
            fontsize=TRACK_MARKER_FONTSIZE,
        )

    fig.savefig(
        str(output_path),
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
    )
    svg_path = str(output_path.with_suffix(".svg"))
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
    )
    plt.close(fig)


def format_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def progress_line(
    done: int,
    total: int,
    display_mode: str,
    surface_mode: str,
    direction: str,
    alpha: float,
    beta_l: float,
    phi_e: float,
    elapsed_s: float,
) -> str:
    frac = done / total if total > 0 else 1.0
    eta_s = (elapsed_s / done) * (total - done) if done > 0 else 0.0
    return (
        f"[{done:05d}/{total:05d} | {100.0*frac:5.1f}% | ETA {format_hms(eta_s)}] "
        f"{display_mode:>16s} | {surface_mode:>13s} | {direction:>7s} | "
        f"alpha={alpha:.2f} | betaL={beta_l:.2f} | phi_e={phi_e:+.2f}"
    )


def run_sweep(
    alpha_values: Iterable[float],
    beta_l_values: Iterable[float],
    phi_e_values: Iterable[float],
    grid_n: int = GRID_N,
) -> None:
    for display_mode in DISPLAY_POTENTIAL_MODES:
        for surface_mode in SURFACE_MODES:
            output_dir_for_render_mode(display_mode, surface_mode).mkdir(
                parents=True, exist_ok=True
            )

    alpha_list = [float(v) for v in alpha_values]
    beta_list = [float(v) for v in beta_l_values]
    phi_all = [float(v) for v in np.round(sorted(set(phi_e_values)), 10)]
    phi_up_from_zero = [p for p in phi_all if p >= -1e-12]
    phi_down_from_zero = [p for p in reversed(phi_all) if p <= 1e-12]

    n_alpha = len(alpha_list)
    n_beta = len(beta_list)
    n_phi = len(phi_all)
    n_dir = 2
    n_display = len(DISPLAY_POTENTIAL_MODES)
    n_surface = len(SURFACE_MODES)

    total_renders = n_alpha * n_beta * n_phi * n_dir * n_display * n_surface
    render_count = 0
    t_start = time.perf_counter()

    def save_frame(
        alpha: float,
        beta_l: float,
        phi_e: float,
        tracked: StablePoint,
        stable: list[StablePoint],
        direction: str,
        previous_state: StablePoint | None,
        display_mode: str,
        surface_mode: str,
    ) -> None:
        nonlocal render_count

        filename = f"Potential_3D_{direction}_alpha{alpha:.2f}_phie{phi_e:+.2f}_betaL{beta_l:.2f}.png"
        outpath = (
            output_dir_for_render_mode(display_mode, surface_mode) / direction / filename
        )

        plot_single_landscape(
            alpha=alpha,
            beta_l=beta_l,
            phi_e=phi_e,
            tracked_state=tracked,
            output_path=outpath,
            grid_n=grid_n,
            surface_mode=surface_mode,
            display_mode=display_mode,
            previous_state=previous_state,
        )

        render_count += 1
        elapsed = time.perf_counter() - t_start
        print(
            progress_line(
                done=render_count,
                total=total_renders,
                display_mode=display_mode,
                surface_mode=surface_mode,
                direction=direction,
                alpha=alpha,
                beta_l=beta_l,
                phi_e=phi_e,
                elapsed_s=elapsed,
            )
        )

    for alpha in alpha_list:
        for beta_l in beta_list:
            anchor = choose_anchor_state(alpha, beta_l)

            tracked = anchor
            for phi_e in phi_down_from_zero[1:]:
                stable = stable_points_for_parameters(alpha, beta_l, phi_e)
                tracked = continue_state(tracked, alpha, beta_l, phi_e, stable)
            forward_start = tracked

            tracked = forward_start
            previous_state = None
            for i, phi_e in enumerate(phi_all):
                if abs(phi_e) < 1e-12:
                    tracked = anchor
                    stable = stable_points_for_parameters(alpha, beta_l, 0.0)
                else:
                    stable = stable_points_for_parameters(alpha, beta_l, phi_e)
                    if i > 0:
                        tracked = continue_state(tracked, alpha, beta_l, phi_e, stable)

                prev_plot = previous_state
                for display_mode in DISPLAY_POTENTIAL_MODES:
                    for surface_mode in SURFACE_MODES:
                        save_frame(
                            alpha=alpha,
                            beta_l=beta_l,
                            phi_e=phi_e,
                            tracked=tracked,
                            stable=stable,
                            direction="forward",
                            previous_state=prev_plot,
                            display_mode=display_mode,
                            surface_mode=surface_mode,
                        )
                previous_state = tracked

            tracked = anchor
            for phi_e in phi_up_from_zero[1:]:
                stable = stable_points_for_parameters(alpha, beta_l, phi_e)
                tracked = continue_state(tracked, alpha, beta_l, phi_e, stable)
            reverse_start = tracked

            tracked = reverse_start
            previous_state = None
            for i, phi_e in enumerate(reversed(phi_all)):
                if abs(phi_e) < 1e-12:
                    tracked = anchor
                    stable = stable_points_for_parameters(alpha, beta_l, 0.0)
                else:
                    stable = stable_points_for_parameters(alpha, beta_l, phi_e)
                    if i > 0:
                        tracked = continue_state(tracked, alpha, beta_l, phi_e, stable)

                prev_plot = previous_state
                for display_mode in DISPLAY_POTENTIAL_MODES:
                    for surface_mode in SURFACE_MODES:
                        save_frame(
                            alpha=alpha,
                            beta_l=beta_l,
                            phi_e=phi_e,
                            tracked=tracked,
                            stable=stable,
                            direction="reverse",
                            previous_state=prev_plot,
                            display_mode=display_mode,
                            surface_mode=surface_mode,
                        )
                previous_state = tracked

    total_elapsed = time.perf_counter() - t_start
    print(f"Completed {total_renders} renders in {format_hms(total_elapsed)}.")


def main() -> None:
    alpha_values = np.round(np.arange(0.0, 1.0 + 0.0001, 0.1), 10)
    beta_l_values = np.round(np.arange(0.0, 0.6 + 0.0001, 0.1), 10)
    phi_e_values = np.round(np.arange(-1.5, 1.5 + 0.0001, 0.05), 10)
    run_sweep(alpha_values, beta_l_values, phi_e_values, grid_n=GRID_N)


if __name__ == "__main__":
    main()
