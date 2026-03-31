from __future__ import annotations

import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
import numpy as np

ROOT_OUTPUT_DIR = Path(
    r"C:\Users\achintya\OneDrive - Chalmers\Documents\1. Project\6. Python Scripts\Plots & Analysis"
)

TOGGLE_TO_DISPLAY_MODE = {
    "no magnetic": "no_magnetic",
    "magnetic clipped": "magnetic_clipped",
    "magnetic full": "magnetic_full",
}

TOGGLE_TO_SURFACE_MODE = {
    "single sector": "single_sector",
    "seamless": "seamless",
}

DISPLAY_MODE_TO_FOLDER = {
    "no_magnetic": "U_no_magnetic",
    "magnetic_clipped": "U_magnetic_clipped",
    "magnetic_full": "U_magnetic_full",
}

ALPHA_VALUES = np.round(np.arange(0.0, 0.9 + 0.0001, 0.3), 1)
BETA_L_VALUES = np.round(np.arange(0.0, 0.6 + 0.0001, 0.3), 1)
PHI_E_VALUES = np.round(np.arange(-1.0, 1.0 + 0.0001, 0.05), 2)

PNG_RE = re.compile(
    r"^Potential_3D_(forward|reverse)_alpha([+-]?\d+\.\d+)_phie([+-]?\d+\.\d+)_betaL([+-]?\d+\.\d+)\.png$",
    re.IGNORECASE,
)


def canonical_zero(x: float, tol: float = 1e-12) -> float:
    return 0.0 if abs(float(x)) < tol else float(x)


def snap_to_allowed(value: float, allowed: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(allowed - value)))
    return canonical_zero(float(allowed[idx]))


def parse_numeric(text: str) -> float | None:
    try:
        return float(text.strip())
    except Exception:
        return None


def rounded_key(alpha: float, beta_l: float, phi_e: float) -> tuple[float, float, float]:
    return (
        round(canonical_zero(alpha), 2),
        round(canonical_zero(beta_l), 2),
        round(canonical_zero(phi_e), 2),
    )


def mode_root_dir(display_mode: str, surface_mode: str) -> Path:
    if display_mode not in DISPLAY_MODE_TO_FOLDER:
        raise ValueError(f"Unsupported display mode: {display_mode}")
    if surface_mode not in {"single_sector", "seamless"}:
        raise ValueError(f"Unsupported surface mode: {surface_mode}")
    return ROOT_OUTPUT_DIR / DISPLAY_MODE_TO_FOLDER[display_mode] / surface_mode


def find_direction_folder(root: Path, direction: str) -> Path:
    direction = direction.lower()
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == direction:
            return p
    raise FileNotFoundError(f"Could not find folder '{direction}' in:\n{root}")


def build_folder_index(folder: Path) -> dict[tuple[str, float, float, float], Path]:
    idx: dict[tuple[str, float, float, float], Path] = {}
    for p in folder.glob("*.png"):
        m = PNG_RE.match(p.name)
        if m is None:
            continue
        direction = m.group(1).lower()
        alpha = canonical_zero(float(m.group(2)))
        phi_e = canonical_zero(float(m.group(3)))
        beta_l = canonical_zero(float(m.group(4)))
        idx[(direction, round(alpha, 2), round(beta_l, 2), round(phi_e, 2))] = p
    return idx


def find_existing_png(
    display_mode: str,
    surface_mode: str,
    direction: str,
    alpha: float,
    beta_l: float,
    phi_e: float,
) -> Path | None:
    root = mode_root_dir(display_mode, surface_mode)
    folder = find_direction_folder(root, direction)
    idx = build_folder_index(folder)
    a, b, p = rounded_key(alpha, beta_l, phi_e)
    return idx.get((direction.lower(), a, b, p))


def main() -> None:
    fig = plt.figure(figsize=(15.2, 9.2), facecolor="white")
    ax_img = fig.add_axes([0.30, 0.15, 0.66, 0.78])
    ax_img.set_axis_off()

    ax_display = fig.add_axes([0.03, 0.70, 0.19, 0.20])
    radio_display = RadioButtons(
        ax_display,
        ("no magnetic", "magnetic clipped", "magnetic full"),
        active=1,
    )
    ax_display.set_title("Display", fontsize=12)

    ax_surface = fig.add_axes([0.03, 0.50, 0.19, 0.13])
    radio_surface = RadioButtons(
        ax_surface,
        ("single sector", "seamless"),
        active=0,
    )
    ax_surface.set_title("Surface", fontsize=12)

    # -------------------------------------------------------------------------
    # Slider rows: [slider] [value box] [reset button]
    # -------------------------------------------------------------------------
    ax_alpha = fig.add_axes([0.30, 0.09, 0.46, 0.03])
    ax_alpha_box = fig.add_axes([0.78, 0.085, 0.08, 0.04])
    ax_alpha_reset = fig.add_axes([0.88, 0.085, 0.07, 0.04])

    ax_beta = fig.add_axes([0.30, 0.055, 0.46, 0.03])
    ax_beta_box = fig.add_axes([0.78, 0.050, 0.08, 0.04])
    ax_beta_reset = fig.add_axes([0.88, 0.050, 0.07, 0.04])

    ax_phi = fig.add_axes([0.30, 0.020, 0.46, 0.03])
    ax_phi_box = fig.add_axes([0.78, 0.015, 0.08, 0.04])
    ax_phi_reset = fig.add_axes([0.88, 0.015, 0.07, 0.04])

    slider_alpha = Slider(
        ax=ax_alpha,
        label=r"$\alpha$",
        valmin=float(ALPHA_VALUES[0]),
        valmax=float(ALPHA_VALUES[-1]),
        valinit=0.0,
        valstep=ALPHA_VALUES,
        valfmt="%.2f",
    )
    slider_beta = Slider(
        ax=ax_beta,
        label=r"$\beta_L$",
        valmin=float(BETA_L_VALUES[0]),
        valmax=float(BETA_L_VALUES[-1]),
        valinit=0.0,
        valstep=BETA_L_VALUES,
        valfmt="%.2f",
    )
    slider_phi = Slider(
        ax=ax_phi,
        label=r"$\Phi_e/\Phi_0$",
        valmin=float(PHI_E_VALUES[0]),
        valmax=float(PHI_E_VALUES[-1]),
        valinit=0.0,
        valstep=PHI_E_VALUES,
        valfmt="%.2f",
    )

    box_alpha = TextBox(ax_alpha_box, "", initial="0.00")
    box_beta = TextBox(ax_beta_box, "", initial="0.00")
    box_phi = TextBox(ax_phi_box, "", initial="0.00")

    btn_alpha_reset = Button(ax_alpha_reset, "Reset")
    btn_beta_reset = Button(ax_beta_reset, "Reset")
    btn_phi_reset = Button(ax_phi_reset, "Reset")

    state = {
        "prev_phi": 0.0,
        "direction": "forward",
        "syncing_widgets": False,
    }

    def current_values() -> tuple[float, float, float]:
        alpha = canonical_zero(snap_to_allowed(slider_alpha.val, ALPHA_VALUES))
        beta_l = canonical_zero(snap_to_allowed(slider_beta.val, BETA_L_VALUES))
        phi_e = canonical_zero(snap_to_allowed(slider_phi.val, PHI_E_VALUES))
        return alpha, beta_l, phi_e

    def set_box_text(box: TextBox, text: str) -> None:
        state["syncing_widgets"] = True
        box.set_val(text)
        state["syncing_widgets"] = False

    def sync_text_boxes() -> None:
        alpha, beta_l, phi_e = current_values()
        set_box_text(box_alpha, f"{alpha:.2f}")
        set_box_text(box_beta, f"{beta_l:.2f}")
        set_box_text(box_phi, f"{phi_e:.2f}")

    def redraw() -> None:
        alpha, beta_l, phi_e = current_values()

        display_mode = TOGGLE_TO_DISPLAY_MODE[str(radio_display.value_selected)]
        surface_mode = TOGGLE_TO_SURFACE_MODE[str(radio_surface.value_selected)]

        prev_phi = float(state["prev_phi"])
        if phi_e > prev_phi + 1e-12:
            state["direction"] = "forward"
        elif phi_e < prev_phi - 1e-12:
            state["direction"] = "reverse"
        state["prev_phi"] = phi_e

        direction = str(state["direction"])
        path = find_existing_png(
            display_mode=display_mode,
            surface_mode=surface_mode,
            direction=direction,
            alpha=alpha,
            beta_l=beta_l,
            phi_e=phi_e,
        )

        ax_img.clear()
        ax_img.set_axis_off()

        if path is not None and path.exists():
            image = mpimg.imread(path)
            ax_img.imshow(image)
            fig.canvas.manager.set_window_title(path.name)
        else:
            ax_img.text(
                0.5,
                0.5,
                "PNG not found",
                ha="center",
                va="center",
                fontsize=20,
                transform=ax_img.transAxes,
            )
            fig.canvas.manager.set_window_title("PNG not found")

        ax_img.set_title(
            rf"$\alpha={alpha:.2f},\ \beta_L={beta_l:.2f},\ \Phi_e/\Phi_0={phi_e:+.2f}$"
            + "\n"
            + rf"${direction}\ sweep,\ {display_mode},\ {surface_mode}$",
            fontsize=16,
            pad=10,
        )

        sync_text_boxes()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def update(_: object | None = None) -> None:
        if state["syncing_widgets"]:
            return
        redraw()

    # -------------------------------------------------------------------------
    # Text-box submit handlers
    # -------------------------------------------------------------------------
    def submit_alpha(text: str) -> None:
        if state["syncing_widgets"]:
            return
        value = parse_numeric(text)
        if value is None:
            sync_text_boxes()
            return
        snapped = snap_to_allowed(value, ALPHA_VALUES)
        slider_alpha.set_val(snapped)

    def submit_beta(text: str) -> None:
        if state["syncing_widgets"]:
            return
        value = parse_numeric(text)
        if value is None:
            sync_text_boxes()
            return
        snapped = snap_to_allowed(value, BETA_L_VALUES)
        slider_beta.set_val(snapped)

    def submit_phi(text: str) -> None:
        if state["syncing_widgets"]:
            return
        value = parse_numeric(text)
        if value is None:
            sync_text_boxes()
            return

        old_phi = canonical_zero(snap_to_allowed(slider_phi.val, PHI_E_VALUES))
        new_phi = snap_to_allowed(value, PHI_E_VALUES)

        if new_phi > old_phi + 1e-12:
            state["direction"] = "forward"
        elif new_phi < old_phi - 1e-12:
            state["direction"] = "reverse"

        state["prev_phi"] = old_phi
        slider_phi.set_val(new_phi)

    # -------------------------------------------------------------------------
    # Reset button handlers
    # -------------------------------------------------------------------------
    def reset_alpha(_: object) -> None:
        slider_alpha.set_val(0.0)

    def reset_beta(_: object) -> None:
        slider_beta.set_val(0.0)

    def reset_phi(_: object) -> None:
        state["direction"] = "forward"
        state["prev_phi"] = 0.0
        slider_phi.set_val(0.0)

    # -------------------------------------------------------------------------
    # Wire up callbacks
    # -------------------------------------------------------------------------
    slider_alpha.on_changed(update)
    slider_beta.on_changed(update)
    slider_phi.on_changed(update)

    radio_display.on_clicked(update)
    radio_surface.on_clicked(update)

    box_alpha.on_submit(submit_alpha)
    box_beta.on_submit(submit_beta)
    box_phi.on_submit(submit_phi)

    btn_alpha_reset.on_clicked(reset_alpha)
    btn_beta_reset.on_clicked(reset_beta)
    btn_phi_reset.on_clicked(reset_phi)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()
