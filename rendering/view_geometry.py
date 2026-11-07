# Shared camera angles and parallel-scale framing for the PyVista renderer (render_pv.py).
import numpy as np

FILL_MARGIN = 1.06      # object fills frame to within ~6% margin (adaptive, never clips)

# 3/4-top views (reveal 3D form) + top/bottom (count features).
VIEWS = [
    ("front-right-top", 30, 45,  (0, 0, 1)),
    ("front-left-top",  30, 135, (0, 0, 1)),
    ("back-left-top",   30, 225, (0, 0, 1)),
    ("back-right-top",  30, 315, (0, 0, 1)),
    ("top",             89, 0,   (0, 1, 0)),
    ("bottom",         -89, 0,   (0, 1, 0)),
]
# Critic views: two opposite three-quarter views slightly above — together they cover every
# side and the top — plus straight top and bottom so the critic can judge the base/underside and footprint.
CRITIC_VIEWS = [
    ("front-right", 25, 45,  (0, 0, 1)),
    ("back-left",   25, 225, (0, 0, 1)),
    ("top",         89, 0,   (0, 1, 0)),
    ("bottom",     -89, 0,   (0, 1, 0)),
]


def view_dir(elev_deg, azim_deg):
    e = np.radians(elev_deg); a = np.radians(azim_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def fit_scale(points, view_list, margin=FILL_MARGIN):
    # largest parallel_scale needed so the object fits (no clip) across all given views.
    # adapts to the object's shape: thin objects get a smaller scale (appear larger), round
    # objects ~1.0. one scale for all views -> consistent size within the object.
    m = 1e-6
    for (_, elev, azim, up) in view_list:
        d = view_dir(elev, azim); fwd = -d / np.linalg.norm(d); up = np.array(up, float)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-8:
            up = np.array([0, 1.0, 0]) if abs(up[2]) > 0.9 else np.array([0, 0, 1.0])
            right = np.cross(fwd, up)
        right = right / np.linalg.norm(right); tup = np.cross(right, fwd)
        m = max(m, float(np.abs(points @ right).max()), float(np.abs(points @ tup).max()))
    return m * margin
