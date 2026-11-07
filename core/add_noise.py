import numpy as np

from geomdl import fitting


def edge_points(points, nu, nv):
    edge_pts = []
    for x in range(nu):
        for y in range(nv):
            if x==0 or x==nu-1 or y==0 or y==nv-1:
                edge_pts.append(points[x,y])
    return np.array(edge_pts)


def add_noise(points, nu, nv, s=0.9, sigma=0.01, alpha=0.2):
    # p := p + (d*sigma/s) * N(0,I),   d = min(alpha, dist(p, nearest edge)).
    points = points.reshape((nu, nv, 3))
    edge_pts = edge_points(points, nu, nv)
    # Remap smoothness s -> effective denominator (larger denom = less noise), smooth/continuous:
    #   s >= 0.9        -> 1.0
    #   0.6 <= s < 0.9  -> linear ramp over input [0.6, 0.9], reaching 1.0 at 0.9 (joins smoothly)
    #   s < 0.6         -> s (unchanged)
    s = float(s)
    if s >= 0.9:
        s = 1.0
    elif s >= 0.6:
        s = 0.6 + (s - 0.6) * (1.0 - 0.6) / (0.9 - 0.6)
    s = max(s, 0.05)  # avoid divide-by-zero

    new_pts = []
    for x in range(nu):
        for y in range(nv):
            d = np.sqrt(np.sum((edge_pts - points[x, y]) ** 2, axis=1)).min()  # Euclidean
            d = min(d, alpha)                                                  # clip at alpha
            amp = d * sigma / s
            new_pts.append(points[x, y] + amp * np.random.randn(3))
    return np.array(new_pts)


DEFAULT_FREQ = 1
_FREQ_MIN, _FREQ_MAX = 1, 10


def resolve_freq(f):
    # Turn an LLM-authored f into a safe resample frequency: clamp to [_FREQ_MIN, _FREQ_MAX];
    # a missing/unparseable or non-positive f falls back to DEFAULT_FREQ.
    try:
        f = int(f)
    except (TypeError, ValueError):
        return DEFAULT_FREQ
    return min(max(f, _FREQ_MIN), _FREQ_MAX) if f > 0 else DEFAULT_FREQ


def resample(pts, nu, nv, frequency=10):

    surf = fitting.interpolate_surface(
                pts.tolist(),
                size_u=nu,
                size_v=nv,
                degree_u=min(nu-1, 2),
                degree_v=min(nv-1, 2)
            )

    new_nu = nu * frequency
    new_nv = nv * frequency

    u_list = np.linspace(0, 1, new_nu)
    v_list = np.linspace(0, 1, new_nv)

    new_pts = np.zeros((new_nu, new_nv, 3))

    for i,u in enumerate(u_list):
        for j,v in enumerate(v_list):
            new_pts[i, j] = surf.evaluate_single((u, v))

    return new_pts.reshape(-1,3), new_nu, new_nv
