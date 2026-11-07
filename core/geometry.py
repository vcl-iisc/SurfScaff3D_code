import re
from collections import namedtuple

import numpy as np
from geomdl import fitting

# A parsed patch: its nu*nv grid of on-surface points, plus the per-patch
# smoothness s, the per-patch detail frequency f (LLM-authored resample density;
# None when the face omits it -> resolve_freq() falls back to DEFAULT_FREQ), and the
# raw label line from the "# obj/part/sub  s=0.95 f=6" comment.
Patch = namedtuple("Patch", "pts nu nv s label f", defaults=(None,))
_S_RE = re.compile(r's\s*=\s*([0-9]*\.?[0-9]+)')   # a trailing '.' (sentence end) is not part of the number
_F_RE = re.compile(r'f\s*=\s*([0-9]+)')


def parse_faces(text):
    # The leading integer is an OPTIONAL count
    # nu/nv headers that disagree with the actual point count are repaired, and
    # degenerate / unparseable faces are dropped with a warning instead of raising.
    # Faces are delimited by their '#' label lines.
    matches = list(re.finditer(r'<faces>(.*?)</faces>', text, re.DOTALL))
    if not matches:
        raise ValueError("No <faces> block found in response")
    lines = [l.strip() for l in matches[-1].group(1).splitlines() if l.strip()]

    declared_n, i = None, 0
    if lines and not lines[0].startswith("#"):
        try:
            declared_n = int(lines[0]); i = 1
        except ValueError:
            pass

    faces, warnings = [], []
    while i < len(lines):
        # Capture the label line ("# obj/part/sub  s=0.95 f=6") to recover smoothness s
        # and the LLM-authored detail frequency f (None if the face omits it).
        label, s, f = None, 0.9, None
        while i < len(lines) and lines[i].startswith("#"):
            label = lines[i]
            m = _S_RE.search(label)
            if m:
                s = float(m.group(1))
            mf = _F_RE.search(label)
            if mf:
                f = int(mf.group(1))
            i += 1
        if i >= len(lines):
            break
        part = (label or "?").lstrip("#").split("s=")[0].strip() or "?"
        hdr = lines[i].split()
        try:
            nu, nv = int(hdr[0]), int(hdr[1]); i += 1
        except (ValueError, IndexError):
            warnings.append(f"expected an 'nu nv' header but got {lines[i]!r} (face skipped)")
            i += 1
            while i < len(lines) and not lines[i].startswith("#"):   # skip the rest of this face: one
                i += 1                                                 # warning, not one per point line
            continue
        pts = []
        while i < len(lines) and not lines[i].startswith("#"):
            p = lines[i].split()
            if len(p) == 3:
                try:
                    pts.append([float(p[0]), float(p[1]), float(p[2])])
                except ValueError:
                    pass
            i += 1
        actual, declared = len(pts), nu * nv
        if not np.isfinite(pts).all():
            warnings.append(f"face '{part}': nan/inf coordinate -> DROPPED. Fix this face's points.")
            continue
        if actual != declared:
            if actual >= 4 and nv > 0 and actual % nv == 0:
                warnings.append(f"face '{part}': header {nu}x{nv}={declared} but {actual} points -> "
                                f"repaired to {actual // nv}x{nv}. Make nu*nv equal your point count.")
                nu = actual // nv
            elif actual >= 4 and nu > 0 and actual % nu == 0:
                warnings.append(f"face '{part}': header {nu}x{nv}={declared} but {actual} points -> "
                                f"repaired to {nu}x{actual // nu}. Make nu*nv equal your point count.")
                nv = actual // nu
            else:
                warnings.append(f"face '{part}': header {nu}x{nv}={declared} but {actual} points "
                                f"(not divisible) -> DROPPED. Fix this face's point count.")
                continue
        if nu < 2 or nv < 2:
            warnings.append(f"face '{part}': degenerate grid {nu}x{nv} -> dropped.")
            continue
        faces.append(Patch(np.array(pts), nu, nv, s, label, f))

    if declared_n is not None and declared_n != len(faces):
        warnings.append(f"declared {declared_n} faces but parsed {len(faces)}; make the count line match.")
    return faces, warnings


# Cap on the upsampled grid size, in points per axis. geomdl's interpolate_surface is pure Python and its
# cost grows ~n^4 with the grid side: a 100x100 patch takes seconds, while an unbounded densification (e.g.
# 26x26 @ f=9 -> 234x234, ~55k points) is ~30x slower, minutes per patch. Raise it for very large / highly-detailed meshes.
_DEFAULT_MAX_RESAMPLE_DIM = 100


def fit_surfaces(faces, sigma=0.0, alpha=0.2, resample=False,
                 max_resample_dim=_DEFAULT_MAX_RESAMPLE_DIM):
    # interpolate a B-spline surface through each patch's on-surface grid.
    # if sigma > 0, apply edge-tapered, smoothness-scaled procedural variation to
    # each patch's grid points (p := p + d*sigma/s * N(0,I)); see add_noise.
    # if resample, first densify each patch (nu*f x nv*f via resolve_freq(p.f)) BEFORE
    # noising, bounding the upsampled grid to max_resample_dim points per axis. 
    if sigma > 0:
        from .add_noise import add_noise
    if resample:
        from .add_noise import resample as _resample, resolve_freq

    surfaces = []
    for p in faces:
        nu, nv = p.nu, p.nv
        pts = np.array(p.pts, dtype=float)
        if resample:
            # dedup BEFORE resample (1e-9), then densify
            seen = set()
            for i, key in enumerate(map(tuple, pts)):
                if key in seen:
                    pts[i] += 1e-9 * ((i % 50) + 1)
                else:
                    seen.add(key)
            # Apply the LLM-authored frequency f (resolve_freq -> [1,10]) but cap it so the
            # upsampled grid stays <= max_resample_dim points per axis: an unbounded densification
            # (e.g. 26x26 @ f=9 -> 234x234) makes geomdl's interpolate_surface very slow (see the cap above).
            dim_cap = max(1, max_resample_dim // max(nu, nv))
            freq = min(resolve_freq(p.f), dim_cap)
            pts, nu, nv = _resample(pts, nu, nv, frequency=freq)
        if sigma > 0:
            pts = add_noise(pts, nu, nv, s=p.s, sigma=sigma, alpha=alpha)
        pts = np.array(pts)
        seen = set()
        for i, key in enumerate(map(tuple, pts)):
            if key in seen:
                # Per-index offset, NOT a constant: a constant +1e-6 makes repeated copies of
                # a point all shift by the SAME amount and re-collide, recreating a zero-length
                # row/column that breaks geomdl's chord-length parametrization ("float division
                # by zero", e.g. nu=2 sliver patches). (i % 50) keeps the nudge bounded (<=5e-5,
                # sub-visible) on large patches while still giving consecutive points distinct
                # offsets.
                pts[i] += 1e-6 * ((i % 50) + 1)
            else:
                seen.add(key)
        deg = 2
        try:
            surf = fitting.interpolate_surface(
                pts.tolist(),
                size_u=nu,
                size_v=nv,
                degree_u=min(deg, nu - 1),
                degree_v=min(deg, nv - 1),
            )
            surfaces.append(surf)
        except Exception as e:
            print(f"Error fitting surface for face with nu={nu}, nv={nv}: {e}")
    return surfaces


# Surface variation: base noise scale and edge-taper clip. txt_to_obj.py applies it to every exported
# mesh; the generation loop's renders (model and critic) show it only with --render-variation.
VARIATION_SIGMA = 0.01
VARIATION_ALPHA = 0.2
VARIATION_SEED = 123     # fixed, so a render shows the same bumps as the default export
_RENDER_VARIATION = False


def set_render_variation(on):
    global _RENDER_VARIATION
    _RENDER_VARIATION = bool(on)


def fit_for_render(faces):
    # the surfaces the generation loop renders: the clean fit, or (--render-variation) the exported look
    if not _RENDER_VARIATION:
        return fit_surfaces(faces)
    np.random.seed(VARIATION_SEED)
    return fit_surfaces(faces, sigma=VARIATION_SIGMA, alpha=VARIATION_ALPHA, resample=True)
