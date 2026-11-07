# PyVista headless renderer: unit-normalizes the object and renders from fixed orthographic cameras.
# render_four_views() = 6-view gen montage; render_critic_views() = 4-view critic set.
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .view_geometry import VIEWS, CRITIC_VIEWS, view_dir as _dir, fit_scale as _fit_scale

# Cap total tessellation so a pathologically large model (thousands of patches) can't
# explode into tens of millions of triangles and OOM/segfault the native (VTK) renderer. Bound the per-patch 
# sample_size by its share of a global triangle budget. Normal-sized models keep the full sample_size; only huge ones scale down.
MAX_TOTAL_TRIS = 3_000_000


def _adaptive_sample_size(n_patches, sample_size):
    if n_patches <= 1:
        return sample_size
    cap = int((MAX_TOTAL_TRIS / (2 * n_patches)) ** 0.5)
    eff = max(2, min(sample_size, cap))
    if eff < sample_size:
        print(f"[render_pv] {n_patches} patches — capping sample_size {sample_size} -> {eff} "
              f"(~{2 * n_patches * eff * eff // 1000}k tris) to avoid OOM/segfault")
    return eff


def _mesh_from_surfaces(surfaces, sample_size):
    import pyvista as pv
    sample_size = _adaptive_sample_size(len(surfaces), sample_size)
    verts, tris, off = [], [], 0
    for s in surfaces:
        s.sample_size = sample_size
        s.tessellate()
        pts = np.array([v.data for v in s.vertices])
        f = np.array([t.vertex_ids for t in s.faces], dtype=np.int64)
        tris.append(f + off)
        verts.append(pts); off += len(pts)
    V = np.vstack(verts); F = np.vstack(tris)
    center = (V.max(0) + V.min(0)) / 2.0
    V = V - center
    V = V / max(float(np.linalg.norm(V, axis=1).max()), 1e-8)
    faces = np.hstack([np.full((len(F), 1), 3, dtype=np.int64), F.astype(np.int64)]).ravel()
    return pv.PolyData(V, faces)


def check_renderer():
    # one tiny off-screen render; exits with a clear message if PyVista/VTK cannot render on this machine
    try:
        import pyvista as pv
        p = pv.Plotter(off_screen=True, window_size=(64, 64))
        p.add_mesh(pv.Sphere())
        p.screenshot(return_img=True)
        p.close()
    except Exception as e:
        raise SystemExit(f"PyVista cannot render on this machine ({type(e).__name__}: {e}); "
                         "render_views and the critic need it — fix the PyVista/VTK install first.")


def render_four_views(surfaces, save_path=None, color=None, sample_size=60, dpi=150) -> bytes:
    return _render(surfaces, save_path, color, sample_size, dpi)


def render_critic_views(surfaces, save_path=None, color=None, sample_size=60, views=CRITIC_VIEWS, size=1536):
    # render each critic view as its OWN full-resolution image and return a list of
    # (label, png_bytes) sent to the critic as separate, individually-labelled images
    # (better per-view detail than tiling into one).
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
        mesh = _mesh_from_surfaces(surfaces, sample_size)
        scale = _fit_scale(mesh.points, views)
        out, raws = [], []
        for name, elev, azim, up in views:
            p = pv.Plotter(off_screen=True, window_size=(size, size))
            _add(p, mesh, color)
            p.camera_position = [tuple(_dir(elev, azim) * 3.0), (0.0, 0.0, 0.0), tuple(up)]
            p.camera.parallel_scale = scale
            img = p.screenshot(return_img=True)
            p.close()
            buf = io.BytesIO(); plt.imsave(buf, img, format="png")
            out.append((name, buf.getvalue())); raws.append((name, img))
        if save_path:
            fig = plt.figure(figsize=(6 * len(raws), 6))
            for i, (name, img) in enumerate(raws, 1):
                ax = fig.add_subplot(1, len(raws), i)
                ax.imshow(img); ax.set_title(name, fontsize=12); ax.axis("off")
            fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.03)
            fig.savefig(save_path, dpi=130); plt.close(fig)
        return out
    except Exception as e:
        print(f"[render_pv] critic-views failed ({type(e).__name__}: {e}); falling back to 6-view montage")
        return [("six standard views (front/back/sides, top, bottom)",
                 render_four_views(surfaces, save_path=save_path, color=color, sample_size=sample_size))]


def _add(p, mesh, color):
    p.set_background("white")
    p.add_mesh(mesh, color=color or "lightsteelblue", smooth_shading=True,
               specular=0.4, specular_power=15, ambient=0.25, diffuse=0.7)
    p.enable_parallel_projection()


def _render(surfaces, save_path, color, sample_size, dpi):
    import pyvista as pv
    pv.OFF_SCREEN = True
    mesh = _mesh_from_surfaces(surfaces, sample_size)
    scale = _fit_scale(mesh.points, VIEWS)

    imgs = []
    for name, elev, azim, up in VIEWS:
        p = pv.Plotter(off_screen=True, window_size=(512, 512))
        _add(p, mesh, color)
        p.camera_position = [tuple(_dir(elev, azim) * 3.0), (0.0, 0.0, 0.0), tuple(up)]
        p.camera.parallel_scale = scale
        imgs.append((name, p.screenshot(return_img=True)))
        p.close()

    fig = plt.figure(figsize=(12, 8))
    for i, (name, img) in enumerate(imgs, 1):
        ax = fig.add_subplot(2, 3, i)
        ax.imshow(img); ax.set_title(name, fontsize=11); ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.02, hspace=0.1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    data = buf.getvalue()
    if save_path:
        with open(save_path, "wb") as f:
            f.write(data)
    return data
