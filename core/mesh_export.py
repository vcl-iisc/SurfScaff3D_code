import os
import numpy as np


def surfaces_to_mesh(surfaces, delta: float = 0.05, max_triangles=None):
    # Convert geomdl Surface objects into a single trimesh.Trimesh.
    # delta controls tessellation density (0.05 -> 20x20 grid per surface).
    # max_triangles: if set, sample each surface more coarsely so the whole mesh stays within about that
    # many triangles (an n x n grid gives 2*(n-1)^2), for shapes with very many patches on low-memory machines.
    import trimesh

    all_vertices = []
    all_faces = []
    offset = 0
    cap = max(2, int((max_triangles / (2 * len(surfaces))) ** 0.5) + 1) if max_triangles and surfaces else None
    capped_from = None

    for surf in surfaces:
        surf.delta = delta
        if cap is not None and surf.sample_size_u > cap:
            capped_from = surf.sample_size_u
            surf.sample_size = cap
        surf.evaluate()
        pts = np.array(surf.evalpts, dtype=np.float32)
        n = int(round(len(pts) ** 0.5))

        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                a = i * n + j
                b = a + 1
                c = (i + 1) * n + j
                d = c + 1
                faces.extend([[a, b, c], [b, d, c]])

        all_vertices.append(pts)
        all_faces.append(np.array(faces, dtype=np.int64) + offset)
        offset += len(pts)

    if capped_from:
        print(f"[mesh_export] {len(surfaces)} patches: sampling {cap}x{cap} per patch instead of "
              f"{capped_from}x{capped_from} to stay under {max_triangles:,} triangles")
    if not all_vertices:
        return trimesh.Trimesh()

    return trimesh.Trimesh(
        vertices=np.vstack(all_vertices),
        faces=np.vstack(all_faces),
        process=False,
    )


def export_surfaces(surfaces, out_path: str, delta: float = 0.05, max_triangles=None) -> str:
    # Export geomdl Surface objects to a mesh file.
    # Format is inferred from the extension by trimesh (.obj/.stl/.ply/...).
    # Returns the output path.
    mesh = surfaces_to_mesh(surfaces, delta=delta, max_triangles=max_triangles)
    if len(mesh.vertices) == 0:
        raise ValueError("No surfaces to export (empty mesh)")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    mesh.export(out_path)
    return out_path
