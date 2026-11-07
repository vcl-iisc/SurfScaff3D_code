import argparse
import glob
import os
import sys

import numpy as np

from core.geometry import fit_surfaces, parse_faces, VARIATION_SEED
from core.geometry import VARIATION_SIGMA as _SIGMA, VARIATION_ALPHA as _ALPHA   # use --no-noise to disable
from core.mesh_export import export_surfaces


def _convert(txt_path, out_path, delta, sigma, resample, seed, png=False, max_triangles=None):
    with open(txt_path) as f:
        text = f.read()
    if "<faces>" not in text:   # bare patch list (e.g. a scratch final_output.txt), as every other script accepts
        text = "<faces>\n" + text.strip() + "\n</faces>"
    faces, warnings = parse_faces(text)
    if warnings:
        print(f"[parse repaired/flagged {len(warnings)} face(s)]:")
        for w in warnings:
            print(f"  - {w}")
    if resample and seed >= 0:
        np.random.seed(seed)   # reproducible noise
    surfaces = fit_surfaces(faces, sigma=sigma, alpha=_ALPHA, resample=resample)
    if not surfaces:
        return "No surfaces were fit (all patches failed) — nothing to export."
    if png:
        from rendering.render_pv import render_four_views
        data = render_four_views(surfaces, color="lightsteelblue")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)   # export_surfaces does this for meshes
        with open(out_path, "wb") as f:
            f.write(data)
    else:
        export_surfaces(surfaces, out_path, delta=delta, max_triangles=max_triangles)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("txt", help="input .txt file OR a directory; "
                    "directory mode finds all final_faces.txt recursively and converts each, "
                    "naming the output after its parent folder (e.g. <uid>.obj)")
    ap.add_argument("-o", "--out", help="output path (file mode) or flat output directory "
                    "(directory mode, named by parent folder); default: .obj/.png next to each input")
    ap.add_argument("--png", action="store_true",
                    help="render a 6-view PNG montage instead of exporting a mesh")
    ap.add_argument("--delta", type=float, default=0.0166667,
                    help="UV tessellation step; lower = finer mesh (default 0.0166667 -> 60x60/surface)")
    ap.add_argument("--no-noise", action="store_true",
                    help="Skip surface-variation noise and grid densification. "
                         "Produces a clean mesh from the B-spline interpolated through the authored points.")
    ap.add_argument("--seed", type=int, default=VARIATION_SEED,
                    help="RNG seed for reproducible surface-variation noise (set <0 to disable seeding).")
    ap.add_argument("--max-triangles", type=int, default=None,
                    help="cap the whole mesh at about this many triangles by sampling each patch more coarsely "
                         "(without it every patch is sampled 60x60). Use it if exporting a shape with very many "
                         "patches runs out of memory, e.g. --max-triangles 3000000")
    args = ap.parse_args()
    if args.max_triangles is not None and args.max_triangles < 1:
        ap.error("--max-triangles must be a positive number")

    sigma    = 0.0 if args.no_noise else _SIGMA
    resample = not args.no_noise
    ext      = ".png" if args.png else ".obj"

    # ── Directory mode: find all final_faces.txt recursively ─────────────────
    if os.path.isdir(args.txt):
        txts = sorted(glob.glob(os.path.join(args.txt, "**", "final_faces.txt"), recursive=True))
        if not txts:
            sys.exit(f"No final_faces.txt found recursively under {args.txt}/")
        out_dir = args.out
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ok = fail = 0
        for i, txt_path in enumerate(txts, 1):
            name = os.path.basename(os.path.dirname(txt_path))
            if out_dir:
                out_path = os.path.join(out_dir, name + ext)
            else:
                out_path = os.path.join(os.path.dirname(txt_path), name + ext)
            try:
                err = _convert(txt_path, out_path, args.delta, sigma, resample,
                               args.seed, png=args.png, max_triangles=args.max_triangles)
            except Exception as e:   # one bad file must not stop the rest of the folder
                err = f"{type(e).__name__}: {e}"
            if err:
                print(f"[{i}/{len(txts)}] FAIL {txt_path}: {err}")
                fail += 1
            else:
                print(f"[{i}/{len(txts)}] OK   {out_path}")
                ok += 1
        print(f"\n{ok} converted, {fail} failed.")
        if fail:
            sys.exit(1)
        return

    # ── Single file mode ──────────────────────────────────────────────────────
    out_path = args.out or os.path.splitext(args.txt)[0] + ext
    err = _convert(args.txt, out_path, args.delta, sigma, resample,
                   args.seed, png=args.png, max_triangles=args.max_triangles)
    if err:
        sys.exit(err)
    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()
