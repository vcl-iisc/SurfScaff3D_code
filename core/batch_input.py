# Prompts-file reader for run_baseline_pool.py (one shape per line, tab-separated).
import glob
import os


def _resolve_image_specs(spec: str, base_dir: str) -> list:
    # Expand a comma-separated image spec (the images column of a prompts-file line) into a list of
    # paths. Each token may be an absolute path, a path relative to the prompts file's directory, or a
    # glob (e.g. views/*.png). Unmatched tokens are kept verbatim so a missing file surfaces
    # loudly downstream rather than being silently dropped.
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        p = tok if os.path.isabs(tok) else os.path.join(base_dir, tok)
        matches = sorted(glob.glob(p))
        out.extend(matches if matches else [p])
    return out


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _looks_like_images(s: str) -> bool:
    # True when every comma-separated token has an image extension.
    toks = [t.strip() for t in s.split(",") if t.strip()]
    return bool(toks) and all(os.path.splitext(t)[1].lower() in _IMG_EXTS for t in toks)


def _load_prompts_file(path: str) -> list:
    # One shape per line, tab-separated, '#' = comment.
    #
    # Column layout (id is always optional — detected by extension heuristic):
    #
    # prompt                        bare text, no id, no images
    # img1.png,img2.png             image-only, no id, no prompt
    # prompt  TAB  images           text + images, no id
    # id      TAB  prompt           text-only with id
    # id      TAB  prompt  TAB  images   full form
    # id      TAB          TAB  images   image-only with id (empty prompt col)
    #
    # Image columns are comma-separated paths/globs resolved relative to this file.
    # For 2-column lines the second column is treated as images if it looks like image
    # paths (file extension heuristic), otherwise as id+prompt.
    base_dir = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")

            if len(parts) == 1:
                tok = parts[0].strip()
                if _looks_like_images(tok):
                    # image-only, no id, no prompt
                    out.append(("", "", _resolve_image_specs(tok, base_dir)))
                else:
                    # bare prompt, no id
                    out.append(("", tok, []))

            elif len(parts) == 2:
                if _looks_like_images(parts[1]):
                    # prompt + images, no id
                    out.append(("", parts[0].strip(), _resolve_image_specs(parts[1], base_dir)))
                else:
                    # id + prompt (no images)
                    out.append((parts[0].strip(), parts[1].strip(), []))

            else:  # 3+ columns — always id TAB prompt TAB images
                uid    = parts[0].strip()
                prompt = parts[1].strip()
                imgs   = (_resolve_image_specs(parts[2], base_dir)
                          if parts[2].strip() else [])
                out.append((uid, prompt, imgs))
    return out
