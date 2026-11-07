import os
import glob
import base64

# Core description for the render_views tool; chat.py appends how to invoke it (a .txt path or inline faces).
RENDER_VIEWS_CORE_DESC = (
    "Render your surface patches and SEE them. Returns ONE combined image — a montage of 6 views "
    "(four three-quarter angles from above showing front/back/both sides, plus a straight top-down "
    "and a straight bottom-up view) so you can check proportions, symmetry, that all parts are "
    "present and connect, and that parts look solid (not flat/degenerate)."
)


# Default description used for image-only shapes (no text prompt given). Used by chat_session
# so the generator AND the critic have a concrete target.
_IMAGE_ONLY_PROMPT = ("a 3D object that matches the reference image(s) — reproduce the object's "
                      "shape, parts, and proportions as accurately as possible")


def load_example(filepath):
    with open(filepath, 'r') as f:
        return f.read().strip()


_LIBRARY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "library")


def load_library(library_dir=_LIBRARY_DIR):
    files = sorted(glob.glob(f"{library_dir}/*.txt"))
    labels = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        name = name.lstrip('0123456789').lstrip('_')  # strip numeric prefix
        labels.append((name.replace("_", " "), f))
    return labels


def build_system_prompt(code_tool_name="generate_surface", has_images=False, strict_reply=False, has_render_tool=True):
    # code_tool_name: generate_surface by default; callers pass the real tool name (run_python / code_execution)
    return f"""You are an expert 3D surface designer. You describe 3D shapes as collections of surface patches (faces).
Each face is a collection of 3D points with nu rows and nv columns, listed row by row (u increases per row, v increases within each row).
You MUST use the {code_tool_name} tool to perform any arithmetic and logic. Dont perform arithmetic by yourself. You can call the tool multiple times.BE LESS VERBOSE FOR ALL YOUR ARITHMETIC AND CALULATIONS.
The representaiton looks as follows
<faces>
N
# object/part/.../subpart  s=0.XX f=N
nu nv
x y z
x y z
...
# object/part/.../subpart  s=0.XX f=N
nu nv
x y z
...
</faces>

Where N is the total number of faces, followed by each face's label line, its nu nv header, and its nu*nv points.

Each face is labelled with a comment line of the form:
    # object/part/.../patch  s=0.XX
- The label is a hierarchical semantic path, "/"-separated, from the whole object (top level)
  down to the individual patch (leaf). Use at least two levels (object/part), and use as many
  more levels as the object's structure needs, so the depth matches its complexity e.g. a
  simple lamp/shade, a chair/seat/seat_top, or a deeper car/door/handle/lever. Patches that
  belong to the same part must share that part's prefix so they group together.
- s is the per-face smoothness parameter in [0,1]. Smooth surfaces have high value; rough, irregular, organic surfaces have low value. Choose a value that matches how smooth that specific patch should look.
- f is the per-face surface fineness: an integer from 1 to 10 setting how FINE the patch's
  surface is i.e. how many small bumps cover it. It is INDEPENDENT of s. f=1 = a clean/polished
  patch ; f=10 = a densely bumpy, fine-grained surface  Pick f accordingly to match the patch's material.
  
Patch design guidelines:
- Use a right-handed coordinate system with +Z up, centered at the origin. Build each object in its natural canonical pose so the orientation it is normally seen or displayed in (eg. an animal standing on its feet, a mug upright, a car on its wheels). For objects with no natural up (eg. a ball, ring, or abstract shape), any orientation is fine.
- Use as many faces as needed to accurately capture the shape's geometry. Do not simplify or reduce detail and prefer more faces over fewer if it results in a more accurate and realistic representation.
- Give every face a hierarchical object/part/.../patch label (2 or more levels), a smoothness value s, and a surface-fineness value f, as shown above.
- Adjacent faces that share an edge must have exactly matching coordinates along that edge.
- Before giving your final answer, use the {code_tool_name} tool to verify your geometry: check that faces meant to connect share matching edge coordinates and that no part is left floating/disconnected; fix any failures and re-check.
{"- Also use the render_views tool to SEE your geometry: inspect the returned images for missing, flat, or disconnected parts and wrong proportions, and fix the coordinates." if has_render_tool else ""}
- Write your final geometry (the complete <faces> block) to a file named final_output.txt in the working directory, and overwrite that SAME file each time you revise it. The system reads your final result from final_output.txt, so {"NEVER put a <faces> block in your chat reply and write it ONLY to final_output.txt. Your reply MUST be a single one-line confirmation; including any <faces> in the reply is an error." if strict_reply else "do NOT repeat the full <faces> block in your chat reply and write it ONLY to final_output.txt and give just a one-line confirmation of what you changed."}- Round all coordinates to exactly 3 decimal places (e.g. 0.500 not 0.5, 1.000 not 1).
- Use more points for the complex surfaces or where fine detail is needed
- Aim for a believable, natural-looking result: realistic proportions, shape and detail, like a real example of what the description depicts UNLESS the description explicitly asks for a non-realistic style (cartoon, low-poly, blocky, geometric, abstract, minimalist, toy-like), in which case match that style instead.
{"- Reference images are provided: study all views to infer 3D geometry and proportions before computing coordinates." if has_images else ""}"""


# media types the Claude/OpenRouter image APIs accept. Anything else (e.g. BMP, TIFF) is
# transcoded to PNG before sending.
_SUPPORTED_IMG = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _sniff_media_type(b: bytes):
    # detect the REAL image format from magic bytes (file extensions lie — e.g. a .jpg that is
    # actually WebP, or a .png that is actually AVIF, both of which the API rejects).
    if b[:8] == b"\x89PNG\r\n\x1a\n":                 return "image/png"
    if b[:3] == b"\xff\xd8\xff":                      return "image/jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":       return "image/webp"
    if b[:6] in (b"GIF87a", b"GIF89a"):               return "image/gif"
    return None


def _read_image(image_path, supported=_SUPPORTED_IMG):
    # Return (media_type, raw_bytes) in `supported` (default: what the Claude/OpenRouter APIs accept).
    # Trusts the actual bytes, not the extension, and transcodes anything else (e.g. BMP, TIFF) to PNG
    # via Pillow.
    with open(image_path, "rb") as f:
        raw = f.read()
    mt = _sniff_media_type(raw)
    if mt not in supported:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        raw, mt = buf.getvalue(), "image/png"
    return mt, raw


def _image_block_openai(image_path):
    # OpenAI-compatible image block (used for OpenRouter / any OpenAI-format API)
    media_type, raw = _read_image(image_path)
    data = base64.standard_b64encode(raw).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}


def build_user_prompt(shape_description, images=None, examples=None):
    if examples is None:
        examples = load_library()

    example_blocks = ""
    for i, (label, filepath) in enumerate(examples, 1):
        content = load_example(filepath)
        # Examples that already contain a <faces> block (with or without
        # [run_python]/[output] worked steps) — include verbatim. Only bare
        # face lists get wrapped, to avoid double-wrapping <faces> tags.
        if "<faces>" in content:
            example_blocks += f"\nExample {i} - {label}:\n{content}\n"
        else:
            example_blocks += f"\nExample {i} - {label}:\n<faces>\n{content}\n</faces>\n"

    if not images:
        examples_text = (
            f"Here are {len(examples)} examples of how to describe 3D shapes "
            f"(each face is labelled with a hierarchical object/part/.../patch label, "
            f"a smoothness value s, and a surface-fineness value f):\n{example_blocks}"
        )
        task_text = (
            f"Now build the surface patches for: {shape_description}.\n\n"
            "Output format:\n"
            "#<thinking>\n"
            "#Plan the shape: what faces are needed, their placement, how they connect, "
            "and what 3D coordinates make sense.\n"
            "#</thinking>\n\n"
            "Write the complete <faces> block to final_output.txt; do NOT repeat it in "
            "your reply — a one-line confirmation is enough."
        )
        return [
            {"type": "text", "text": examples_text,
             "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": task_text},
        ]

    image_intro = (
        f"I am providing {len(images)} reference image(s) of the shape I want. "
        f"Use them along with the following description to generate surface patches in the same format:"
    )

    # Only append shape description if one was actually provided
    shape_part = f"\nAdditional description: {shape_description}" if shape_description else ""

    text = f"""Here are {len(examples)} examples of how to describe 3D shapes (each face is labelled with a hierarchical object/part/.../patch label, a smoothness value s, and a surface-fineness value f):
{example_blocks}
{image_intro}{shape_part}"""

    content = [_image_block_openai(img) for img in images]
    task_text = (
        "Now build the surface patches for the shape shown above.\n\n"
        "Output format:\n"
        "#<thinking>\n"
        "#Plan the shape: what faces are needed, their placement, how they connect, "
        "and what 3D coordinates make sense.\n"
        "#</thinking>\n\n"
        "Write the complete <faces> block to final_output.txt; do NOT repeat it in "
        "your reply — a one-line confirmation is enough."
    )
    content.append({"type": "text", "text": text,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}})
    content.append({"type": "text", "text": task_text})
    return content
