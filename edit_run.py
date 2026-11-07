import os, time
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
from openai import OpenAI
import chat
from core.geometry import set_render_variation
from chat import run_turn, _resolve_faces, _ensure_shape, _save_shape, save_views, set_sandbox_cwd
from core.critic import vlm_edit_critic
from core.prompt import build_system_prompt, _image_block_openai
from rendering.render_pv import render_critic_views, check_renderer

# User message appended when the edit critic still has issues to resolve.
_EDIT_FIX = (
    "The edit still has these issues:\n{feedback}\nMake only the "
    "changes needed to fix them — you may edit, add, or remove patches as "
    "the fix requires — but leave patches unrelated to the issues unchanged "
    "(same label, same coordinates), and overwrite the SAME final_output.txt. "
    "Do NOT repeat the full <faces> block in your reply — a one-line "
    "confirmation of what you changed is enough."
)


def _auto_refine_edit(client, messages, instruction, text, before_text, gen_model, critic_model,
                      shape="", max_iters=2, tau=0.9, save_prefix="chat_out/edit", ref_images=None):
    # Render the pre-edit geometry once for the critic's BEFORE views. Best-effort: if it fails to
    # build, the edit critic still judges AFTER against the instruction alone.
    before_imgs = None
    _bf, _bs, _bp, _bw, _berr = chat.realize(before_text or "")
    if not _berr and _bs:
        before_imgs = render_critic_views(_bs, save_path=f"{save_prefix}_before_critic.png")

    best_text, best_score, best_len, _, _, _ = chat._refine_loop(
        client, messages, text, gen_model,
        critic_fn=lambda imgs: vlm_edit_critic(instruction, before_imgs, imgs,
                                               shape=shape, model=critic_model, ref_images=ref_images),
        fix_msg_fn=lambda fb: _EDIT_FIX.format(feedback=fb),
        max_iters=max_iters, tau=tau, save_prefix=save_prefix, log_label="edit-refine",
    )
    # Revert to best-scoring iteration and persist to final_output.txt (mirrors auto_refine).
    del messages[best_len:]
    try:
        bt = best_text if "<faces>" in best_text else f"<faces>\n{best_text}\n</faces>"
        with open(os.path.join(chat._SANDBOX_CWD, chat.FINAL_OUTPUT_FILE), "w") as f:
            f.write(bt)
    except Exception:
        pass
    return best_text, best_score


def edit_run(run_dir, instruction, shape="", gen_model="moonshotai/kimi-k2.6",
             critic_iters=2, critic_model=None, images=None,    # critic_model None -> per-provider default (CRITIC_PROVIDER)
             render_variation=False):
    # Edit the shape staged in run_dir/scratch/final_output.txt (by edit_faces_file); all outputs go to run_dir.
    scratch = os.path.join(run_dir, "scratch")
    if not os.path.exists(os.path.join(scratch, "final_output.txt")):
        raise SystemExit(f"no final_output.txt in {scratch}")
    # snapshot the PRE-edit geometry now: the edit overwrites final_output.txt, and the critic needs it
    # for the BEFORE views (to judge that unrelated parts were preserved).
    before_text = open(os.path.join(scratch, "final_output.txt")).read().strip()
    set_sandbox_cwd(scratch)
    set_render_variation(render_variation)
    chat._LOG_PATH = os.path.join(run_dir, "api_log_edit.jsonl")
    chat._CRITIC_RAW_LOG = os.path.join(run_dir, "edit_log.txt")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    user_text = ((f"You are being asked to edit an object with the following description: {shape}\n\n") if shape else "") + (
        "The current 3D model is already saved in final_output.txt in the working directory. "
        "Edit it according to the following:\n" + instruction + "\nRead final_output.txt and make only the changes needed to apply this so "
        "you may edit, add, or remove patches as the edit requires but leave patches unrelated to "
        "the edit unchanged (same label, same coordinates), and overwrite the SAME final_output.txt. "
        "Do NOT repeat the full <faces> block in your reply and a one-line confirmation of what you changed is enough.")
    if images:
        user_content = [_image_block_openai(img) for img in images]
        user_content.append({"type": "text", "text": user_text})
    else:
        user_content = user_text
    messages = [{"role": "system", "content": build_system_prompt(code_tool_name="run_python", has_images=bool(images),
                                                                  has_render_tool=gen_model not in chat.NO_VISION_MODELS)},
                {"role": "user", "content": user_content}]

    _n0 = chat._logcount()
    text = run_turn(client, messages, gen_model)
    chat._append_agent_response("edit", _n0)
    text = _ensure_shape(client, messages, text, gen_model, "edit")
    open(os.path.join(run_dir, "edit_raw.txt"), "w").write(text)   # raw edit output (pre-critic), mirrors chat_gen_raw.txt
    if critic_iters > 0:
        text, score = _auto_refine_edit(client, messages, instruction, text, before_text,
                                        gen_model, critic_model, shape=shape, max_iters=critic_iters,
                                        save_prefix=os.path.join(run_dir, "edit"), ref_images=images)
        print(f"=== edit refined (best score {score:.2f}) ===")
    chat._write_critic_outputs()  # writes critic_outputs.txt from the edit-critic vlm_critic log entries
    chat._append_run_summary(os.path.basename(run_dir))
    out = os.path.join(run_dir, "chat_manualedit")
    err = save_views(text, out)
    print(("saved " if not err else "build error: ") + (err or out + ".png"))
    # persist the edited geometry as a clearly-named file, like chat.py's final_faces.txt
    if not _save_shape(os.path.join(run_dir, "final_faces.txt"), _resolve_faces(text)):
        raise SystemExit(f"the edit left no parseable <faces>: {run_dir}/final_faces.txt not written")


def edit_faces_file(faces_path, instruction, shape="", out_dir=None, **kw):
    # Outputs (edited geometry, logs, renders) all land under that working dir.
    faces_path = os.path.abspath(faces_path)
    if not os.path.exists(faces_path):
        raise SystemExit(f"no such faces file: {faces_path}")
    run_dir = os.path.abspath(out_dir) if out_dir else (
        os.path.join(os.path.dirname(faces_path), f"edit_{int(time.time())}")
    )
    scratch = os.path.join(run_dir, "scratch")
    os.makedirs(scratch, exist_ok=True)
    geom = open(faces_path).read().strip()
    if "<faces>" not in geom:
        geom = "<faces>\n" + geom + "\n</faces>"
    with open(os.path.join(scratch, "final_output.txt"), "w") as f:
        f.write(geom)
    print(f"=== editing {faces_path} -> working dir {run_dir} ===")
    return edit_run(run_dir, instruction, shape=shape, **kw)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Edit one shape.")
    ap.add_argument("pos", nargs="*", metavar="ARG", help='<faces_file> "edit instruction" ["shape"]')
    ap.add_argument("--gen-model", default="moonshotai/kimi-k2.6")
    ap.add_argument("--critic-model", default=None,
                    help="critic model id; default picks the per-provider default (see CRITIC_PROVIDER)")
    ap.add_argument("--images", default=None, help="comma-separated reference image paths")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--critic-iters", type=int, default=2, help="critic rounds (default 2; 0 = no critic)")
    ap.add_argument("--render-variation", action="store_true",
                    help="show the exported mesh's surface variation in the model's and critic's renders")
    args = ap.parse_intermixed_args()
    pos = args.pos
    if not 2 <= len(pos) <= 3:
        ap.error('need: <faces_file> "edit instruction" ["shape"] (quote the instruction and the shape)')
    # renders are needed by every critic round and by the render_views tool (models not in NO_VISION_MODELS)
    if args.critic_iters > 0 or args.gen_model not in chat.NO_VISION_MODELS:
        check_renderer()
    imgs = [x.strip() for x in args.images.split(",") if x.strip()] if args.images else None
    edit_faces_file(pos[0], pos[1], pos[2] if len(pos) > 2 else "", out_dir=args.out_dir,
                    critic_iters=args.critic_iters, critic_model=args.critic_model, images=imgs,
                    gen_model=args.gen_model, render_variation=args.render_variation)
