import base64
import contextlib
import io
import json
import openai
import os
import random
import re
import time


from core.geometry import parse_faces, fit_for_render, set_render_variation
import core.geometry as _geometry 
from rendering.render_pv import render_four_views, render_critic_views, check_renderer
from core.prompt import build_system_prompt, build_user_prompt, RENDER_VIEWS_CORE_DESC, _IMAGE_ONLY_PROMPT, _image_block_openai
from core.critic import vlm_critic
from core.chat_checkpoint import save_checkpoint, load_checkpoint

GENERATE_SURFACE_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": ("Execute Python locally to compute coordinates / verify geometry. "
                        "State does NOT persist between calls; print what you need back."),
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
}

_SANDBOX_CWD = "/tmp"


def set_sandbox_cwd(path):
    global _SANDBOX_CWD
    _SANDBOX_CWD = path


def _run_python_sandbox(code: str, timeout: int = 500) -> str:
    import errno, subprocess
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=timeout,
            cwd=_SANDBOX_CWD,
        )
    except subprocess.TimeoutExpired:
        # don't let a slow code-exec kill the whole shape — this string is returned to the model as the
        # tool result (run_turn appends it as role=tool), so it reacts and the loop continues.
        return (f"[stderr]\nYour code exceeded the {timeout}s execution limit and was killed. "
                "Make it faster (vectorize with numpy, avoid large Python loops / huge tessellation) "
                "and write the full <faces> to final_output.txt, then continue.")
    except OSError as e:
        # code over Linux's 128 KiB per-argument limit can't be passed via -c: return the error to the
        # model like any other instead of crashing the run (other start-up errors still raise)
        if e.errno != errno.E2BIG:
            raise
        return f"[stderr]\n{type(e).__name__}: {e}"
    output = result.stdout
    if result.stderr:
        output += "\n[stderr]\n" + result.stderr
    return output or "(no output)"


RENDER_VIEWS_TOOL = {
    "type": "function",
    "function": {
        "name": "render_views",
        "description": (RENDER_VIEWS_CORE_DESC +
                        " Preferred: pass `path` = the .txt file you saved in the working directory "
                        "containing the faces (it may hold the full <faces>...</faces> block or just "
                        "the patch content) so you do not have to re-paste a large block. Or inline "
                        "the block via `faces`. Use this BEFORE giving your final answer to catch "
                        "problems you cannot tell from coordinates alone; if a render looks wrong, "
                        "fix the coordinates and render again."),
        "parameters": {"type": "object",
                       "properties": {
                           "path": {"type": "string",
                                    "description": "Path to a .txt file (in the working directory) with the faces to render."},
                           "faces": {"type": "string",
                                     "description": "Alternatively, the complete <faces>...</faces> block inline."},
                       }},
    },
}

# Text-only models: they get only run_python (render_views would hand them an image_url
# content block they cannot consume: OpenRouter returns 404 "No endpoints found that support image input",
# which the retry policy below raises without retrying), and
# their system prompt does not mention render_views.
NO_VISION_MODELS = {"deepseek/deepseek-v4-flash-0731"}


def _client():
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1",
                  api_key=os.environ["OPENROUTER_API_KEY"])


_LOG_PATH = "chat_out/api_log.jsonl"
_CRITIC_RAW_LOG = "chat_out/critic_raw_log.txt"


def _append_critic_raw(save_prefix, it, res):
    try:
        run = os.path.basename(os.path.dirname(save_prefix) or save_prefix)
        raw = (res.get("raw") or "").strip()
        body = raw or "(empty response)"
        try:
            body = json.dumps(json.loads(raw[raw.index("{"):raw.rindex("}") + 1]), indent=2)
        except Exception:
            pass
        os.makedirs(os.path.dirname(_CRITIC_RAW_LOG) or ".", exist_ok=True)
        with open(_CRITIC_RAW_LOG, "a") as f:
            f.write(f"\n{'='*72}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {run} | iter {it} "
                    f"| score={res.get('score')}\n{'='*72}\n")
            f.write(body + "\n")
    except Exception:
        pass


def _logcount():
    try:
        with open(_LOG_PATH) as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _append_agent_response(label, start_line):
    try:
        with open(_LOG_PATH) as f:
            lines = f.readlines()[start_line:]
        reason_tok = 0; thinking = []; codes = []; nrender = 0; reply = ""
        for l in lines:
            try:
                e = json.loads(l)
            except Exception:
                continue
            if e.get("tag") != "run_turn":
                continue
            ch = ((e.get("response", {}) or {}).get("choices") or [{}])[0]
            m = ch.get("message", {}) or {}
            u = (e.get("response", {}) or {}).get("usage", {}) or {}
            reason_tok += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            if m.get("reasoning"):
                thinking.append(str(m["reasoning"]).strip())
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                if fn.get("name") == "run_python":
                    try:
                        codes.append(json.loads(fn.get("arguments", "")).get("code", ""))
                    except Exception:
                        pass
                elif fn.get("name") == "render_views":
                    nrender += 1
            if ch.get("finish_reason") == "stop":
                reply = m.get("content") or ""
        with open(_CRITIC_RAW_LOG, "a") as f:
            f.write(f"\n------ agent {label}: reasoning_tokens={reason_tok} | "
                    f"run_python x{len(codes)} | render_views x{nrender} ------\n")
            if thinking:
                f.write("thinking:\n" + "\n".join(thinking) + "\n")
            for c in codes:
                f.write("\ncode:\n" + c.strip() + "\n")
            if reply.strip():
                f.write("\nreply: " + reply.strip() + "\n")
    except Exception:
        pass


def _append_run_summary(run_label):
    try:
        rows = [json.loads(l) for l in open(_LOG_PATH) if l.strip()]
        cost = tin = tout = cached = crit = empt = 0
        provs = {}
        for r in rows:
            u = (r.get("response", {}) or {}).get("usage", {}) or {}
            cost += (u.get("cost_details") or {}).get("upstream_inference_cost") or u.get("cost") or 0
            tin += u.get("prompt_tokens") or 0
            tout += u.get("completion_tokens") or 0
            cached += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            p = (r.get("response", {}) or {}).get("provider", "?")
            provs[p] = provs.get(p, 0) + 1
            if r.get("tag") == "vlm_critic":
                crit += 1
            ch = ((r.get("response", {}) or {}).get("choices") or [{}])[0]
            m = ch.get("message", {}) or {}
            if ch.get("finish_reason") is None and not (m.get("content") or "") and not (m.get("tool_calls") or []):
                empt += 1
        pct = 100 * cached / max(tin, 1)
        summary = (f"{run_label}: {len(rows)} calls | {crit} critic iters | empties:{empt}\n"
                   f"providers: {provs}\n"
                   f"input tokens: {tin:,} | cached: {cached:,} ({pct:.0f}% cache hit) | output: {tout:,}\n"
                   f"TOTAL COST: ${cost:.3f}")
        with open(_CRITIC_RAW_LOG, "a") as f:
            f.write(f"\n{'#'*72}\n{summary}\n{'#'*72}\n")
        print("\n" + summary)
    except Exception:
        pass


def _strip_images(request):
    # replace base64 image data URIs with a short placeholder so api_log.jsonl stays
    # readable (images would otherwise re-send every turn)
    msgs = request.get("messages")
    if not isinstance(msgs, list):
        return request
    out_msgs = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            c = [{"type": "image_url", "image_url": {"url": "<base64 image omitted>"}}
                 if isinstance(p, dict) and p.get("type") == "image_url" else p for p in c]
            m = {**m, "content": c}
        out_msgs.append(m)
    return {**request, "messages": out_msgs}


def _log_api(tag, request, response, path=None):
    path = path or _LOG_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"tag": tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "request": _strip_images(request), "response": response},
                           default=str) + "\n")


def _write_critic_outputs(jsonl_path=None):
    jsonl_path = jsonl_path or _LOG_PATH
    if not os.path.exists(jsonl_path):
        return None
    crit = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                if e.get("tag") == "vlm_critic":
                    crit.append(e)
    if not crit:
        return None
    out_path = os.path.join(os.path.dirname(jsonl_path) or ".", "critic_outputs.txt")
    with open(out_path, "w") as f:
        for i, e in enumerate(crit, 1):
            try:
                raw = e["response"]["choices"][0]["message"]["content"] or ""
            except Exception:
                raw = ""
            f.write(f"{'='*70}\n# critic - iteration {i}\n{'='*70}\n{raw}\n\n")
    return out_path


def _render_views_tool(args, rendered_images):
    # handle a render_views tool call: read faces from a .txt path (in the sandbox working
    # dir) or inline, render via render_four_views, and queue the image for a follow-up user
    # message; returns the text tool-result
    src = (args.get("path") or "").strip()
    try:
        if src:
            fp = src if os.path.isabs(src) else os.path.join(_SANDBOX_CWD, src)
            with open(fp) as fh:
                faces_text = fh.read()
        else:
            faces_text = args.get("faces", "") or ""
        if "<faces>" not in faces_text:
            faces_text = "<faces>\n" + faces_text.strip() + "\n</faces>"
        faces, _w = parse_faces(faces_text)
        if not faces:
            return "[error] no parseable faces to render"
        surfaces = fit_for_render(faces)
        png = render_four_views(surfaces, color="lightsteelblue")
        b64 = base64.standard_b64encode(png).decode("utf-8")
        rendered_images.append((f"data:image/png;base64,{b64}",
                                f"Render of {len(surfaces)} patches (4 three-quarter-top views + top + bottom)."))
        return (f"Rendered {len(surfaces)} patches" + (f" from {src}" if src else "") +
                ". Image attached in the next message.")
    except FileNotFoundError:
        return f"[error] file not found: {src}"
    except Exception as e:
        return f"[error] could not render: {type(e).__name__}: {e}"


# OpenRouter provider pins: unpinned, OpenRouter load-balances across hosts whose quantisation and
# sampling differ, so repeat runs are not comparable. allow_fallbacks is off for every pin.
_PINNED_PROVIDERS = {
    "deepseek/deepseek-v4-flash-0731": ["StreamLake", "DeepInfra"],  # StreamLake 429s under concurrency; same 1M ctx
}


def _routing(model):
    # extra_body for this model's requests: provider pin + reasoning/caching settings
    if model.startswith("moonshotai/"):
        return {"provider": {"order": ["Moonshot AI"], "allow_fallbacks": False},   # keeps reasoning + caching
                "reasoning_effort": "medium"}                                       # best-effort: kimi may ignore it
    if model.startswith("anthropic/"):
        # extended thinking is OFF on OpenRouter unless requested (reasoning.max_tokens = Anthropic thinking budget)
        # pin the Anthropic host so every turn reuses the same host's prompt cache 
        return {"reasoning": {"max_tokens": 8000},
                "provider": {"order": ["Anthropic"], "allow_fallbacks": False},
                "cache_control": {"type": "ephemeral"}}
    if model in _PINNED_PROVIDERS:
        return {"provider": {"order": _PINNED_PROVIDERS[model], "allow_fallbacks": False}}
    return {}


def run_turn(client, messages, model, max_tool_calls=20):
    text = ""
    for _step in range(1, max_tool_calls + 1):
        extra = _routing(model)
        tools = ([GENERATE_SURFACE_TOOL] if model in NO_VISION_MODELS
                  else [GENERATE_SURFACE_TOOL, RENDER_VIEWS_TOOL])
        req = {"model": model, "messages": messages,
               "tools": tools,
               "tool_choice": "auto",
               **({"max_tokens": 64000} if model.startswith("anthropic/") else {"temperature": 0.0})}
        if extra:
            req["extra_body"] = extra
        # Single retry layer (SDK retries off for THIS call so they don't stack) — retries transient
        # failures, incl. malformed JSON bodies the SDK surfaces as JSONDecodeError without retrying.
        # Other errors (400, 401, 403, 404, ...) are NOT retried — they raise.
        c = client.with_options(max_retries=0)
        for attempt in range(4):
            try:
                resp = c.chat.completions.create(**req)
                break
            except (json.JSONDecodeError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError, openai.RateLimitError) as e:
                if attempt == 3:
                    _log_api("run_turn:error", req, {"error": f"{type(e).__name__}: {e}"})
                    print(f"[run_turn] API ERROR after 4 attempts: {type(e).__name__}: {e}")
                    raise
                delay = min(2 ** attempt, 20) + random.uniform(0, 1)
                print(f"[run_turn] transient API error (attempt {attempt + 1}/4): "
                      f"{type(e).__name__}; retrying in {delay:.1f}s")
                time.sleep(delay)
            except Exception as e:
                _log_api("run_turn:error", req, {"error": f"{type(e).__name__}: {e}"})
                total = sum(len(str(m.get("content") or "")) for m in messages)
                print(f"[run_turn] API ERROR: {type(e).__name__}: {e}")
                print(f"  diagnostic: {len(messages)} messages, ~{total} chars (~{total // 4} tokens), "
                      f"{sum(1 for m in messages if m.get('role') == 'tool')} tool-msgs, "
                      f"roles={[m.get('role') for m in messages]}")
                raise
        _log_api("run_turn", req, resp.model_dump())
        msg = resp.choices[0].message
        text = msg.content or ""
        am = {"role": "assistant", "content": text}
        # Moonshot rule: keep the turn's reasoning_details and send it back WITHIN the tool-call loop.
        # Without it a reasoning-only turn collapses to an empty assistant message that Moonshot 400s
        # ("must not be empty"); keeping it also stops kimi re-deriving its reasoning each step.
        rd = msg.model_dump().get("reasoning_details")
        if rd:
            am["reasoning_details"] = rd
        if msg.tool_calls:
            am["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(am)
        if not msg.tool_calls:
            # Turn TRUNCATED at the output limit (reasoning runaway): don't stop. Append a user nudge
            # so history ends on a USER turn (an assistant(empty+reasoning_details)->user pattern
            # Moonshot accepts) and the model continues from its preserved reasoning_details.
            # Bounded by max_tool_calls.
            if resp.choices[0].finish_reason == "length":
                print(f"  [run_turn] step {_step}: finish_reason=length (truncated) — nudging to finish")
                messages.append({"role": "user", "content": "Continue and finish your answer."})
                continue
            return text
        rendered_images = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if tc.function.name == "render_views":
                out = _render_views_tool(args, rendered_images)
                print(f"  [render_views] step {_step}: {out[:90].strip()!r}")
            else:
                code = args.get("code", "")
                out = _run_python_sandbox(code) if code else "[error] no code argument"
                print(f"  [run_python] step {_step}: {len(code)} chars -> {out[:80].strip()!r}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        if rendered_images:
            content = [{"type": "text", "text": "Here are the render(s) you requested:"}]
            for data_uri, caption in rendered_images:
                content.append({"type": "text", "text": caption})
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
            messages.append({"role": "user", "content": content})
    print(f"  [run_turn] WARNING: hit max_tool_calls={max_tool_calls} without a final answer")
    return text


FINAL_OUTPUT_FILE = "final_output.txt"


def _resolve_faces(text):
    # a <faces> block typed in the reply is used only if it holds patches; otherwise read final_output.txt
    if text and "<faces>" in text and "</faces>" in text and _shape_error(text) is None:
        return text
    try:
        fp = os.path.join(_SANDBOX_CWD, FINAL_OUTPUT_FILE)
        if os.path.exists(fp):
            content = open(fp).read().strip()
            if content:
                return content if "<faces>" in content else f"<faces>\n{content}\n</faces>"
    except Exception:
        pass
    return text or ""


def _shape_error(text):
    # None if text holds at least one parseable patch, else the reason
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            faces, _ = parse_faces(text)
    except Exception as e:
        return str(e)
    return None if faces else "no patches found"


def _save_shape(path, text):
    # write text to path only if it holds at least one parseable patch; returns True if written
    ok = _shape_error(text) is None
    if ok:
        with open(path, "w") as fh:
            fh.write(text)
    return ok


_BUILD_RETRIES = 2   # extra turns when a turn ends without a parseable shape (no critic needed)


def _ensure_shape(client, messages, text, gen_model, label):
    # if the model's turn left no parseable shape, tell it so and let it write final_output.txt again
    for k in range(1, _BUILD_RETRIES + 1):
        err = _shape_error(_resolve_faces(text))
        if err is None:
            break
        print(f"  [build check] {err} — asking the model to fix it ({k}/{_BUILD_RETRIES})")
        messages.append({"role": "user", "content": f"The geometry could not be built (parse/fit failed): {err}. "
                                                    "Write the complete <faces> block to final_output.txt."})
        _n0 = _logcount()
        text = run_turn(client, messages, gen_model)
        _append_agent_response(f"{label} build fix {k}", _n0)
    return text


def realize(text, save_path=None):
    text = _resolve_faces(text)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            faces, warnings = parse_faces(text)
            surfaces = fit_for_render(faces)
        png = render_four_views(surfaces, save_path=save_path, color="lightsteelblue")
        return faces, surfaces, png, warnings, None
    except Exception as e:
        return None, None, None, [], str(e)


def save_views(text, stem):
    text = _resolve_faces(text)
    d = os.path.dirname(stem)
    if d:
        os.makedirs(d, exist_ok=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            faces, warnings = parse_faces(text)
            surfaces = fit_for_render(faces)
        render_four_views(surfaces, save_path=f"{stem}.png", color="lightsteelblue")
        if warnings:
            print(f"  [parse repaired {len(warnings)} face(s)]")
        return None
    except Exception as e:
        return str(e)


def _format_feedback(vlm_issues):
    return "\n".join(f"- {i.get('part','?')}: {i.get('problem','')} -> {i.get('fix','')}"
                     for i in vlm_issues)


# Critic fix directive for chat.py's generation refine loop. Used to continue the loop and, on a
# budget-stopped iteration, to leave the pending fix in the conversation for --resume.
# {feedback} is filled with the formatted critic issues.
_FIX_INSTRUCTION = (
    "The current model has these issues. Make only the changes needed "
    "to fix them — you may edit, add, or remove patches as the fix "
    "requires — but leave patches unrelated to the issues unchanged "
    "(same label, same coordinates) unless the issue is global. Overwrite the SAME "
    "final_output.txt file with the corrected full model (do not "
    "create a new file); you need not paste the block here:\n{feedback}"
)


def _refine_loop(client, messages, text, gen_model,
                 critic_fn, fix_msg_fn,
                 max_iters, tau, save_prefix, log_label,
                 start_iter=1, init_best=None, on_iter_start=None):
    # Shared iterative critic-refine loop used by both generation and editing.
    #
    # critic_fn(critic_imgs)  -> res dict {score, issues, raw, unavailable, request, response}
    # fix_msg_fn(feedback)    -> str   the user message content appended when issues remain
    # on_iter_start(it, messages, text, best_text, best_score, best_len) -> None  (optional,
    #     called at the top of each iteration — used by auto_refine for per-iter checkpoints)
    #
    # Returns (best_text, best_score, best_len, last_it, pending, text).
    # pending=True means a fix message was appended but the loop ended before running it
    # (budget stop for --resume); last_it is the final value of the loop variable.
    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    if init_best is not None:
        best_text, best_score, best_len = init_best
    else:
        best_text, best_score, best_len = _resolve_faces(text), -1.0, len(messages)
    it = start_iter - 1   # sentinel: valid post-loop checkpoint even when loop body never runs
    pending = False
    for it in range(start_iter, max_iters + 1):
        if on_iter_start:
            on_iter_start(it, messages, text, best_text, best_score, best_len)
        faces_text = _resolve_faces(text)
        with open(f"{save_prefix}_iter{it}_faces.txt", "w") as fh:
            fh.write(faces_text or "")
        faces, surfaces, png, warnings, err = realize(faces_text, save_path=f"{save_prefix}_iter{it}.png")
        critic_unavailable = False
        if err:
            score, issues = 0.0, []
            feedback, accept = f"The geometry could not be built (parse/fit failed): {err}", False
        else:
            critic_imgs = render_critic_views(surfaces, save_path=f"{save_prefix}_iter{it}_critic.png")
            try:
                res = critic_fn(critic_imgs)
                _log_api("vlm_critic", res["request"], res["response"])
                _append_critic_raw(save_prefix, it, res)
            except openai.APIStatusError as e:   # refused (bad key, no access, unknown model, bad request): retrying won't help
                hint = (f"continue with: python chat.py --resume {os.path.dirname(save_prefix) or '.'}"
                        if log_label == "refine" else "run the edit again")
                raise SystemExit(f"critic request refused ({type(e).__name__}: {e}) — fix the critic key/model, then {hint}")
            except Exception as e:
                print(f"  [critic] call failed: {type(e).__name__}: {e} — accepting current geometry")
                res = {"unavailable": True, "score": None, "issues": []}
            issues, score = res["issues"], res["score"]
            if res.get("unavailable"):
                # critic empty even after retries — accept current geometry, send no feedback
                score = best_score if best_score >= 0 else 0.0
                issues, feedback, accept = [], "", True
                critic_unavailable = True   # synthetic score == best: must NOT win the >= checkpoint
            else:
                feedback, accept = _format_feedback(issues), (not issues)
        if warnings:
            feedback = (feedback + "\n\nFORMAT WARNINGS — fix these in your next output:\n- "
                        + "\n- ".join(warnings)).strip()
            accept = False
        print(f"  [{log_label} {it}/{max_iters}] score={score:.2f}  issues={len(issues)}  warnings={len(warnings)}")
        # >= so a later iteration that TIES the best wins — prefer the most recent geometry on a tie.
        # Skip when the critic was unavailable: synthetic score == best, so >= would steal the checkpoint.
        if not critic_unavailable and score >= best_score:
            best_text, best_score, best_len = faces_text, score, len(messages)
        improve = not accept and score < tau
        if improve:
            messages.append({"role": "user", "content": fix_msg_fn(feedback)})
        if accept or score >= tau or it == max_iters:
            pending = improve   # True only when a fix was appended but won't be run (budget stop)
            break
        _n0 = _logcount()
        text = run_turn(client, messages, gen_model)
        _append_agent_response(f"{log_label} iter {it}", _n0)
    return best_text, best_score, best_len, it, pending, text


def auto_refine(client, messages, prompt, text, gen_model, critic_model,
                max_iters=5, tau=0.9, save_prefix="chat_out/chat",
                start_iter=1, init_best=None, ref_images=None):
    # start_iter / init_best support --resume: continue refining from a saved checkpoint instead
    # of from iteration 1 with fresh best-state.
    run_dir = os.path.dirname(save_prefix) or "."

    def _on_iter_start(it, msgs, txt, best_text, best_score, best_len):
        try:
            save_checkpoint(run_dir, msgs, text=txt, best_text=best_text, best_score=best_score,
                            best_len=best_len, next_iter=it, max_iters=max_iters,
                            prompt=prompt, gen_model=gen_model, critic_model=critic_model, tau=tau,
                            images=ref_images, faces=_resolve_faces(txt),
                            render_variation=_geometry._RENDER_VARIATION)
        except Exception as e:
            print(f"  [checkpoint] skipped (non-fatal): {type(e).__name__}: {e}")

    best_text, best_score, best_len, last_it, pending, text = _refine_loop(
        client, messages, text, gen_model,
        critic_fn=lambda imgs: vlm_critic(prompt, imgs, model=critic_model, ref_images=ref_images),
        fix_msg_fn=lambda fb: _FIX_INSTRUCTION.format(feedback=fb),
        max_iters=max_iters, tau=tau, save_prefix=save_prefix, log_label="refine",
        start_iter=start_iter, init_best=init_best, on_iter_start=_on_iter_start,
    )
    # Final checkpoint with the TRUE completed count (next_iter = last done + 1), written BEFORE the
    # revert/truncation so --resume continues from the full conversation at the correct iteration
    # number (clean iter1..N progression, no overwrite).
    try:
        save_checkpoint(run_dir, messages, text=text, best_text=best_text, best_score=best_score,
                        best_len=best_len, next_iter=last_it + 1, max_iters=max_iters, pending=pending,
                        prompt=prompt, gen_model=gen_model, critic_model=critic_model, tau=tau,
                        images=ref_images, faces=_resolve_faces(text),
                        render_variation=_geometry._RENDER_VARIATION)
    except Exception as e:
        print(f"  [checkpoint] final skipped (non-fatal): {type(e).__name__}: {e}")
    # Revert to the BEST-scoring iteration and persist best to final_output.txt so the
    # conversation, the file, chat_refined.png, and any later in-session edit are all consistent.
    del messages[best_len:]
    try:
        bt = best_text if "<faces>" in best_text else f"<faces>\n{best_text}\n</faces>"
        with open(os.path.join(_SANDBOX_CWD, FINAL_OUTPUT_FILE), "w") as f:
            f.write(bt)
    except Exception:
        pass
    return best_text, best_score


def chat_session(prompt, gen_model="moonshotai/kimi-k2.6",
                 critic_model=None,   # None -> per-provider default (see CRITIC_PROVIDER)
                 refine_iters=5, uid=None, images=None, render_variation=False):
    from dotenv import load_dotenv
    load_dotenv()
    client = _client()

    if images and not prompt:
        prompt = _IMAGE_ONLY_PROMPT
    if uid:
        run_dir = os.path.join("chat_out", gen_model.split("/")[-1], uid)
    else:
        slug = re.sub(r"[^a-z0-9]+", "_", (prompt or "").lower()).strip("_")[:40] or "shape"
        run_dir = os.path.join("chat_out", gen_model.split("/")[-1], f"{slug}_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)
    scratch = os.path.join(run_dir, "scratch")
    os.makedirs(scratch, exist_ok=True)
    set_sandbox_cwd(scratch)
    set_render_variation(render_variation)
    global _LOG_PATH, _CRITIC_RAW_LOG
    _LOG_PATH = os.path.join(run_dir, "api_log.jsonl")
    _CRITIC_RAW_LOG = os.path.join(run_dir, "critic_raw_log.txt")
    print(f"=== Output Directory: {run_dir} ===")

    messages = [
        {"role": "system", "content": build_system_prompt(code_tool_name="run_python", strict_reply=True, has_images=bool(images),
                                                            has_render_tool=gen_model not in NO_VISION_MODELS)},
        {"role": "user", "content": build_user_prompt(prompt, images=images)},
    ]
    print(f"=== Generating: {prompt!r} ===")
    _n0 = _logcount()
    text = run_turn(client, messages, gen_model)
    _append_agent_response("generation", _n0)
    text = _ensure_shape(client, messages, text, gen_model, "generation")
    open(os.path.join(run_dir, "chat_gen_raw.txt"), "w").write(text)
    err = save_views(text, os.path.join(run_dir, "chat_gen"))
    print(f"saved {run_dir}/chat_gen.png" if not err
          else f"build error: {err}  (raw output in {run_dir}/chat_gen_raw.txt)")

    # refine_iters=0: no critic call, but checkpoint.json is still written so --resume can add rounds later
    text, score = auto_refine(client, messages, prompt, text, gen_model, critic_model,
                              max_iters=refine_iters, save_prefix=os.path.join(run_dir, "chat"),
                              ref_images=images)
    if refine_iters > 0:
        print(f"=== refined (best score {score:.2f}) ===")
    err = save_views(text, os.path.join(run_dir, "chat_refined"))
    print(f"saved {run_dir}/chat_refined.png" if not err else f"build error: {err}")
    # persist the EXACT geometry chat_refined.png was rendered from (the best checkpoint when
    # refining) as final_faces.txt; in-session edits overwrite it and keep this one as before_edits_faces.txt
    refined = _resolve_faces(text)
    has_shape = _save_shape(os.path.join(run_dir, "final_faces.txt"), refined)
    if not has_shape:
        print(f"no shape produced (no parseable <faces>): {run_dir}/final_faces.txt not written")

    print("\nEnter edit instructions ('quit' to stop):")
    edit_n = 0
    while True:
        try:
            instr = input("edit> ").strip()
        except EOFError:
            break
        if not instr or instr.lower() in ("quit", "exit"):
            break
        else:
            edit_imgs = []
            if "::img" in instr:
                instr, _, spec = instr.partition("::img")
                instr = instr.strip()
                edit_imgs = []
                for _p in spec.split(","):
                    _p = _p.strip()
                    if not _p:
                        continue
                    try:
                        _image_block_openai(_p)      # readable AND an image the API accepts, else skip it
                        edit_imgs.append(_p)
                    except Exception as _e:
                        print(f"  [warn] skipped {_p!r}: {type(_e).__name__}: {_e}")
            if not instr and not edit_imgs:
                print("  [skip] empty instruction")
                continue
            img_note = ("Use the reference image(s) for the object's SHAPE, parts, and proportions ONLY — "
                        "ignore their color, material, texture, lighting, and background.\n") if edit_imgs and not instr else ""
            edit_text = f"{img_note}{instr}\nUpdate final_output.txt with the full corrected <faces> block — do not paste it in your reply."
            if edit_imgs:
                content = [_image_block_openai(img) for img in edit_imgs]
                content.append({"type": "text", "text": edit_text})
            else:
                content = edit_text
            geom_path = os.path.join(scratch, FINAL_OUTPUT_FILE)
            geom_before = open(geom_path).read() if os.path.exists(geom_path) else None
            n_before = len(messages)
            messages.append({"role": "user", "content": content})
            _n0 = _logcount()
            try:
                text = run_turn(client, messages, gen_model)
            except Exception as e:
                print(f"  [error] edit turn failed: {type(e).__name__}: {e}")
                del messages[n_before:]            # drop the whole failed turn so the next instruction is clean
                if geom_before is not None:        # and undo any partial edit it wrote
                    with open(geom_path, "w") as f:
                        f.write(geom_before)
                continue
            _append_agent_response(f"edit {edit_n + 1}", _n0)
        edit_n += 1
        err = save_views(text, os.path.join(run_dir, f"chat_edit{edit_n}"))
        print(f"saved {run_dir}/chat_edit{edit_n}.png" if not err else f"build error: {err}")

    if edit_n > 0:
        _save_shape(os.path.join(run_dir, "before_edits_faces.txt"), refined)
        if _save_shape(os.path.join(run_dir, "final_faces.txt"), _resolve_faces(text)):
            has_shape = True
        else:
            print(f"the last edit left no parseable <faces>: {run_dir}/final_faces.txt not updated")
    crit = _write_critic_outputs()
    if crit:
        print(f"=== critic outputs: {crit} ===")
    _append_run_summary(os.path.basename(run_dir))
    if not has_shape:   # non-zero exit, so the pools count it as FAIL and --retry-failed picks it up
        raise SystemExit(f"no shape produced in {run_dir}")
    return text


def resume_session(run_dir, add_iters=5, render_variation=None):
    # Continue a previous run's critic refinement from its checkpoint.json for `add_iters` more
    # iterations. Restores the exact API state (messages incl. reasoning_details / tool_calls / base64
    # renders) plus the sandbox (re-points _LOG_PATH/_CRITIC_RAW_LOG at run_dir and the run_python
    # CWD at run_dir/scratch).
    from dotenv import load_dotenv
    load_dotenv()
    if not os.path.exists(os.path.join(run_dir, "checkpoint.json")):
        raise FileNotFoundError(f"no checkpoint.json in {run_dir} (it is written once generation finishes)")
    ck = load_checkpoint(run_dir)

    client = _client()
    global _LOG_PATH, _CRITIC_RAW_LOG
    _LOG_PATH = os.path.join(run_dir, "api_log.jsonl")
    _CRITIC_RAW_LOG = os.path.join(run_dir, "critic_raw_log.txt")
    set_sandbox_cwd(os.path.join(run_dir, "scratch"))   # so run_python + _resolve_faces find final_output.txt
    set_render_variation(ck.get("render_variation", False) if render_variation is None else render_variation)
    if "<faces>" in (ck.get("faces") or ""):
        # a finished run reverted final_output.txt to its BEST iteration; put back the geometry the saved
        # conversation ends at, so the resumed turns (and any pending fix) edit what they discuss
        with open(os.path.join(run_dir, "scratch", FINAL_OUTPUT_FILE), "w") as f:
            f.write(ck["faces"])

    messages = ck["messages"]                            # full live conversation, verbatim (images intact)
    text = ck["text"]
    start_iter = ck["next_iter"]
    new_max = (start_iter - 1) + add_iters               # `add_iters` more completed iterations
    init_best = (ck["best_text"], ck["best_score"], ck["best_len"])
    print(f"=== Resuming {run_dir}: {start_iter - 1} iters done, running {start_iter}..{new_max} ===")

    if ck.get("pending"):
        print("  [resume] answering the pending critic fix before re-critiquing")
        _n0 = _logcount()
        text = run_turn(client, messages, ck["gen_model"])
        _append_agent_response("resume improvement", _n0)
    text = _ensure_shape(client, messages, text, ck["gen_model"], "resume")
    if _shape_error(init_best[0] or "") is not None:
        # the saved best has no shape (the run ended without one): don't revert to it after the rounds
        init_best = (_resolve_faces(text), -1.0, len(messages))

    text, score = auto_refine(client, messages, ck["prompt"], text,
                              ck["gen_model"], ck["critic_model"],
                              max_iters=new_max, tau=ck.get("tau", 0.9),
                              save_prefix=os.path.join(run_dir, "chat"),
                              start_iter=start_iter, init_best=init_best,
                              ref_images=ck.get("images"))
    print(f"=== Resumed refine done (best score {score:.2f}) ===")
    err = save_views(text, os.path.join(run_dir, "chat_refined"))
    print(f"saved {run_dir}/chat_refined.png" if not err else f"build error: {err}")
    resolved = _resolve_faces(text)
    has_shape = _save_shape(os.path.join(run_dir, "final_faces.txt"), resolved)
    if not has_shape:
        print(f"no shape produced (no parseable <faces>): {run_dir}/final_faces.txt not written")
    crit = _write_critic_outputs()
    if crit:
        print(f"=== critic outputs: {crit} ===")
    _append_run_summary(os.path.basename(run_dir))
    if not has_shape:   # non-zero exit, so the pools count it as FAIL and --retry-failed picks it up
        raise SystemExit(f"no shape produced in {run_dir}")
    return text


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", "-s", help="prompt to model (omit when using --resume)")
    ap.add_argument("--resume", default=None,
                    help="path to a run dir with checkpoint.json to continue refining")
    ap.add_argument("--gen-model", default="moonshotai/kimi-k2.6")
    ap.add_argument("--critic-model", default=None,
                    help="critic model id; default picks the per-provider default (see CRITIC_PROVIDER)")
    ap.add_argument("--critic-iters", type=int, default=5,
                    help="critic rounds after generation (default 5; 0 = no critic; with --resume: how many MORE to run)")
    ap.add_argument("--uid", default=None)
    ap.add_argument("--images", default=None,
                    help="comma-separated reference image paths (e.g. front.png,side.png)")
    ap.add_argument("--render-variation", action="store_true", default=None,
                    help="show the exported mesh's surface variation in the model's and critic's renders "
                         "(with --resume: the run's own unless given)")
    args = ap.parse_args()
    # renders are needed by every critic round and by the render_views tool (models not in NO_VISION_MODELS)
    if args.critic_iters > 0 or (not args.resume and args.gen_model not in NO_VISION_MODELS):
        check_renderer()
    if args.resume:
        resume_session(args.resume, add_iters=args.critic_iters, render_variation=args.render_variation)
    else:
        if not args.prompt and not args.images:
            ap.error("--prompt/-s or --images is required (or use --resume <run_dir>)")
        imgs = [x.strip() for x in args.images.split(",") if x.strip()] if args.images else None
        chat_session(args.prompt, gen_model=args.gen_model, critic_model=args.critic_model,
                     refine_iters=args.critic_iters, uid=args.uid, images=imgs,
                     render_variation=bool(args.render_variation))
