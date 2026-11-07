import base64
import json
import os
import re
import time

from .prompt import _read_image


# Critic VLM provider. Both providers speak the OpenAI chat-completions wire format, so ONLY the
# base_url, the API-key env var, the model-name prefix and the Gemini thinking setting differ — the
# messages (image_url data-URIs), the retry loop and the response parsing are identical for both.
# Pick the provider with the CRITIC_PROVIDER env var.
_PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "google/gemini-2.5-pro",
        "thinking": {"extra_body": {"reasoning": {"max_tokens": 8192}}},   # passed to Google as thinkingBudget
    },
    "gemini": {  # Google AI Studio's OpenAI-compatible endpoint; uses a Google (NOT OpenRouter) key
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-pro",
        "thinking": {"reasoning_effort": "medium"},                        # = an 8,192-token budget on Gemini 2.5
    },
}


def _resolve_provider():
    name = os.environ.get("CRITIC_PROVIDER", "openrouter").lower()
    if name not in _PROVIDERS:
        raise ValueError(f"unknown CRITIC_PROVIDER {name!r}; choose from {sorted(_PROVIDERS)}")
    return _PROVIDERS[name]


def _critic_client():
    from openai import OpenAI
    p = _resolve_provider()
    return OpenAI(base_url=p["base_url"], api_key=os.environ[p["api_key_env"]])


_MAX_RETRIES = 4   # extra attempts when the critic returns an empty or unparseable reply, or a temporary API error


_CRITIC_SYS = (
    "You are a meticulous 3D shape critic. You are shown FOUR rendered views of a 3D object built "
    "from surface patches: two opposite three-quarter views looking down at it - one from the "
    "upper-front-right and one from the upper-back-left (together showing the front, back, both "
    "sides, and top) plus a straight top-down view and a straight bottom-up view so you can also "
    "judge the base/underside and footprint. You also get the text description it was meant to match. The object is "
    "rendered in a single uniform color with no texture, so judge only the GEOMETRY (not color or "
    "material): correct parts present, proportions, symmetry, and that parts connect properly. "
    "Also judge REALISM: the shape should look like a believable, natural example of what it depicts. "
    "EXCEPTION if the description "
    "explicitly asks for a non-realistic style (e.g. cartoon, cartoonish, low-poly, blocky, geometric, "
    "abstract, minimalist, simplified, toy/toy-like, chibi), judge it against that style instead of "
    "realism. So if no style is mentioned, assume a realistic, natural-looking result is wanted. "
)

# Appended to the critic system prompt when REFERENCE images are supplied (image-guided generation):
# the rendered object must match those images, which are the ground truth and outrank the text.
_CRITIC_REF_NOTE = (
    "\n\nYou are ALSO given one or more REFERENCE images: these show the actual target object the "
    "rendered patches are meant to reproduce. Treat the reference images as the GROUND TRUTH - judge "
    "how faithfully the rendered geometry matches their shape, parts, and proportions. Where the "
    "reference images and the text description disagree, the reference images take priority. "
    "If a text description is also provided, it may specify style, features, or requirements "
    "that supplement the visual reference and account for both when scoring. "
    "If no meaningful text description is given (the target is specified ONLY by the images), judge the "
    "rendered object purely against the reference images. Judge only geometry, not the "
    "color/material/background of the reference images."
)


# reference-image formats the Gemini critic takes as-is (it does not list GIF); anything else -> PNG
_CRITIC_IMG = {"image/jpeg", "image/png", "image/webp"}


def _file_image_url(path):
    # data: URL for a reference image FILE on disk, for the OpenAI-format critic call. Read like the
    # generator's images: real format from the bytes, anything Gemini does not accept converted to PNG.
    mt, raw = _read_image(path, supported=_CRITIC_IMG)
    b64 = base64.standard_b64encode(raw).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}}


_CRITIC_INSTR = (
    "Return STRICT JSON: {{\"score\": <0..1 fraction of checks passed>, "
    "\"issues\": [{{\"part\": \"<object/part name or 'global'>\", "
    "\"problem\": \"<what is wrong, visible in the render>\", "
    "\"fix\": \"<concrete actionable change>\"}}]}}. "
    "Use \"global\" for whole-object problems (wrong overall proportions, missing major part). "
    "When a fix could distort other parts, you can phrase it as 'change X while keeping Y unchanged' "
    "so that correcting one issue does not break a part that was already right. "
    "You may also briefly note what already looks good when it helps frame the fix (e.g. what to keep while making the change). "
    "Report the most important issues, AT MOST THREE (the issues list must contain at most three items), "
    "so they can be fixed before the next review. "
    "If you return more than one issue, make sure they can be solved independently (i.e. fix of one issue not interfering with "
    "the other). "
    "A 'global' issue must be the ONLY issue so never mix a global issue with per-part issues. "
    "If the object is good, return an empty issues list and a high score.\n\n"
    "Object description:\n{prompt}"
)


def _parse_verdict(raw):
    # The first JSON object in the reply with a numeric score and a list of issue objects, as
    # (score, issues); None if there is none (empty, malformed, no score, or issues that are not objects).
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", raw or ""):
        try:
            data, _ = dec.raw_decode(raw, m.start())
            score, issues = float(data["score"]), data.get("issues") or []
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        if isinstance(issues, list) and all(isinstance(i, dict) for i in issues):
            return score, issues
    return None


def _ask_critic(system, content, model):
    # Send one critic request and parse its JSON verdict (shared by vlm_critic and vlm_edit_critic).
    import openai
    client = _critic_client()
    model = model or _resolve_provider()["model"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    # medium thinking for a Gemini critic: the same 8,192-token thinking budget on both endpoints
    thinking = _resolve_provider()["thinking"] if "gemini" in model.lower() else {}
    # gemini/OpenRouter intermittently returns an EMPTY (or non-JSON) completion, or a transient API
    # error (rate limit, 5xx, dropped connection). Retry until we get a parseable critique rather than
    # fabricating a bogus "could not parse" issue and feeding it to the agent (which makes it "fix" a
    # problem that isn't there). If every attempt fails, mark the critique UNAVAILABLE and return NO
    # issues — the caller must not send anything to the agent. Other API errors (bad key, bad request) raise.
    score = issues = resp = err = None
    for _attempt in range(_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(model=model, temperature=0.0, messages=messages, **thinking)
        except openai.APIConnectionError as e:                      # incl. timeouts
            resp, err = None, e
        except openai.APIStatusError as e:
            if e.status_code != 429 and e.status_code < 500:
                raise
            resp, err = None, e
        choices = (resp.choices if resp is not None else None) or []
        raw = ((choices[0].message.content if choices else "") or "").strip()
        verdict = _parse_verdict(raw)
        if verdict:
            score, issues = verdict
            # never mix a global issue with per-part ones: a global issue implies a broad
            # rework, so if any issue is global, keep only the global one(s).
            if any(i.get("part") == "global" for i in issues):
                issues = [i for i in issues if i.get("part") == "global"]
            break
        # empty/unparseable -> the gemini/OpenRouter empty-completion glitch tends to come in a brief
        # window, so back off (spread the retries out) instead of hammering it back-to-back.
        if _attempt < _MAX_RETRIES:
            time.sleep(min(3 * 2 ** _attempt, 20))   # exponential: 3, 6, 12, 20 s
    unavailable = score is None
    if unavailable:
        score, issues = None, []   # never fabricate a geometry issue from a critic failure
    # log-safe request: omit the multi-MB base64 images so the log stays readable
    log_content = [{"type": "image_url", "image_url": {"url": "<base64 png omitted>"}}
                   if c.get("type") == "image_url" else c for c in content]
    log_req = {"model": model, "temperature": 0.0, **thinking, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": log_content},
    ]}
    return {"score": score, "issues": issues, "raw": raw, "unavailable": unavailable, "request": log_req,
            "response": resp.model_dump() if resp is not None else {"error": f"{type(err).__name__}: {err}"}}


def vlm_critic(prompt, images, model=None, ref_images=None):
    # images: a list of (label, png_bytes) sent as separate, individually-labelled images
    # ref_images: optional list of reference image FILE PATHS the object must match (image-guided
    #   generation). When given, they are shown FIRST as the ground-truth target and the critic
    #   scores how faithfully the rendered geometry reproduces them. Missing files are skipped.
    # model defaults to the CRITIC_PROVIDER's gemini-2.5-pro id.
    ref_paths = [p for p in (ref_images or []) if p and os.path.exists(p)]
    sys = _CRITIC_SYS + (_CRITIC_REF_NOTE if ref_paths else "")
    text = _CRITIC_INSTR.format(prompt=prompt)
    content = [{"type": "text", "text": text}]
    # reference images FIRST, clearly marked as the target the rendered object must match
    for i, rp in enumerate(ref_paths, 1):
        content.append({"type": "text",
                        "text": f"REFERENCE image {i} (the target the rendered object must match):"})
        content.append(_file_image_url(rp))
    if ref_paths:
        content.append({"type": "text", "text": "Rendered views of the CURRENT 3D object to evaluate:"})
    for i, (label, png) in enumerate(images, 1):
        b64 = base64.standard_b64encode(png).decode("utf-8")
        caption = f"Image {i}: {label}" if label else f"Image {i}"
        content.append({"type": "text", "text": caption})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return _ask_critic(sys, content, model)


_EDIT_CRITIC_SYS = (
    "You are a meticulous reviewer of EDITS to a 3D object built from surface patches. You are shown "
    "rendered views of the object BEFORE the edit and AFTER the edit. Each set has FOUR views: two "
    "opposite three-quarter views looking down at it (upper-front-right and upper-back-left, together "
    "showing front, back, both sides, and top), plus a straight top-down and a straight bottom-up view. "
    "The object is a single uniform color with no texture, so judge only the GEOMETRY (not color or material). "
    "Judge that the edit was APPLIED — the AFTER object carries out the requested edit — and PRESERVED — "
    "everything the edit did not ask to change is unchanged from BEFORE (same parts, proportions, positions). "
)

# Appended to the edit critic system prompt when reference images are supplied.
# The images show what the edited result should look like — judge fidelity to them in addition
# to checking APPLIED and PRESERVED.
_EDIT_CRITIC_REF_NOTE = (
    "\n\nYou are ALSO given REFERENCE image(s) showing what the edited result should look like. "
    "In addition to judging APPLIED and PRESERVED, judge how faithfully the AFTER render matches "
    "the reference image's shape, parts, and proportions. "
    "If a text description or instruction is also provided, it may specify style, features, or requirements "
    "that supplement the visual reference — account for both when scoring. "
    "Judge only geometry — ignore color, "
    "material, lighting, and background of the reference images."
)

_EDIT_CRITIC_INSTR = (
    "Return STRICT JSON: {{\"score\": <0..1, where 1.0 means the edit is fully applied AND all unrelated "
    "parts are preserved>, \"issues\": [{{\"part\": \"<part name or 'global'>\", "
    "\"problem\": \"<what is wrong in the AFTER render: edit not applied, applied wrongly, or an "
    "unrelated part changed from BEFORE>\", \"fix\": \"<concrete actionable change>\"}}]}}. "
    "Report the most important issues, AT MOST THREE (the issues list must contain at most three items), "
    "each independently fixable. "
    "A 'global' issue must be the ONLY issue — never mix a global issue with per-part issues. "
    "If the edit is correct and nothing unrelated changed, return an empty issues list and a high score.\n\n"
    "Requested edit:\n{instruction}{shape_ctx}"
)


def vlm_edit_critic(instruction, before_images, after_images, shape="",
                    model=None, ref_images=None):
    # ref_images: optional list of reference image FILE PATHS showing what the edited result should look
    #   like (image-guided edits). When given, shown FIRST and the critic also judges fidelity to them.
    def _img_parts(images, tag):
        parts = []
        for i, (label, png) in enumerate(images or [], 1):
            b64 = base64.standard_b64encode(png).decode("utf-8")
            caption = f"{tag} image {i}: {label}" if label else f"{tag} image {i}"
            parts.append({"type": "text", "text": caption})
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        return parts

    ref_paths = [p for p in (ref_images or []) if p and os.path.exists(p)]
    sys = _EDIT_CRITIC_SYS + (_EDIT_CRITIC_REF_NOTE if ref_paths else "")
    shape_ctx = (f"\n\nFor context, the object overall is: {shape}\nThe edit should change only what it "
                 "asks for; treat the rest as parts that must be PRESERVED.") if shape else ""
    content = [{"type": "text", "text": _EDIT_CRITIC_INSTR.format(instruction=instruction, shape_ctx=shape_ctx)}]
    # reference images first, clearly marked as the target the edit should match
    for i, rp in enumerate(ref_paths, 1):
        content.append({"type": "text", "text": f"REFERENCE image {i} (target the edit result should match):"})
        content.append(_file_image_url(rp))
    if before_images:
        content.append({"type": "text", "text": "=== BEFORE the edit ==="})
        content += _img_parts(before_images, "BEFORE")
    content.append({"type": "text", "text": "=== AFTER the edit ==="})
    content += _img_parts(after_images, "AFTER")
    return _ask_critic(sys, content, model)
