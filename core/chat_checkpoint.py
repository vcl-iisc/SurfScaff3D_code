# Checkpoint/resume for chat.py's critic-refine loop.
# Serializes the live messages list (reasoning_details, tool_calls, base64 renders); api_log strips the renders.
import os, json, copy


def _msg_to_dict(m):
    # One message -> a plain JSON-serializable dict, preserving ALL fields.
    if isinstance(m, dict):
        return copy.deepcopy(m)
    if hasattr(m, "model_dump"):
        return m.model_dump(exclude_none=True)
    if hasattr(m, "to_dict"):
        return m.to_dict()
    raise TypeError(f"cannot serialize message of type {type(m)}")


def serialize_messages(messages):
    return [_msg_to_dict(m) for m in messages]


# fields whose loss would change the next API request -- asserted on round-trip
_CRITICAL_FIELDS = ("role", "content", "reasoning_details", "tool_calls", "tool_call_id")


def assert_roundtrip(messages):
    # Returns the serialized list; raises AssertionError on any mismatch or stripped-image leak.
    ser = serialize_messages(messages)
    reloaded = json.loads(json.dumps(ser))
    assert len(reloaded) == len(messages), f"message count changed: {len(messages)}->{len(reloaded)}"
    for i, (orig, rl) in enumerate(zip(messages, reloaded)):
        od = _msg_to_dict(orig)
        for f in _CRITICAL_FIELDS:
            assert od.get(f) == rl.get(f), f"msg {i}: field {f!r} not preserved through round-trip"
        oc, rc = od.get("content"), rl.get("content")
        if isinstance(oc, list):
            o_imgs = [p.get("image_url", {}).get("url") for p in oc if isinstance(p, dict) and p.get("type") == "image_url"]
            r_imgs = [p.get("image_url", {}).get("url") for p in rc if isinstance(p, dict) and p.get("type") == "image_url"]
            assert o_imgs == r_imgs, f"msg {i}: image url(s) not preserved"
            assert not any("base64 image omitted" in (u or "") for u in r_imgs), \
                f"msg {i}: a STRIPPED image placeholder leaked into the checkpoint (must keep real base64)"
    return ser


CHECKPOINT_NAME = "checkpoint.json"


def save_checkpoint(run_dir, messages, *, text, best_text, best_score, best_len,
                    next_iter, max_iters, prompt, gen_model, critic_model, tau, pending=False, images=None,
                    faces=None, render_variation=False):
    # Write a faithful checkpoint atomically (verifies round-trip first).
    # next_iter = the iteration about to run (iters 1..next_iter-1 are fully done).
    # pending   = True if `messages` already ends with an un-answered critic fix 
    # faces     = the geometry the conversation ends at (final_output.txt now)
    #             since a finished run reverts final_output.txt to its best iteration.
    # render_variation = the run's --render-variation; 
    ser = assert_roundtrip(messages)             # fail loudly rather than write a lossy checkpoint
    ckpt = {
        "version": 1,
        "messages": ser,
        "text": text,
        "best_text": best_text, "best_score": best_score, "best_len": best_len,
        "next_iter": next_iter, "max_iters": max_iters, "pending": pending,
        "prompt": prompt, "gen_model": gen_model, "critic_model": critic_model, "tau": tau,
        "images": [os.path.abspath(p) for p in images] if images else None,
        "faces": faces,
        "render_variation": render_variation,
    }
    path = os.path.join(run_dir, CHECKPOINT_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    os.replace(tmp, path)                         # atomic: never leave a half-written checkpoint
    return path


def load_checkpoint(run_dir):
    with open(os.path.join(run_dir, CHECKPOINT_NAME)) as f:
        return json.load(f)
