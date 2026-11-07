# Concurrent generation pool — runs chat.py over a prompts file with N parallel workers (any OpenRouter model via
# --model; default Kimi K2.6).
import argparse, json, os, re, subprocess, sys, time
from collections import Counter
from datetime import datetime

from core.batch_input import _load_prompts_file

ROOT = os.path.dirname(os.path.abspath(__file__))       # repo root
RUNNER = "chat.py"                                       # generator script, one subprocess per shape
def _ts(): return datetime.now().strftime("%H:%M:%S")


def _iters_done(d):
    # Critic iterations already completed in run dir `d`, from its checkpoint.json.
    # checkpoint stores next_iter = the iteration about to run, so done = next_iter - 1.
    try:
        with open(os.path.join(ROOT, d, "checkpoint.json")) as f:
            return max(0, int(json.load(f).get("next_iter", 1)) - 1)
    except Exception:
        return 0


def _load_prompts(path):
    # Return [(name, prompt, images)]. The file is read by core/batch_input.py
    # prompts file (tab-separated, '#' = comment; image paths resolve from the file's folder):
    #   prompt / img1,img2 (image-only) / prompt<TAB>imgs / id<TAB>prompt / id<TAB>prompt<TAB>imgs / id<TAB><TAB>imgs
    # name (the output folder) = the id, else a slug of the prompt (or of the first image's file name).
    # A repeated id is an error; a repeated slug gets a _2, _3, ... suffix, so no prompt is dropped.
    rows = [(re.sub(r"[^A-Za-z0-9._-]+", "_", uid).strip("_"), prompt, imgs)
            for uid, prompt, imgs in _load_prompts_file(path)]
    dup = sorted(n for n, c in Counter(uid for uid, _, _ in rows if uid).items() if c > 1)
    if dup:
        raise SystemExit(f"{path}: repeated id(s) {', '.join(dup)} — each id names its own output folder")
    out, used = [], {uid for uid, _, _ in rows if uid}
    for uid, prompt, imgs in rows:
        name = uid
        if not name:
            src = prompt or (os.path.splitext(os.path.basename(imgs[0]))[0] if imgs else "")
            base = re.sub(r"[^a-z0-9]+", "_", src.lower()).strip("_")[:40] or "shape"
            name, k = base, 2
            while name in used:
                name, k = f"{base}_{k}", k + 1
        used.add(name); out.append((name, prompt, imgs))
    return out


def run_pool(jobs, launch, concurrency, timeout):
    # Run one subprocess per job, at most `concurrency` at a time; 
    # A job is OK when it exits 0 and wrote out_dir/final_faces.txt. Returns (ok, fail).
    pending, running = list(jobs), {}
    ok = fail = 0
    while pending or running:
        while pending and len(running) < concurrency:
            name, out_dir, job = pending.pop(0)
            running[name] = (*launch(job), time.time(), out_dir)
        # kill any child past the per-job timeout; it gets reaped (as FAIL) on the next poll below
        if timeout:
            for name, (p, _, t0, _) in running.items():
                if p.poll() is None and time.time() - t0 > timeout:
                    p.kill()
                    print(f"[{_ts()}] TIMEOUT {name} (> {timeout}s) -> killed", flush=True)
        for name in [n for n, (p, *_) in running.items() if p.poll() is not None]:
            p, logf, t0, out_dir = running.pop(name)
            logf.close()
            done_ok = p.returncode == 0 and os.path.exists(os.path.join(out_dir, "final_faces.txt"))
            ok += done_ok; fail += not done_ok
            print(f"[{_ts()}] {'OK  ' if done_ok else 'FAIL'} [{ok+fail}/{len(jobs)}] {name}  "
                  f"{time.time()-t0:6.0f}s", flush=True)
        time.sleep(2)
    return ok, fail


def main():
    ap = argparse.ArgumentParser(description="Concurrent baseline (chat.py) generation over a prompts file")
    ap.add_argument("--prompts-file", "-f", required=True,
                    help="prompts file, one shape per line (# = comment), tab-separated: "
                         "'prompt', 'img1,img2', 'prompt<TAB>img1,img2', 'id<TAB>prompt', "
                         "'id<TAB>prompt<TAB>img1,img2' or 'id<TAB><TAB>img1,img2'; ids must be unique; "
                         "a relative path resolves from the repo root")
    ap.add_argument("--subdir", default="pool", help="chat_out/<model>/<subdir>/<id>/ (default: pool)")
    ap.add_argument("--model", default="moonshotai/kimi-k2.6")
    ap.add_argument("--critic-model", default=None,
                    help="critic model id; default: the CRITIC_PROVIDER's own default (see .env)")
    ap.add_argument("--concurrency", "-j", type=int, default=24)
    ap.add_argument("--critic-iters", type=int, default=5,
                    help="critic rounds (0 = no critic). Fresh/--resume-all: passed straight to chat.py (resume = "
                         "this many MORE). --retry-failed: treated as the TOTAL target -> each resumed "
                         "shape runs only (target - iters_already_done) more, so all end at this many.")
    ap.add_argument("--timeout", type=int, default=7200, help="per-shape wall-clock seconds; 0 disables")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip shapes that already have final_faces.txt (re-run a crashed sweep)")
    ap.add_argument("--resume-all", action="store_true",
                    help="CONTINUE refining every shape that has a checkpoint.json (--critic-iters "
                         "= how many MORE iterations); shapes without one are skipped")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-run only the FAILED/unfinished shapes (no final_faces.txt). Each one RESUMES "
                         "from its checkpoint.json if present (continue where it died), else regenerates "
                         "fresh. Leaves already-done shapes untouched.")
    ap.add_argument("--render-variation", action="store_true",
                    help="passed to chat.py: show surface variation in the model's and critic's renders "
                         "(resumed shapes keep their own setting unless this is given)")
    args = ap.parse_args()
    model_base = args.model.split("/")[-1]

    items = _load_prompts(os.path.join(ROOT, args.prompts_file))
    def run_dir(name): return os.path.join("chat_out", model_base, args.subdir, name)

    if args.resume_all:
        before = len(items)
        items = [(n, p, imgs) for n, p, imgs in items
                 if os.path.exists(os.path.join(ROOT, run_dir(n), "checkpoint.json"))]
        print(f"[{_ts()}] --resume-all: continuing {len(items)}/{before} shapes that have a checkpoint "
              f"(+{args.critic_iters} iters each)", flush=True)
    elif args.retry_failed:
        before = len(items)
        items = [(n, p, imgs) for n, p, imgs in items
                 if not os.path.exists(os.path.join(ROOT, run_dir(n), "final_faces.txt"))]
        n_ckpt = sum(os.path.exists(os.path.join(ROOT, run_dir(n), "checkpoint.json")) for n, *_ in items)
        print(f"[{_ts()}] --retry-failed: {before - len(items)} done (left alone), {len(items)} to retry "
              f"-> {n_ckpt} resume from checkpoint, {len(items) - n_ckpt} regenerate fresh", flush=True)
    elif args.skip_done:
        before = len(items)
        items = [(n, p, imgs) for n, p, imgs in items
                 if not os.path.exists(os.path.join(ROOT, run_dir(n), "final_faces.txt"))]
        print(f"[{_ts()}] --skip-done: {before - len(items)} already done, {len(items)} remaining", flush=True)

    print(f"[{_ts()}] BASELINE: {len(items)} shapes -> chat_out/{model_base}/{args.subdir}/<id>/ | "
          f"{args.concurrency}-way | model={args.model}", flush=True)

    def launch(item):
        name, prompt, imgs = item
        d = run_dir(name)
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
        # chat.py drives generation+refine. Two modes:
        #  - fresh: chat_session(uid=<subdir>/<name>, -s prompt) ; - resume: resume_session(<dir>) from checkpoint.
        # --resume-all always resumes; --retry-failed resumes per-shape IF a checkpoint exists, else fresh.
        do_resume = args.resume_all or (args.retry_failed and
                                        os.path.exists(os.path.join(ROOT, d, "checkpoint.json")))
        # chat.py --resume treats --critic-iters as "this many MORE". For --retry-failed we want every
        # shape to land at the SAME total, so subtract what it already did; --resume-all keeps "MORE".
        child_iters = args.critic_iters
        if do_resume and args.retry_failed and not args.resume_all:
            child_iters = max(0, args.critic_iters - _iters_done(d))
        cmd = [sys.executable, "-u", RUNNER, "--gen-model", args.model]
        if args.critic_model:
            cmd += ["--critic-model", args.critic_model]
        if args.render_variation:
            cmd.append("--render-variation")
        if do_resume:
            logf = open(os.path.join(ROOT, d, "run.log"), "a")    # append: keep the original run.log
            cmd += ["--resume", d, "--critic-iters", str(child_iters)]
        else:
            logf = open(os.path.join(ROOT, d, "run.log"), "w")
            cmd += ["--uid", f"{args.subdir}/{name}", "--critic-iters", str(child_iters)]
            if prompt:
                cmd += ["-s", prompt]
            if imgs:
                cmd += ["--images", ",".join(imgs)]
        p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT)
        tag = "RESUME" if do_resume else "START"
        extra = f" ({_iters_done(d)} done +{child_iters} -> {args.critic_iters})" if (do_resume and args.retry_failed and not args.resume_all) else ""
        print(f"[{_ts()}] {tag} {name}{extra}", flush=True)
        return p, logf

    ok, fail = run_pool([(it[0], os.path.join(ROOT, run_dir(it[0])), it) for it in items],
                        launch, args.concurrency, args.timeout)
    print(f"[{_ts()}] FINISHED: {ok} ok, {fail} failed -> chat_out/{model_base}/{args.subdir}/", flush=True)


if __name__ == "__main__":
    main()
