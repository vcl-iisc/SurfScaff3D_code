# Concurrent edit pool — runs edit_run.py over a JSONL jobs file with N parallel workers (any OpenRouter model via
# --model; default Kimi K2.6).
import argparse, json, os, subprocess, sys
from collections import Counter
from run_baseline_pool import run_pool, _ts

ROOT   = os.path.dirname(os.path.abspath(__file__))
RUNNER = "edit_run.py"


def _load_jobs(path):
    base = os.path.dirname(os.path.abspath(path))
    jobs = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: invalid JSON: {e}")
            if not job.get("src"):
                raise SystemExit(f"{path}:{ln}: missing required field 'src'")
            if not job.get("instruction"):
                raise SystemExit(f"{path}:{ln}: missing required field 'instruction'")
            if not os.path.isabs(job["src"]):
                job["src"] = os.path.join(base, job["src"])
            if job.get("images"):   # like src: relative image paths resolve from the jobs file's folder
                imgs = job["images"].split(",") if isinstance(job["images"], str) else job["images"]
                job["images"] = [p.strip() if os.path.isabs(p.strip()) else os.path.join(base, p.strip())
                                 for p in imgs if p.strip()]
            jobs.append(job)
    return jobs


def main():
    ap = argparse.ArgumentParser(description="Concurrent edit_run.py pool")
    ap.add_argument("--jobs-file", "-f", required=True, help="JSONL file of edit jobs")
    ap.add_argument("--model", default="moonshotai/kimi-k2.6")
    ap.add_argument("--critic-model", default=None,
                    help="critic model id; default: the CRITIC_PROVIDER's own default (see .env)")
    ap.add_argument("--concurrency", "-j", type=int, default=8)
    ap.add_argument("--critic-iters", type=int, default=2, help="critic rounds per edit (default 2; 0 = no critic)")
    ap.add_argument("--output-dir", default="edit_out")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip jobs whose output dir already has final_faces.txt")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-job wall-clock seconds; 0 disables")
    ap.add_argument("--render-variation", action="store_true",
                    help="passed to edit_run.py: show surface variation in the model's and critic's renders")
    args = ap.parse_args()
    model_base = args.model.split("/")[-1]

    jobs = _load_jobs(args.jobs_file)
    for i, job in enumerate(jobs):
        uid = str(job.get("uid") or "").strip() or str(i).zfill(4)
        job["_uid"] = uid
        job["_out"] = os.path.join(ROOT, args.output_dir, model_base, uid)
    dup = sorted(u for u, c in Counter(j["_uid"] for j in jobs).items() if c > 1)
    if dup:
        raise SystemExit(f"{args.jobs_file}: repeated uid(s) {', '.join(dup)} — each job needs its own uid "
                         "(it names the output folder)")

    if args.skip_done:
        before = len(jobs)
        jobs = [j for j in jobs if not os.path.exists(os.path.join(j["_out"], "final_faces.txt"))]
        print(f"[{_ts()}] --skip-done: {before - len(jobs)} done, {len(jobs)} remaining", flush=True)

    print(f"[{_ts()}] EDIT POOL: {len(jobs)} jobs | {args.concurrency}-way | model={args.model}", flush=True)

    def launch(job):
        out_dir = job["_out"]
        os.makedirs(out_dir, exist_ok=True)
        # positionals: src, instruction, [shape]; then --key=value flags
        cmd = [sys.executable, "-u", RUNNER, job["src"], job["instruction"]]
        if job.get("shape"):
            cmd.append(job["shape"])
        cmd += [
            f"--gen-model={args.model}",
            f"--critic-iters={args.critic_iters}",
            f"--out-dir={out_dir}",
        ]
        if args.critic_model:
            cmd.append(f"--critic-model={args.critic_model}")
        if job.get("images"):
            cmd.append(f"--images={','.join(job['images'])}")
        if args.render_variation:
            cmd.append("--render-variation")
        logf = open(os.path.join(out_dir, "run.log"), "w")
        p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                             stdout=logf, stderr=subprocess.STDOUT)
        print(f"[{_ts()}] START {job['_uid']}", flush=True)
        return p, logf

    ok, fail = run_pool([(j["_uid"], j["_out"], j) for j in jobs], launch, args.concurrency, args.timeout)
    print(f"[{_ts()}] DONE: {ok} ok, {fail} failed -> {args.output_dir}/{model_base}/", flush=True)


if __name__ == "__main__":
    main()
