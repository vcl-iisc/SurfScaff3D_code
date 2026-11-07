# SurfScaff3D

Reference implementation of **SurfScaff3D: Bridging Geometric and Semantic
Representations for LLM-based 3D Object Generation**, built on the
**Hierarchical Patch Scaffold (HPS)** representation.

SurfScaff3D is a training-free, agentic text/image-to-3D framework. A
**generator agent** programmatically constructs B-spline surface patches via
tools and a **critic agent** scores rendered views and feeds back prescriptive fixes over an N-iteration loop.

## Install

```bash
conda create -n surfscaff3d python=3.10 && conda activate surfscaff3d 
pip install -r requirements.txt
cp .env.example .env                                                    # then fill in API keys
```

**Generators:** you can use an OpenRouter model (`OPENROUTER_API_KEY`), e.g. Claude Sonnet 4.6 (`anthropic/claude-sonnet-4.6`, **SurfScaff3D**), Kimi K2.6 (`moonshotai/kimi-k2.6`, **SurfScaff3D-K**), Qwen3.6-35B-A3B (`qwen/qwen3.6-35b-a3b`, **SurfScaff3D-Q**), MiMo-V2.5 (`xiaomi/mimo-v2.5`), DeepSeek-V4-flash (`deepseek/deepseek-v4-flash-0731`)

To use a text-only (non-vision) model, add its id to `NO_VISION_MODELS` in `chat.py`

**Critic:** Gemini 2.5 Pro — set `CRITIC_PROVIDER` in `.env` to choose the endpoint:
- `gemini` — Google AI Studio direct (`GEMINI_API_KEY`); this is what `.env.example` sets
- `openrouter` — via OpenRouter (`OPENROUTER_API_KEY`, no separate key needed if already set); used if `CRITIC_PROVIDER` is not set


## Single-shape runs

**Generate:**
```bash
# Kimi (SurfScaff3D-K)
python chat.py --gen-model moonshotai/kimi-k2.6 --prompt "a chair" --critic-iters 5
python chat.py --gen-model moonshotai/kimi-k2.6 --images front.png,side.png
python chat.py --gen-model moonshotai/kimi-k2.6 --prompt "a teapot" --images ref.png

# Claude (SurfScaff3D) — via OpenRouter, synchronous
python chat.py --gen-model anthropic/claude-sonnet-4.6 --prompt "a chair" --critic-iters 5
python chat.py --gen-model anthropic/claude-sonnet-4.6 --images front.png,side.png
```
Supports `--critic-iters N` (critic loops after generation; `0` skips the critic).
After generation, an interactive `edit>` prompt lets you refine the shape (`final_faces.txt` then holds the edited shape; the shape before the edits is kept as `before_edits_faces.txt`); attach reference images with `::img`:
```
edit> make the legs thinner
edit> make it look like this ::img ref.png, side.png
edit> ::img target_shape.png
```

**Edit:**
```bash
python edit_run.py <faces_file> "make the legs thinner"
python edit_run.py <faces_file> "make it look like this" --images ref.png

# use Claude instead of Kimi
python edit_run.py <faces_file> "make the legs thinner" --gen-model anthropic/claude-sonnet-4.6
```

## Large-scale runs

**Generate — OpenRouter pool:**
```bash
python run_baseline_pool.py -f my_prompts.txt -j 24 --critic-iters 5
```
`my_prompts.txt` — one shape per line, tab-separated, `#` = comment. `id` is optional; it names the output folder, so ids must be unique.
```
# prompt only
a wooden chair with four legs

# image-only (detected by file extension)
views/shape.png

# prompt + reference images
a wooden chair	front.png,side.png

# with id (names the output folder — useful for dataset evaluation)
3b5195c53046e4bd6e9ad486918e1764	a wooden chair with four legs
9e37ff6c	a bridge	front.png,side.png
7effccaa		views/shape.png
```

**Edit — OpenRouter pool:**
```bash
python run_edit_pool.py -f jobs.jsonl -j 8 --critic-iters 2
```
JSONL jobs file, one edit per line (`uid` names the output folder and must be unique; relative paths resolve from the jobs file's folder):
```
{"src":"chair/final_faces.txt","instruction":"make the legs thinner","shape":"a wooden chair"}
{"uid":"chair1","src":"...","instruction":"...","images":"ref.png,side.png"}
```

## Layout

```
.
├── chat.py                single-shape generator + critic-refine loop 
├── edit_run.py            single-shape editing (text / text+image) via OpenRouter
├── txt_to_obj.py          CLI: convert generated patch .txt -> .obj
├── run_baseline_pool.py   concurrent generation pool (spawns chat.py workers)
├── run_edit_pool.py       concurrent editing pool (spawns edit_run.py workers)
├── examples/library/      few-shot in-context examples
├── rendering/             rendering package 
└── core/                  core engine 
```