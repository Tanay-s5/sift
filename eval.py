# ======================================================================================
# RUN FROM A COLAB CELL (one eval at a time):
#   !python eval.py --arm plain_fine_tuning --tests flores    # FLORES EN->VI and VI->EN (BLEU, chrF++)
#   !python eval.py --arm plain_fine_tuning --tests mlqa      # MLQA-vi (exact match, F1)
#   !python eval.py --arm plain_fine_tuning --tests ifeval    # IFEval (strict / loose)
#
# Extras:  --tests flores mlqa    several at once        --tests all   everything (default)
#          --arm base             the un-tuned model     --quick       8 examples, pipeline check
#          --force                redo finished tests    --n-flores N  random sample of N (default: all rows)
#
# ALL metrics go into ONE file:  <adapter-root>/<arm>/eval_result/results.json
# Each test is written into it the moment it finishes, so a crash never loses finished tests.
# (Per-example files stay separate: *_predictions.jsonl and *_generations.json.)
# ======================================================================================
"""
Evaluate ONE arm (a LoRA adapter folder, or the un-tuned base model) on:
  FLORES  EN->VI and VI->EN   (BLEU + chrF++)     <- new-language ability (translation)
  MLQA-vi                      (exact match, F1)   <- Vietnamese reading comprehension
  IFEval                       (strict / loose)    <- English instruction following (forgetting)

Colab setup (once per runtime, in normal notebook cells BEFORE running this script):
  !pip -q install peft sacrebleu langdetect immutabledict "nltk>=3.9.1"
  from google.colab import drive; drive.mount('/content/drive')
  from huggingface_hub import login; login()
First run with --quick (8 examples per test, ~2 min) to check the whole pipeline, then run without it.
"""
import argparse
import glob
import importlib
import json
import os
import re
import string
import subprocess
import sys
import time
import types
import unicodedata
from collections import Counter

import pandas as pd
import peft
import sacrebleu
import torch
import transformers
from langdetect import DetectorFactory, LangDetectException, detect
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DetectorFactory.seed = 0  # langdetect is random by default; this makes language detection repeatable

# ============================================================ SETTINGS (defaults; the command line overrides some)
ADAPTER_ROOT = "/content/drive/MyDrive/adapters"
DATA_DIR = "/content/drive/MyDrive/SIFT_dataset"  # ifeval.jsonl, mlqa_vi.jsonl, flores.csv
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
SEED = 42
BS = {"flores": 32, "mlqa": 16, "ifeval": 50}  # generation batch sizes; lower them if you hit out-of-memory
MAX_NEW = {"flores": 200, "mlqa": 32, "ifeval": 1280}  # max generated tokens (IFEval needs room for long answers)

MLQA_PROMPT = (
    "Context: {context}\n\nQuestion: {question}\n\n"
    "Answer the question with a short phrase copied from the context. Do not explain."
)
QUICK_N = 8
SCHEMA = 2  # bump when the metrics change; saved results with another schema are rescored automatically
TEST_NAMES = ("flores_en2vi", "flores_vi2en", "mlqa_vi", "ifeval")

# filled in by main() from the command line
ARM = None
N_FLORES = None  # None = use ALL rows in flores.csv; a number = fixed random sample of that size
QUICK = False  # True = 8 examples per test, saved to eval_result_quick/ (pipeline check only)
FORCE = False  # True = redo tests even if their results already exist
ADAPTER_DIR = None
RESULT_DIR = None
RESULTS_PATH = None

model = tok = None  # filled by load_model()


# ============================================================ small helpers
def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:QUICK_N] if QUICK else rows


def mean(xs):
    return sum(xs) / max(1, len(xs))


def pct(xs):
    """Percentage of True values in a list of booleans."""
    return round(100 * mean([bool(x) for x in xs]), 2)


def ci95(xs):
    """95% confidence half-width (in percentage points) for the mean of per-example scores in [0, 1].
    Read it as: score +/- this. Two arms whose difference is smaller than this are not clearly different."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return round(100 * 1.96 * (var / n) ** 0.5, 2)


def lang_of(text):
    """Detected language code ('en', 'vi', ...) or None if the text is empty / undetectable."""
    try:
        return detect(text)
    except LangDetectException:
        return None


def gen_stats(outs, max_new):
    """Facts about the raw generations. truncated_pct is approximate (re-tokenised text): answers that hit the token cap."""
    lens = [len(ids) for ids in tok(outs, add_special_tokens=False).input_ids]
    return {
        "avg_gen_tokens": round(mean(lens), 1),
        "truncated_pct": pct([n >= max_new - 2 for n in lens]),
        "empty_pct": pct([not o.strip() for o in outs]),
    }


def pct_table(counts):
    """{key: [n, strict_hits, loose_hits]} -> {key: {"n", "strict", "loose"}} in percent."""
    return {k: {"n": n, "strict": round(100 * s / n, 2), "loose": round(100 * l / n, 2)}
            for k, (n, s, l) in sorted(counts.items())}


# ============================================================ the ONE results file
def load_results():
    if RESULTS_PATH and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results):
    """Atomic write (temp file, then rename) so an interrupted save can never corrupt results.json."""
    results["last_updated"] = time.strftime("%Y-%m-%d %H:%M")
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, RESULTS_PATH)


def run_info():
    """Everything needed to reproduce / compare this arm later, stored once at the top of results.json."""
    info = {
        "arm": ARM, "base_model": BASE_MODEL, "quick_mode": QUICK, "seed": SEED,
        "generation": {"decoding": "greedy", "dtype": "float16", "max_new_tokens": MAX_NEW, "batch_size": BS},
        "flores_sample": "all rows" if N_FLORES is None else f"random {N_FLORES} (seed {SEED})",
        "environment": {
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__, "sacrebleu": sacrebleu.__version__,
        },
    }
    cfg = os.path.join(ADAPTER_DIR, "adapter_config.json")
    if ARM != "base" and os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as f:
            c = json.load(f)
        info["lora"] = {k: c.get(k) for k in ("r", "lora_alpha", "lora_dropout", "target_modules", "use_rslora", "use_dora")}
    return info


# ============================================================ model + batched generation
def load_model():
    global model, tok
    assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU."
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"  # required for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16).to("cuda")
    if ARM != "base":
        assert os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")), (
            f"No adapter found in {ADAPTER_DIR} (expected adapter_config.json + adapter_model.safetensors)."
        )
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        model = model.merge_and_unload()  # fold the adapter into the weights: same outputs, faster generation
        print(f"loaded base + adapter: {ADAPTER_DIR}")
    else:
        print("evaluating the BASE model (no adapter)")
    model.eval()


@torch.no_grad()
def generate(prompts, max_new_tokens, bs):
    """Greedy, batched, chat-templated generation. Returns one string per prompt, in the original order."""
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)  # longest first: OOM shows up at once
    outs = [None] * len(texts)
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s: s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
            pad_token_id=tok.pad_token_id,
        )
        for j, i in enumerate(idx):
            outs[i] = tok.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"   generated {min(s + bs, len(order))}/{len(order)}  ({time.time() - t0:.0f}s)")
    return outs


def generate_cached(name, prompts, max_new_tokens, bs):
    """Like generate(), but saves the raw outputs first, so a scoring bug can never cost you the generation time."""
    path = os.path.join(RESULT_DIR, f"{name}_generations.json")
    if os.path.exists(path) and not FORCE:
        with open(path, encoding="utf-8") as f:
            outs = json.load(f)
        if len(outs) == len(prompts):
            print(f"   reusing saved generations: {path}")
            return outs
    outs = generate(prompts, max_new_tokens, bs)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(outs, f, ensure_ascii=False)
    return outs


# ============================================================ MLQA scoring (Vietnamese, same normalisation as the official script)
PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")}.union(string.punctuation)


def norm_vi(s):
    s = "".join(ch for ch in s.lower() if ch not in PUNCT)
    s = re.sub(r"\b(của|là|cái|chiếc|những)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = norm_vi(pred).split(), norm_vi(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred, gold):
    return float(norm_vi(pred) == norm_vi(gold))


# ============================================================ IFEval checker (lm-evaluation-harness code, used standalone)
def setup_ifeval_checker():
    """Fetch lm-eval's IFEval checker files from PyPI WITHOUT installing lm-eval itself
    (that could change your Colab package versions), then import them."""
    root = "/content/lm_eval_ifeval"
    if not os.path.exists(f"{root}/lm_eval/tasks/ifeval/utils.py"):
        subprocess.run([sys.executable, "-m", "pip", "download", "lm_eval==0.4.13", "--no-deps",
                        "-d", "/content/lme_whl", "-q"], check=True)
        import zipfile
        whl = glob.glob("/content/lme_whl/lm_eval-*.whl")[0]
        with zipfile.ZipFile(whl) as z:
            for n in z.namelist():
                if n.startswith("lm_eval/tasks/ifeval/") and n.endswith(".py"):
                    z.extract(n, root)
    for name, sub in [("lm_eval", "lm_eval"), ("lm_eval.tasks", "lm_eval/tasks"),
                      ("lm_eval.tasks.ifeval", "lm_eval/tasks/ifeval")]:
        m = types.ModuleType(name)
        m.__path__ = [f"{root}/{sub}"]  # lets submodules import without running lm-eval's heavy __init__ files
        sys.modules[name] = m
    return importlib.import_module("lm_eval.tasks.ifeval.utils")


# ============================================================ the three tests
def load_flores():
    df = pd.read_csv(os.path.join(DATA_DIR, "flores.csv")).dropna(subset=["en", "vi"])
    if N_FLORES is not None and len(df) > N_FLORES:
        df = df.sample(n=N_FLORES, random_state=SEED)
    df = df.reset_index(drop=True)
    return df.head(QUICK_N) if QUICK else df


def run_flores(direction):
    df = load_flores()
    if direction == "en2vi":
        src, ref, tmpl, target = list(df.en), list(df.vi), "Translate to Vietnamese: {}", "vi"
    else:
        src, ref, tmpl, target = list(df.vi), list(df.en), "Translate to English: {}", "en"
    hyp = generate_cached(f"flores_{direction}", [tmpl.format(s) for s in src], MAX_NEW["flores"], BS["flores"])
    bleu = sacrebleu.corpus_bleu(hyp, [ref])
    chrf = sacrebleu.corpus_chrf(hyp, [ref], word_order=2)
    langs = [lang_of(h) for h in hyp]
    metrics = {
        "n": len(hyp),
        "bleu": round(bleu.score, 2),
        "chrfpp": round(chrf.score, 2),
        "bleu_length_ratio": round(bleu.ratio, 3),  # hypothesis length / reference length (1.0 = same length)
        "bleu_brevity_penalty": round(bleu.bp, 3),  # < 1.0 means outputs are too short and BLEU was reduced
        "target_language_pct": pct([lang == target for lang in langs]),  # output really is in the requested language
        "copied_source_pct": pct([h.strip() == s.strip() for h, s in zip(hyp, src)]),  # model just repeated the input
        "multiline_pct": pct(["\n" in h for h in hyp]),  # extra chatter such as "Here is the translation:"
    }
    metrics.update(gen_stats(hyp, MAX_NEW["flores"]))
    rows = [{"id": int(i), "source": s, "reference": r, "hypothesis": h, "lang": lang,
             "chrfpp": round(sacrebleu.sentence_chrf(h, [r], word_order=2).score, 2)}
            for i, s, r, h, lang in zip(df.id, src, ref, hyp, langs)]
    return metrics, rows


def run_mlqa():
    rows = load_jsonl(os.path.join(DATA_DIR, "mlqa_vi.jsonl"))
    prompts = [MLQA_PROMPT.format(context=r["context"], question=r["question"]) for r in rows]
    outs = generate_cached("mlqa_vi", prompts, MAX_NEW["mlqa"], BS["mlqa"])
    ems, f1s, in_ctx, pred_words, gold_words, out_rows = [], [], [], [], [], []
    for r, o in zip(rows, outs):
        pred = o.split("\n")[0].strip()
        golds = list(r["answers"]["text"])
        em, f1 = max(em_score(pred, g) for g in golds), max(f1_score(pred, g) for g in golds)
        p_norm = norm_vi(pred)
        found = bool(p_norm) and p_norm in norm_vi(r["context"])  # did it copy a span, as the prompt asks?
        ems.append(em)
        f1s.append(f1)
        in_ctx.append(found)
        pred_words.append(len(pred.split()))
        gold_words.append(len(golds[0].split()))
        out_rows.append({"id": r.get("id"), "question": r["question"], "gold": golds, "prediction": pred,
                         "em": em, "f1": f1, "in_context": found})
    metrics = {
        "n": len(rows),
        "exact_match": round(100 * mean(ems), 2), "exact_match_ci95": ci95(ems),
        "f1": round(100 * mean(f1s), 2), "f1_ci95": ci95(f1s),
        "answer_in_context_pct": pct(in_ctx),
        "avg_pred_words": round(mean(pred_words), 1), "avg_gold_words": round(mean(gold_words), 1),
        "multiline_pct": pct(["\n" in o.strip() for o in outs]),  # model explained instead of answering briefly
    }
    metrics.update(gen_stats(outs, MAX_NEW["mlqa"]))
    return metrics, out_rows


def fix_ints(kwargs):
    """The saved IFEval kwargs can contain 3.0 instead of 3 (pandas turns ints next to None into floats),
    which crashes the checker (e.g. list index 2.0). Turn whole-number floats back into ints."""
    return [{k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in d.items()} for d in kwargs]


def run_ifeval():
    utils = setup_ifeval_checker()
    rows = load_jsonl(os.path.join(DATA_DIR, "ifeval.jsonl"))
    outs = generate_cached("ifeval", [r["prompt"] for r in rows], MAX_NEW["ifeval"], BS["ifeval"])
    ps, pl, ins, inl, out_rows, errors = [], [], [], [], [], 0
    by_id, by_cat, langs = {}, {}, []
    for r, o in zip(rows, outs):
        doc = dict(r, kwargs=fix_ints(r["kwargs"]))
        try:
            res = utils.process_results(doc, [o])
        except Exception as e:  # one odd response must not kill the whole evaluation; it is counted and reported
            errors += 1
            print(f"   scoring error on key {r['key']}: {type(e).__name__}: {e}")
            n_inst = len(r["instruction_id_list"])
            res = {"prompt_level_strict_acc": False, "prompt_level_loose_acc": False,
                   "inst_level_strict_acc": [False] * n_inst, "inst_level_loose_acc": [False] * n_inst}
        ps.append(res["prompt_level_strict_acc"])
        pl.append(res["prompt_level_loose_acc"])
        ins += res["inst_level_strict_acc"]
        inl += res["inst_level_loose_acc"]
        # per instruction type, e.g. "length_constraints:number_words" and its category "length_constraints"
        for iid, s, l in zip(r["instruction_id_list"], res["inst_level_strict_acc"], res["inst_level_loose_acc"]):
            for table, key in ((by_id, iid), (by_cat, iid.split(":")[0])):
                c = table.setdefault(key, [0, 0, 0])
                c[0] += 1
                c[1] += int(s)
                c[2] += int(l)
        # English-drift check: skip prompts that explicitly ask for another language
        lang = None
        if "language:response_language" not in r["instruction_id_list"]:
            lang = lang_of(o)
            if lang:
                langs.append(lang)
        out_rows.append({"key": r["key"], "prompt": r["prompt"], "response": o, "lang": lang,
                         "strict": bool(res["prompt_level_strict_acc"]), "loose": bool(res["prompt_level_loose_acc"])})
    metrics = {
        "n": len(rows),
        "prompt_strict": round(100 * mean(ps), 2), "prompt_strict_ci95": ci95([float(x) for x in ps]),
        "prompt_loose": round(100 * mean(pl), 2), "prompt_loose_ci95": ci95([float(x) for x in pl]),
        "inst_strict": round(100 * mean(ins), 2), "inst_loose": round(100 * mean(inl), 2),
        "non_english_pct": pct([lg != "en" for lg in langs]),  # English prompts answered in another language
        "vietnamese_pct": pct([lg == "vi" for lg in langs]),  # ... specifically in Vietnamese (forgetting / drift)
        "scoring_errors": errors,
    }
    metrics.update(gen_stats(outs, MAX_NEW["ifeval"]))
    metrics["by_category"] = pct_table(by_cat)
    metrics["by_instruction"] = pct_table(by_id)
    if errors:
        print(f"   WARNING: {errors} responses could not be scored and were counted as failures")
    return metrics, out_rows


def cached(name, fn, expect_n=None):
    """Run one test and append its metrics to results.json. Skipped if a current-format result already exists."""
    saved = load_results().get(name)
    if saved is not None and not FORCE:
        if saved.get("schema") != SCHEMA:
            print(f"[{name}] saved result is from an older version of this script -> rescoring (saved generations are reused)")
        elif expect_n is not None and saved.get("n") != expect_n:
            print(f"[{name}] saved result has n={saved.get('n')} but n={expect_n} is needed -> rerunning")
        else:
            print(f"[{name}] already done -> loaded from {RESULTS_PATH}")
            return saved
    print(f"\n[{name}] running ...")
    t0 = time.time()
    metrics, rows = fn()
    metrics["seconds"] = round(time.time() - t0)
    metrics["finished_at"] = time.strftime("%Y-%m-%d %H:%M")
    metrics["schema"] = SCHEMA
    save_jsonl(rows, os.path.join(RESULT_DIR, f"{name}_predictions.jsonl"))
    results = load_results()  # re-read from disk, then add this test: results.json grows as tests complete
    results[name] = metrics
    save_results(results)
    short = {k: v for k, v in metrics.items() if not isinstance(v, dict)}  # keep the console line readable
    print(f"[{name}] {short}")
    print(f"   saved to {RESULTS_PATH}")
    print("   example output:", repr(str(rows[0].get("hypothesis") or rows[0].get("prediction") or rows[0].get("response"))[:150]))
    return metrics


# ============================================================ summary + command line
def print_summary():
    s = load_results()
    print("\n================ SUMMARY:", ARM, "(QUICK MODE)" if QUICK else "", "================")
    for k in TEST_NAMES:
        m = s.get(k)
        if m is None:
            continue
        if m.get("schema") != SCHEMA:
            print(f"{k:14s} (old format: rerun this test to refresh it)")
        elif k.startswith("flores"):
            print(f"{k:14s} BLEU {m['bleu']:6.2f}   chrF++ {m['chrfpp']:6.2f}   right language {m['target_language_pct']:5.1f}%   "
                  f"length ratio {m['bleu_length_ratio']:.2f}   (n={m['n']})")
        elif k == "mlqa_vi":
            print(f"{k:14s} EM   {m['exact_match']:6.2f} +/-{m['exact_match_ci95']:.1f}   F1 {m['f1']:6.2f} +/-{m['f1_ci95']:.1f}   "
                  f"answer in context {m['answer_in_context_pct']:5.1f}%   (n={m['n']})")
        else:
            print(f"{k:14s} prompt strict {m['prompt_strict']:.2f} +/-{m['prompt_strict_ci95']:.1f} / loose {m['prompt_loose']:.2f}   "
                  f"inst strict {m['inst_strict']:.2f} / loose {m['inst_loose']:.2f}   "
                  f"non-English {m['non_english_pct']:.1f}%   truncated {m['truncated_pct']:.1f}%   (n={m['n']})")
    print("saved to", RESULTS_PATH)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ONE arm (LoRA adapter folder or the base model).")
    p.add_argument("--arm", required=True,
                   help='adapter folder name inside --adapter-root, or "base" for the un-tuned model')
    p.add_argument("--tests", nargs="+", default=["all"], choices=["flores", "mlqa", "ifeval", "all"],
                   help="which evals to run (default: all)")
    p.add_argument("--quick", action="store_true", help="8 examples per test, saved to eval_result_quick/")
    p.add_argument("--force", action="store_true", help="redo tests even if results already exist")
    p.add_argument("--n-flores", type=int, default=None,
                   help="FLORES sentences per direction (default: all rows in flores.csv)")
    p.add_argument("--adapter-root", default=ADAPTER_ROOT)
    p.add_argument("--data-dir", default=DATA_DIR)
    return p.parse_args()


def main():
    global ARM, ADAPTER_ROOT, DATA_DIR, N_FLORES, QUICK, FORCE, ADAPTER_DIR, RESULT_DIR, RESULTS_PATH
    args = parse_args()
    ARM, ADAPTER_ROOT, DATA_DIR = args.arm, args.adapter_root, args.data_dir
    N_FLORES, QUICK, FORCE = args.n_flores, args.quick, args.force
    ADAPTER_DIR = os.path.join(ADAPTER_ROOT, ARM)
    RESULT_DIR = os.path.join(ADAPTER_DIR, "eval_result_quick" if QUICK else "eval_result")
    RESULTS_PATH = os.path.join(RESULT_DIR, "results.json")
    tests = ["flores", "mlqa", "ifeval"] if "all" in args.tests else args.tests

    os.makedirs(RESULT_DIR, exist_ok=True)
    load_model()

    results = load_results()
    results.pop("date", None)  # key from the previous version of this script
    results.update(run_info())
    save_results(results)

    if "flores" in tests:
        n_flores = len(load_flores())
        cached("flores_en2vi", lambda: run_flores("en2vi"), expect_n=n_flores)
        cached("flores_vi2en", lambda: run_flores("vi2en"), expect_n=n_flores)
    if "mlqa" in tests:
        cached("mlqa_vi", run_mlqa)
    if "ifeval" in tests:
        cached("ifeval", run_ifeval)

    print_summary()


if __name__ == "__main__":
    main()