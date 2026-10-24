import os
import json
import time
import csv
import random
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model

# ==========================================
# 0. PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

ARM_NAME = "replay_5pct_50k"
OUT = os.path.join(SCRIPT_DIR, "results", ARM_NAME)
os.makedirs(OUT, exist_ok=True)

LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, "models", "llama-3.2-1b-instruct")
MODEL_DIR = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "meta-llama/Llama-3.2-1B-Instruct"

OPUS100_CSV_PATH = os.path.join(SCRIPT_DIR, "opus100.csv")
OPUS100_JSON_PATH = os.path.join(SCRIPT_DIR, "opus100.json")

# Alpaca rows [15000:15200] are reserved as eval probes; never train on them
PROBE_START, PROBE_END = 15000, 15200

# Hyperparameters
N_TOTAL = 50_000
N_REPLAY = 2_500
N_VI = N_TOTAL - N_REPLAY
MAX_LEN = 256
BS = 4
ACCUM = 4
LR = 2e-4
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# 1. DATA
# ==========================================
def load_local_opus100(n_samples):
    pairs = []
    if os.path.exists(OPUS100_JSON_PATH):
        print(f"Loading OPUS-100 from: {OPUS100_JSON_PATH}")
        with open(OPUS100_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_slice = data[:n_samples] if isinstance(data, list) else data.get("train", [])[:n_samples]
        for item in data_slice:
            en = item.get("en") or item.get("english") or item.get("translation", {}).get("en")
            vi = item.get("vi") or item.get("vietnamese") or item.get("translation", {}).get("vi")
            if en and vi:
                pairs.append((en, vi))
    elif os.path.exists(OPUS100_CSV_PATH):
        print(f"Loading OPUS-100 from: {OPUS100_CSV_PATH}")
        df = pd.read_csv(OPUS100_CSV_PATH).head(n_samples)
        for _, row in df.iterrows():
            en = row.get("en") or row.get("english")
            vi = row.get("vi") or row.get("vietnamese")
            if pd.notna(en) and pd.notna(vi):
                pairs.append((str(en), str(vi)))
    else:
        raise FileNotFoundError(f"Could not find {OPUS100_JSON_PATH} or {OPUS100_CSV_PATH}")
    return pairs

def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)

def encode(tok, user, ans):
    p = tok(chat(tok, user), add_special_tokens=False).input_ids
    full = tok(
        tok.apply_chat_template(
            [{"role": "user", "content": user}, {"role": "assistant", "content": ans}],
            tokenize=False
        ),
        add_special_tokens=False
    ).input_ids
    if len(full) > MAX_LEN:
        return None
    return full, [-100] * len(p) + full[len(p):]

def build_data(tok):
    raw_vi_pairs = load_local_opus100(N_VI)

    print(f"Loading English replay samples (excluding probe rows {PROBE_START}:{PROBE_END})...")
    ds_raw = load_dataset("tatsu-lab/alpaca", split="train")
    keep = [i for i in range(len(ds_raw)) if not (PROBE_START <= i < PROBE_END)]
    ds_en = ds_raw.select(keep).shuffle(seed=SEED).select(range(N_REPLAY))

    raw_items = []
    for i, (en, vi) in enumerate(raw_vi_pairs):
        if i % 2 == 0:
            raw_items.append((f"Translate the following text to Vietnamese. Output only the translation.\n\n{en}", vi, False))
        else:
            raw_items.append((f"Translate the following text to English. Output only the translation.\n\n{vi}", en, False))

    for a in ds_en:
        user_msg = a["instruction"] + (f"\n{a['input']}" if a["input"] else "")
        raw_items.append((user_msg, a["output"], True))

    random.shuffle(raw_items)

    data = []
    n_vi, n_en = 0, 0
    for u, a, is_rep in raw_items:
        enc = encode(tok, u, a)
        if enc:
            data.append(enc)
            if is_rep:
                n_en += 1
            else:
                n_vi += 1

    total = len(data)
    pct = (n_en / total) * 100.0 if total > 0 else 0.0
    print(f"\nSubmitted: {len(raw_items)} | Surviving: {total} | VI: {n_vi} | EN replay: {n_en} | Effective replay: {pct:.2f}%\n")
    return data, total, n_vi, n_en, pct

def collate(rows, pad):
    m = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), m), pad)
    lab = torch.full((len(rows), m), -100)
    mask = torch.zeros((len(rows), m), dtype=torch.long)
    for i, (x, y) in enumerate(rows):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return ids.cuda(), mask.cuda(), lab.cuda()

# ==========================================
# 2. TRAINING
# ==========================================
def train(model, tok):
    data, n_tot, n_vi, n_en, eff_pct = build_data(tok)
    total_steps = len(data) // (BS * ACCUM)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)

    ck = {max(1, int(total_steps * f)): int(f * 100) for f in (0.25, 0.50, 0.75, 1.0)}

    f_log = open(f"{OUT}/train_log.csv", "w", newline="")
    w_log = csv.writer(f_log)
    w_log.writerow(["step", "loss", "lr", "grad_norm"])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    idx = 0
    model.train()

    for step in range(1, total_steps + 1):
        tot_loss = 0.0
        lr = opt.param_groups[0]["lr"]

        for _ in range(ACCUM):
            ids, mask, lab = collate(data[idx:idx + BS], tok.pad_token_id)
            idx += BS
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
            (loss / ACCUM).backward()
            tot_loss += loss.item() / ACCUM

        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        w_log.writerow([step, f"{tot_loss:.6f}", f"{lr:.8f}", f"{float(gn):.6f}"])

        if step % 20 == 0:
            f_log.flush()
            print(f"Step {step}/{total_steps} | Loss: {tot_loss:.4f} | LR: {lr:.2e} | Grad Norm: {float(gn):.2f}")

        if step in ck:
            ckpt_dir = f"{OUT}/ckpt_{ck[step]}pct"
            print(f"--> Saving {ck[step]}% checkpoint to {ckpt_dir}")
            model.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)

    f_log.close()

    return {
        "arm_name": ARM_NAME,
        "dataset_pool_submitted": N_TOTAL,
        "surviving_total": n_tot,
        "surviving_vi": n_vi,
        "surviving_en_replay": n_en,
        "effective_replay_pct": round(eff_pct, 2),
        "total_training_steps": total_steps,
        "train_minutes": round((time.time() - t0) / 60.0, 2),
        "train_peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2),
    }

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Training {ARM_NAME} | model: {MODEL_DIR} | out: {OUT}")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.pad_token = tok.pad_token or tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    telemetry = train(model, tok)

    with open(os.path.join(OUT, "telemetry.json"), "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    print(f"\nDone. Effective replay: {telemetry['effective_replay_pct']}% | Adapters in {OUT}/ckpt_*pct")