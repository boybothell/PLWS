#!/usr/bin/env python3
import json
import re
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if len(y) < 2 or y.min() == y.max(): return float('nan')
    return float(roc_auc_score(y, s))

def extract_features(row):
    text = str(row.get("reasoning_prefix", ""))
    ans = str(row.get("answer", ""))
    
    # 1. Length features
    char_len = len(text)
    ans_len = len(ans)
    
    # 2. Hesitation / Self-correction
    hesitation_words = ["wait", "actually", "however", "let me re", "wrong", "mistake", "hold on", "hmm"]
    hesitation_count = sum(text.lower().count(w) for w in hesitation_words)
    hesitation_density = hesitation_count / (char_len + 1) * 1000
    
    # 3. Certainty / Flow words
    certainty_words = ["therefore", "thus", "clearly", "obviously", "hence", "so the answer is"]
    certainty_count = sum(text.lower().count(w) for w in certainty_words)
    
    # 4. Mathematical density
    eq_count = text.count("=")
    latex_count = text.count("\\")
    frac_count = text.count("\\frac") + text.count("\\dfrac")
    math_density = (eq_count + latex_count) / (char_len + 1) * 1000
    
    # 5. Answer characteristics
    ans_has_frac = 1 if "\\" in ans or "/" in ans else 0
    ans_is_int = 1 if re.match(r"^-?\d+$", ans.strip()) else 0
    
    # 6. Structural
    newlines = text.count("\n\n")
    avg_paragraph_len = char_len / (newlines + 1)
    
    return {
        "step": row.get("decision_step", 0),
        "char_len": char_len,
        "ans_len": ans_len,
        "hesitation_count": hesitation_count,
        "hesitation_density": hesitation_density,
        "certainty_count": certainty_count,
        "eq_count": eq_count,
        "frac_count": frac_count,
        "math_density": math_density,
        "ans_has_frac": ans_has_frac,
        "ans_is_int": ans_is_int,
        "avg_paragraph_len": avg_paragraph_len
    }

for ds in ["math-500", "olympiadbench"]:
    print(f"\n========== {ds} ==========")
    path = Path(f'/mnt/d/lsj/visual-latent-tts/repos/attn-early-exit/results/confcal_judge/v2/info_signals/{ds}_solver_s0.jsonl')
    path2 = Path(f'/mnt/d/lsj/visual-latent-tts/repos/attn-early-exit/results/confcal_judge/v2/info_signals/{ds}_solver_s1.jsonl')
    
    rows = []
    if path.exists(): rows.extend([json.loads(x) for x in path.read_text().splitlines() if x.strip()])
    if path2.exists(): rows.extend([json.loads(x) for x in path2.read_text().splitlines() if x.strip()])
    
    y = np.array([r["is_g"] for r in rows])
    print(f"Total n={len(rows)}, G={sum(y)}, nonG={len(y)-sum(y)}")
    
    features = [extract_features(r) for r in rows]
    feature_names = list(features[0].keys())
    
    print(f"{'Feature':<20} | {'AUROC':<6} | {'G_mean':<8} | {'nonG_mean':<8}")
    print("-" * 50)
    for name in feature_names:
        s = np.array([f[name] for f in features])
        roc = auc(y, s)
        g_mean = s[y==1].mean()
        n_mean = s[y==0].mean()
        
        # If AUROC < 0.5, we invert it so we can easily spot strong signals
        display_roc = roc if roc > 0.5 else 1 - roc
        dir_sign = "+" if roc > 0.5 else "-"
        
        if display_roc > 0.55:  # Only show somewhat relevant features
            print(f"{name:<20} | {display_roc:.3f}{dir_sign} | {g_mean:<8.2f} | {n_mean:<8.2f}")
