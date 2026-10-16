"""
Pick correct / wrong samples from heatmap/ and build a comparison sheet gradcam_montage.png.
Usage: run grad_cam_batch.py first, then:
    python make_heatmap_montage.py
Output: gradcam_montage.png in project root.
All titles are in English to avoid CJK font issues.
"""
import os
import re
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.rcParams["axes.unicode_minus"] = False

# ==================== Config ====================
HEAT_DIR = "./heatmap"
OUT_IMG = "./gradcam_montage.png"
PER_CORRECT = 2      # correct samples per true class
PER_WRONG = 1        # wrong samples per true class (skipped if none)
# ==============================================

# filename: {stem}_pred{正常|肺炎}_conf{0.xxx}.jpg
PATTERN = re.compile(r"_pred(正常|肺炎)_conf([0-9.]+)\.jpg$")


def parse_filename(fname, true_folder):
    m = PATTERN.search(fname)
    if not m:
        return None
    pred_cn, conf = m.group(1), float(m.group(2))
    pred_label = 0 if pred_cn == "正常" else 1
    true_label = 0 if true_folder.upper() == "NORMAL" else 1
    return {
        "pred": pred_label,
        "conf": conf,
        "correct": pred_label == true_label,
        "true_label": true_label,
    }


def collect():
    samples = {"correct": [], "wrong": []}
    if not os.path.isdir(HEAT_DIR):
        print(f"Folder {HEAT_DIR} not found; run grad_cam_batch.py first")
        return samples

    for folder in os.listdir(HEAT_DIR):
        folder_path = os.path.join(HEAT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            info = parse_filename(fname, folder)
            if info is None:
                continue
            info["path"] = os.path.join(folder_path, fname)
            info["true_folder"] = folder
            bucket = "correct" if info["correct"] else "wrong"
            samples[bucket].append(info)
    return samples


def main():
    samples = collect()
    picks = []
    for true_folder in ("NORMAL", "PNEUMONIA"):
        pool = [s for s in samples["correct"] if s["true_folder"] == true_folder]
        picks += pool[:PER_CORRECT]
    picks += samples["wrong"][: PER_WRONG * 2]

    if not picks:
        print("No samples to display")
        return

    n = len(picks)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(3.2 * cols, 3.6 * rows))
    for i, s in enumerate(picks):
        img = Image.open(s["path"]).convert("RGB")
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        true_en = "Normal" if s["true_label"] == 0 else "Pneumonia"
        pred_en = "Normal" if s["pred"] == 0 else "Pneumonia"
        color = "green" if s["correct"] else "red"
        status = "OK" if s["correct"] else "Wrong"
        plt.title(f"True:{true_en} Pred:{pred_en}\n{status} conf={s['conf']:.2f}",
                  color=color, fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=200, bbox_inches="tight")
    print(f"Saved: {os.path.abspath(OUT_IMG)} ({n} samples)")


if __name__ == "__main__":
    main()
