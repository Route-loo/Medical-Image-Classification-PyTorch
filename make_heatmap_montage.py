"""
从 heatmap/ 文件夹挑选正确、错误样例，拼成一张对比图 gradcam_montage.png 用于 README。
用法：先跑完 grad_cam_batch.py 生成 heatmap/，再运行：
    python make_heatmap_montage.py
输出：项目根目录 gradcam_montage.png
"""
import os
import re
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# ===== 中文字体配置（解决中文标题显示为方框）=====
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
# =================================================

# ==================== 配置 ====================
HEAT_DIR = "./heatmap"
OUT_IMG = "./gradcam_montage.png"
PER_CORRECT = 2      # 每个类别挑几张预测正确的
PER_WRONG = 1        # 每个类别挑几张预测错误的（没有就自动跳过）
# ==============================================

# 文件名格式：{stem}_pred{正常/肺炎}_conf{0.xxx}.jpg
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
        print(f"未找到 {HEAT_DIR}，请先运行 grad_cam_batch.py")
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
    # 每个真实类别各取正确样例
    for true_folder in ("NORMAL", "PNEUMONIA"):
        pool = [s for s in samples["correct"] if s["true_folder"] == true_folder]
        picks += pool[:PER_CORRECT]
    # 错误样例（两类合计取 PER_WRONG*2，有几张取几张）
    picks += samples["wrong"][: PER_WRONG * 2]

    if not picks:
        print("没有可展示的样例图")
        return

    n = len(picks)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(3.2 * cols, 3.6 * rows))
    for i, s in enumerate(picks):
        img = Image.open(s["path"]).convert("RGB")
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        true_cn = "正常" if s["true_label"] == 0 else "肺炎"
        pred_cn = "正常" if s["pred"] == 0 else "肺炎"
        color = "green" if s["correct"] else "red"
        status = "✓正确" if s["correct"] else "✗错误"
        plt.title(f"真实:{true_cn} 预测:{pred_cn}\n{status} conf={s['conf']:.2f}",
                  color=color, fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=200, bbox_inches="tight")
    print(f"✅ 拼图已保存: {os.path.abspath(OUT_IMG)}（共 {n} 张样例）")


if __name__ == "__main__":
    main()
