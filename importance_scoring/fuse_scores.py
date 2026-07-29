import json
import numpy as np


def normalize_dict(d):
    vals = np.array(list(d.values()), dtype=np.float32)
    mn, mx = vals.min(), vals.max()
    return {
        k: float((v - mn) / (mx - mn + 1e-8))
        for k, v in d.items()
    }


def fuse_scores(
    cached_scores_path,
    output_json_path,
    alpha=0.5,
    beta=0.3,
    gamma=0.2,
    scale_to_100=True
):
    """
    Stage 9: Fuse modality scores and rank clips.

    Output JSON preserves ranking order explicitly.
    """

    with open(cached_scores_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 🔹 collect raw scores
    visual_raw = {k: v["visual"] for k, v in data.items()}
    audio_raw  = {k: v["audio"]  for k, v in data.items()}
    text_raw   = {k: v["text"]   for k, v in data.items()}

    # 🔹 normalize to [0,1]
    visual = normalize_dict(visual_raw)
    audio  = normalize_dict(audio_raw)
    text   = normalize_dict(text_raw)

    fused_list = []

    for clip in data.keys():
        score = (
            alpha * visual[clip] +
            beta  * audio[clip] +
            gamma * text[clip]
        )

        if scale_to_100:
            score *= 100.0

        fused_list.append({
            "clip": clip,
            "importance": round(score, 2),
            "visual": round(visual[clip], 4),
            "audio": round(audio[clip], 4),
            "text": round(text[clip], 4),
            "transcript": data[clip].get("transcript", "")
        })

    # 🔹 sort by importance (descending)
    fused_list = sorted(
        fused_list,
        key=lambda x: x["importance"],
        reverse=True
    )

    # 🔹 assign ranks
    for idx, item in enumerate(fused_list, start=1):
        item["rank"] = idx

    output = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "ranked_clips": fused_list
    }

    # 🔹 save JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n Ranked fusion saved → {output_json_path}\n")

    # =========================
    # PRINT RANKING TO TERMINAL
    # =========================

    print(" FINAL RANKING (Highest → Lowest)")
    print("-" * 50)
    print(f"{'Rank':<6} {'Clip':<15} {'Importance':>10}")
    print("-" * 50)

    for item in fused_list:
        print(
            f"{item['rank']:<6} "
            f"{item['clip']:<15} "
            f"{item['importance']:>10.2f}"
        )

    print("-" * 50)

    return output
