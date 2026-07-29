import json


def refine_scene_boundaries(
    scenes_with_timestamps_path,
    sentence_segments_path,
    output_json_path
):
    print("[STAGE 6] Refining scene boundaries using sentence alignment")

    # Load data
    with open(scenes_with_timestamps_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    with open(sentence_segments_path, "r", encoding="utf-8") as f:
        whisper = json.load(f)

    segments = whisper["segments"]

    def snap_end_time(t):
        """Snap scene end to the next sentence end."""
        for seg in segments:
            if seg["end"] >= t:
                return seg["end"]
        return t

    def snap_start_time(t):
        """Snap scene start to the previous sentence start."""
        for seg in reversed(segments):
            if seg["start"] <= t:
                return seg["start"]
        return t

    refined = []

    for scene in scenes:
        start = snap_start_time(scene["start_time"])
        end = snap_end_time(scene["end_time"])

        refined.append({
            **scene,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration": round(end - start, 3)
        })

    # Save
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(refined, f, indent=2)

    print(f"[DONE] Refined {len(refined)} scenes → {output_json_path}")
    return output_json_path
