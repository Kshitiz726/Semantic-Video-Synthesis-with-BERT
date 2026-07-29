"""
Main Pipeline Controller
Multimodal AI System for Clip Extraction from Long Videos
"""

import argparse
from pathlib import Path


# --------------------------------------------------
# Stage 1: Shot Detection + ASR
# --------------------------------------------------
def stage_1_shot_extraction(video_path, work_dir):
    from preprocessing.TransNetV2.transnet_shot_extractor import (
        extract_shots_with_dialogue
    )

    output_json = (
        work_dir
        / "preprocessing"
        / "TransNetV2"
        / "lgss_input"
        / "shots_with_dialogue.json"
    )

    extract_shots_with_dialogue(
        video_path=str(video_path),
        output_json_path=str(output_json),
        debug_dir=str(work_dir / "debug")
    )


# --------------------------------------------------
# Stage 2: Dialogue → Text Embeddings
# --------------------------------------------------
def stage_2_dialogue_embedding(video_path, work_dir):
    from synthesis.embed_dialogue import generate_shot_text_embeddings

    input_json = (
        work_dir
        / "preprocessing"
        / "TransNetV2"
        / "lgss_input"
        / "shots_with_dialogue.json"
    )

    output_npy = (
        work_dir
        / "synthesis"
        / "shot_text_embeddings.npy"
    )

    generate_shot_text_embeddings(
        input_json_path=str(input_json),
        output_npy_path=str(output_npy)
    )


# --------------------------------------------------
# Stage 3: Raw Scene Segmentation (LGSS)
# --------------------------------------------------
def stage_3_scene_segmentation(video_path, work_dir):
    from scene_segmentation.SceneSeg.run_sceneseg import run_scene_segmentation

    embeddings_path = (
        work_dir
        / "synthesis"
        / "shot_text_embeddings.npy"
    )

    output_scene_json = (
        work_dir
        / "scene_segmentation"
        / "SceneSeg"
        / "scene_boundaries.json"
    )

    run_scene_segmentation(
        shot_embeddings_path=str(embeddings_path),
        output_scene_json_path=str(output_scene_json),
        threshold=0.55
    )


# --------------------------------------------------
# Stage 4: Scene Merging
# --------------------------------------------------
def stage_4_scene_merging(video_path, work_dir):
    from scene_segmentation.SceneSeg.merge_scenes import merge_scenes

    scene_boundaries = (
        work_dir
        / "scene_segmentation"
        / "SceneSeg"
        / "scene_boundaries.json"
    )

    embeddings_path = (
        work_dir
        / "synthesis"
        / "shot_text_embeddings.npy"
    )

    output_final_scenes = (
        work_dir
        / "scene_segmentation"
        / "SceneSeg"
        / "final_scene_boundaries.json"
    )

    merge_scenes(
        scene_boundaries_path=str(scene_boundaries),
        shot_embeddings_path=str(embeddings_path),
        output_json_path=str(output_final_scenes)
    )


# --------------------------------------------------
# Stage 5: Scene → Timestamps
# --------------------------------------------------
def stage_5_scene_to_timestamps(video_path, work_dir):
    from scene_segmentation.SceneSeg.scenes_to_timestamps import scenes_to_timestamps

    final_scene_boundaries = (
        work_dir
        / "scene_segmentation"
        / "SceneSeg"
        / "final_scene_boundaries.json"
    )

    shots_json = (
        work_dir
        / "preprocessing"
        / "TransNetV2"
        / "lgss_input"
        / "shots_with_dialogue.json"
    )

    output_json = (
        work_dir
        / "scene_segmentation"
        / "SceneSeg"
        / "final_scenes_with_timestamps.json"
    )

    scenes_to_timestamps(
        final_scene_boundaries_path=str(final_scene_boundaries),
        shots_with_dialogue_path=str(shots_json),
        output_json_path=str(output_json)
    )


# --------------------------------------------------
# Stage 6: Scene Boundary Refinement
# --------------------------------------------------
# --------------------------------------------------
# Stage 5.5: Build Sentence Segments
# --------------------------------------------------
def stage_5_5_build_sentence_segments(video_path, work_dir):
    import json
    
    input_json = (
        work_dir
        / "preprocessing"
        / "TransNetV2"
        / "lgss_input"
        / "shots_with_dialogue.json"
    )
    
    output_json = (
        work_dir
        / "preprocessing"
        / "TransNetV2"
        / "lgss_input"
        / "sentence_segments.json"
    )

    with open(input_json, "r", encoding="utf-8") as f:
        shots = json.load(f)

    segments = []
    current_text = ""
    current_start = None
    current_end = None

    def normalize(text):
        return text.strip().lower()

    for shot in shots:
        text = shot.get("dialogue", "").strip()
        start = shot["start_time"]
        end = shot["end_time"]

        if not text:
            continue

        if current_text == "":
            current_text = text
            current_start = start
            current_end = end
            continue

        # Logic to merge fragmented sub-sentences or repeated ASR outputs
        if normalize(text) == normalize(current_text):
            current_end = end
            continue

        if normalize(text).startswith(normalize(current_text)):
            current_text = text
            current_end = end
            continue

        if normalize(current_text).startswith(normalize(text)):
            current_end = end
            continue

        segments.append({
            "start": round(current_start, 3),
            "end": round(current_end, 3),
            "text": current_text
        })

        current_text = text
        current_start = start
        current_end = end

    if current_text:
        segments.append({
            "start": round(current_start, 3),
            "end": round(current_end, 3),
            "text": current_text
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, indent=2, ensure_ascii=False)
    
    print(f"[STAGE 5.5] Generated sentence segments → {output_json}")


# --------------------------------------------------
# Stage 6: Scene Boundary Refinement
# --------------------------------------------------
def stage_6_scene_refinement(video_path, work_dir):
    from refine_scenes.refine_scene_boundaries import refine_scene_boundaries

    scenes_with_timestamps = (
        work_dir
        / "scene_segmentation"
        / "SceneSeg"
        / "final_scenes_with_timestamps.json"
    )

    sentence_segments = (
        work_dir
        / "preprocessing"
        / "TransNetV2"
        / "lgss_input"
        / "sentence_segments.json"
    )

    output_refined_scenes = (
        work_dir
        / "refine_scenes"
        / "refined_scenes_with_timestamps.json"
    )

    refine_scene_boundaries(
        scenes_with_timestamps_path=str(scenes_with_timestamps),
        sentence_segments_path=str(sentence_segments),
        output_json_path=str(output_refined_scenes)
    )


# --------------------------------------------------
# Stage 7: Scene Clip Extraction
# --------------------------------------------------
def stage_7_clip_extraction(video_path, work_dir):
    from scene_segmentation.SceneSeg.extract_scene_clips import extract_scene_clips

    refined_scenes = (
        work_dir
        / "refine_scenes"
        / "refined_scenes_with_timestamps.json"
    )

    output_dir = (
        work_dir
        / "importance_scoring"
        / "clips"
    )

    extract_scene_clips(
        video_path=str(video_path),
        refined_scenes_path=str(refined_scenes),
        output_dir=str(output_dir)
    )


# --------------------------------------------------
# Stage 8: Raw Importance Scoring
# --------------------------------------------------
def stage_8_importance_scoring(video_path, work_dir):
    from importance_scoring.extract_and_cache_scores import extract_and_cache_scores

    clips_dir = work_dir / "importance_scoring" / "clips"
    output_json = work_dir / "importance_scoring" / "cached_scores.json"
    
    # Model is static, located in the codebase, not the session work_dir
    project_root = Path(__file__).parent
    model_path = project_root / "importance_scoring" / "visual_importance_model.pt"

    extract_and_cache_scores(
        clips_dir=str(clips_dir),
        output_json_path=str(output_json),
        visual_model_path=str(model_path),
        force_recompute=False
    )


# --------------------------------------------------
# Stage 9: Score Fusion
# --------------------------------------------------
def stage_9_score_fusion(video_path, work_dir):
    from importance_scoring.fuse_scores import fuse_scores

    cached_scores = (
        work_dir
        / "importance_scoring"
        / "cached_scores.json"
    )

    fused_output = (
        work_dir
        / "importance_scoring"
        / "fused_scores.json"
    )

    fuse_scores(
        cached_scores_path=str(cached_scores),
        output_json_path=str(fused_output),
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        scale_to_100=True
    )


# --------------------------------------------------
# Stage 10: Visualization
# --------------------------------------------------
def stage_10_visualization(video_path, work_dir):
    from importance_scoring.visualize_scores import visualize_scores

    fused_scores = (
        work_dir
        / "importance_scoring"
        / "fused_scores.json"
    )

    save_path = (
        work_dir
        / "importance_scoring"
        / "visualization.png"
    )

    visualize_scores(
        fused_scores_path=str(fused_scores),
        show=False,
        save_path=str(save_path)
    )



# --------------------------------------------------
# Main Runner
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Multimodal AI System for Clip Extraction from Long Videos")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--from_stage", type=int, default=1)
    parser.add_argument("--to_stage", type=int, default=10)

    args = parser.parse_args()

    video_path = Path(args.video)
    work_dir = Path(args.work_dir)

    stages = [
        stage_1_shot_extraction,
        stage_2_dialogue_embedding,
        stage_3_scene_segmentation,
        stage_4_scene_merging,
        stage_5_scene_to_timestamps,
        stage_5_5_build_sentence_segments,
        stage_6_scene_refinement,
        stage_7_clip_extraction,
        stage_8_importance_scoring,
        stage_9_score_fusion,
        stage_10_visualization,
    ]

    for idx, stage in enumerate(stages, start=1):
        if args.from_stage <= idx <= args.to_stage:
            print(f"\n▶ Running Stage {idx}: {stage.__name__}")
            stage(video_path, work_dir)

    print("\n Pipeline completed successfully.")


if __name__ == "__main__":
    main()
