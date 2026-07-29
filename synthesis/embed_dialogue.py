import json
import numpy as np
from sentence_transformers import SentenceTransformer


def generate_shot_text_embeddings(
    input_json_path,
    output_npy_path,
    model_name="all-MiniLM-L6-v2"
):
    print("[STAGE 2] Generating shot text embeddings")

    # Load model
    model = SentenceTransformer(model_name)

    # Load shots
    with open(input_json_path, "r", encoding="utf-8") as f:
        shots = json.load(f)

    embeddings = []

    for shot in shots:
        text = shot.get("dialogue", "").strip()
        if text == "":
            emb = np.zeros(model.get_sentence_embedding_dimension())
        else:
            emb = model.encode(text)
        embeddings.append(emb)

    embeddings = np.array(embeddings)
    np.save(output_npy_path, embeddings)

    print(f"[DONE] Embeddings saved → {output_npy_path}")
    print(f"[INFO] Shape: {embeddings.shape}")

    return output_npy_path
