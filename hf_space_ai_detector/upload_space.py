from pathlib import Path

from huggingface_hub import HfApi, whoami


SPACE_NAME = "ai-vs-real-image-detector"
FOLDER = Path(__file__).resolve().parent


def main():
    api = HfApi()
    user = whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"
    print(f"Creating or reusing Space: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )
    print(f"Uploading folder: {FOLDER}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(FOLDER),
        commit_message="Update demo with Tiny GenImage comparison",
        ignore_patterns=["__pycache__/*", "*.pyc"],
        delete_patterns=[
            "models/resnet18_best.pt",
            "models/efficientnet_b0_best.pt",
            "models/mobilenet_v3_small_best.pt",
            "models/vit_b_16_best.pt",
            "models/final_results.json",
        ],
    )
    print(f"Done: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
