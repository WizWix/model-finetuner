from __future__ import annotations

import os
from typing import Any


def upload_finetune_output(
    *,
    output_dir: str,
    repo_id: str,
    token: str,
    private: bool = True,
    create_repo_if_missing: bool = True,
) -> dict[str, Any]:
    """Upload finetune artifacts from output_dir to Hugging Face Hub."""
    if not repo_id.strip():
        raise ValueError("repo_id가 비어 있습니다.")
    if not token.strip():
        raise ValueError("HF token이 비어 있습니다.")

    lora_dir = os.path.join(output_dir, "lora")
    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(
            f"LoRA 디렉터리를 찾을 수 없습니다. expected={lora_dir}"
        )

    eval_dir = os.path.join(output_dir, "evaluation")
    readme_path = os.path.join(output_dir, "README.md")

    from huggingface_hub import create_repo, upload_file, upload_folder

    if create_repo_if_missing:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            private=private,
            exist_ok=True,
        )

    upload_folder(
        folder_path=lora_dir,
        path_in_repo=".",
        repo_id=repo_id,
        token=token,
    )

    eval_uploaded = False
    if os.path.isdir(eval_dir):
        upload_folder(
            folder_path=eval_dir,
            path_in_repo="evaluation",
            repo_id=repo_id,
            token=token,
        )
        eval_uploaded = True

    readme_uploaded = False
    if os.path.isfile(readme_path):
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        readme_uploaded = True

    return {
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/{repo_id}",
        "lora_dir": lora_dir,
        "eval_uploaded": eval_uploaded,
        "readme_uploaded": readme_uploaded,
        "create_repo_if_missing": create_repo_if_missing,
    }
