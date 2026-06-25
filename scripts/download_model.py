"""
모델 다운로드 헬퍼 스크립트 (Windows에서 실행)
HuggingFace에서 모델을 다운로드하여 로컬에 저장한다.

사용 예시:
  python scripts/download_model.py --model Qwen/Qwen2.5-VL-3B-Instruct
  python scripts/download_model.py --model Qwen/Qwen2.5-VL-7B-Instruct --output ./models
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress

console = Console()


def download_model(model_id: str, output_dir: Path):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[red]huggingface_hub가 설치되지 않았습니다.[/red]")
        console.print("pip install huggingface_hub")
        sys.exit(1)

    local_dir = output_dir / model_id.split("/")[-1]
    local_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]다운로드 시작: {model_id}[/cyan]")
    console.print(f"저장 위치: {local_dir}")
    console.print("[yellow]모델 크기에 따라 수 분~수십 분이 소요될 수 있습니다.[/yellow]\n")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # 불필요한 파일 제외
    )

    console.print(f"\n[green]OK 다운로드 완료: {local_dir}[/green]")
    console.print(
        f"\n서버 전송 명령 (Git Bash / WSL):\n"
        f"  scp -r {local_dir} cvlab308@10.10.15.112:/workspace/models/"
    )


def main():
    parser = argparse.ArgumentParser(description="HuggingFace 모델 다운로드")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace 모델 ID (기본: Qwen/Qwen2.5-VL-3B-Instruct)",
    )
    parser.add_argument(
        "--output",
        default="./models",
        help="저장 디렉터리 (기본: ./models)",
    )
    args = parser.parse_args()
    download_model(args.model, Path(args.output))


if __name__ == "__main__":
    main()
