"""
CLI 추론 스크립트
사용 예시:
  # mock 모드 (GPU 없이)
  python scripts/run_inference.py --video data/samples/sample.mp4 --model mock

  # Qwen2.5-VL (로컬 경로)
  python scripts/run_inference.py --video data/samples/sample.mp4 \
    --model qwen2_5_vl --model-path ./models/Qwen2.5-VL-3B-Instruct
"""
import sys
import json
import argparse
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

console = Console()


def get_model(model_type: str, model_path: str | None = None, n_frames: int = 16):
    """모델 타입에 따라 적절한 모델 인스턴스를 반환한다."""
    if model_type == "mock":
        from src.models.mock_model import MockModel
        return MockModel()

    elif model_type == "qwen2_5_vl":
        try:
            from src.models.qwen_vl import QwenVLModel
            return QwenVLModel(model_path=model_path, n_frames=n_frames)
        except ImportError as e:
            console.print(f"[red]Qwen2.5-VL 로드 실패: {e}[/red]")
            console.print("[yellow]torch/transformers가 설치되어 있는지 확인하세요.[/yellow]")
            sys.exit(1)
    elif model_type == "videollama3":
        try:
            from src.models.videollama import VideoLLaMA3Model
            return VideoLLaMA3Model(model_path=model_path, n_frames=n_frames)
        except ImportError as e:
            console.print(f"[red]VideoLLaMA3 로드 실패: {e}[/red]")
            console.print("[yellow]torch/transformers가 설치되어 있는지 확인하세요.[/yellow]")
            sys.exit(1)
    else:
        console.print(f"[red]알 수 없는 모델 타입: {model_type}[/red]")
        console.print("지원 모델: mock, qwen2_5_vl, videollama3")
        sys.exit(1)


def analyze_video(model, video_path: Path, output_dir: Path):
    import gc
    import torch
    console.print(f"\n[bold cyan]▶ 영상 분석 중: {video_path.name}[/bold cyan]")
    result = model.analyze(video_path)
    # 영상 분석 후 GPU 메모리 정리 (연속 분석 시 OOM 방지)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 결과 저장
    json_path = output_dir / f"{video_path.stem}_output.json"
    summary_path = output_dir / f"{video_path.stem}_output_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(result.to_summary())

    # 결과 출력 (터미널)
    result_dict = result.model_dump()
    # Enum → 문자열 변환
    result_dict["primary_situation"] = result.primary_situation.value
    result_dict["secondary_situations"] = [s.value for s in result.secondary_situations]
    result_dict["risk_level"] = result.risk_level.value

    try:
        console.print(Panel(JSON(json.dumps(result_dict, ensure_ascii=False, indent=2)),
                            title=f"[분석 결과: {video_path.name}]"))
    except UnicodeEncodeError:
        console.print("[warning]터미널 인코딩 문제로 JSON 전체 출력을 생략합니다.[/warning]")
        
    try:
        console.print(Panel(result.to_summary(), title=f"[요약: {video_path.name}]"))
    except UnicodeEncodeError:
        pass

    console.print(f"[green]OK JSON 저장됨:[/green] {json_path}")
    console.print(f"[green]OK 요약 저장됨:[/green] {summary_path}")


def run(args):
    input_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model != "mock" and not input_path.exists():
        console.print(f"[red]경로를 찾을 수 없습니다: {input_path}[/red]")
        sys.exit(1)

    if input_path.is_dir():
        video_paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])
        if not video_paths:
            console.print(f"[red]해당 폴더에 영상 파일(.mp4 등)이 없습니다: {input_path}[/red]")
            sys.exit(1)
    else:
        video_paths = [input_path]

    console.print(Panel(
        f"[bold cyan]킥보드 영상 상황 분석[/bold cyan]\n"
        f"대상: {input_path} (총 {len(video_paths)}개 영상)\n"
        f"모델: {args.model}",
        title="[Kickboard Video Analyzer]"
    ))

    # 모델 로드
    console.print(f"[cyan]모델 로딩 중...[/cyan]")
    model = get_model(args.model, args.model_path, args.n_frames)
    console.print(f"[green]모델 준비 완료: {model}[/green]")

    # 순차 분석
    for v_path in video_paths:
        analyze_video(model, v_path, output_dir)

    console.print(f"\n[bold green]모든 영상({len(video_paths)}개) 분석 및 저장이 완료되었습니다![/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="킥보드 주행 영상 상황 분류 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="입력 영상 파일 경로")
    parser.add_argument(
        "--model",
        default="mock",
        choices=["mock", "qwen2_5_vl", "videollama3"],
        help="사용할 모델 (기본: mock)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="로컬 모델 경로 (qwen2_5_vl / videollama3 사용 시 필요)",
    )
    parser.add_argument(
        "--output",
        default="data/outputs",
        help="결과 저장 디렉터리 (기본: data/outputs)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=16,
        help="영상에서 샘플링할 프레임 수 (기본: 16)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
