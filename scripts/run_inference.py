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


def get_model(model_type: str, model_path: str | None = None):
    """모델 타입에 따라 적절한 모델 인스턴스를 반환한다."""
    if model_type == "mock":
        from src.models.mock_model import MockModel
        return MockModel()

    elif model_type == "qwen2_5_vl":
        try:
            from src.models.qwen_vl import QwenVLModel
            return QwenVLModel(model_path=model_path)
        except ImportError as e:
            console.print(f"[red]Qwen2.5-VL 로드 실패: {e}[/red]")
            console.print("[yellow]torch/transformers가 설치되어 있는지 확인하세요.[/yellow]")
            sys.exit(1)
    else:
        console.print(f"[red]알 수 없는 모델 타입: {model_type}[/red]")
        console.print("지원 모델: mock, qwen2_5_vl")
        sys.exit(1)


def run(args):
    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold cyan]킥보드 영상 상황 분석[/bold cyan]\n"
        f"영상: {video_path}\n"
        f"모델: {args.model}",
        title="[Kickboard Video Analyzer]"
    ))

    # 영상 존재 여부 확인 (mock 모드는 건너뜀)
    if args.model != "mock" and not video_path.exists():
        console.print(f"[red]영상 파일을 찾을 수 없습니다: {video_path}[/red]")
        sys.exit(1)

    # 모델 로드 및 분석
    console.print(f"[cyan]모델 로딩 중...[/cyan]")
    model = get_model(args.model, args.model_path)
    console.print(f"[green]모델 준비 완료: {model}[/green]")

    console.print(f"[cyan]영상 분석 중...[/cyan]")
    result = model.analyze(video_path)

    # 결과 출력
    result_dict = result.model_dump()
    # Enum → 문자열 변환
    result_dict["primary_situation"] = result.primary_situation.value
    result_dict["secondary_situations"] = [s.value for s in result.secondary_situations]
    result_dict["risk_level"] = result.risk_level.value

    console.print(Panel(JSON(json.dumps(result_dict, ensure_ascii=False, indent=2)),
                        title="[분석 결과]"))
    console.print(Panel(result.to_summary(), title="[요약]"))

    # 파일 저장
    stem = video_path.stem if video_path.exists() else "mock_output"
    json_path = output_dir / f"{stem}.json"
    summary_path = output_dir / f"{stem}_summary.md"

    json_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_path.write_text(
        f"# 분석 결과: {stem}\n\n{result.to_summary()}\n", encoding="utf-8"
    )

    console.print(f"\n[green]OK JSON 저장: {json_path}[/green]")
    console.print(f"[green]OK 요약 저장: {summary_path}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="킥보드 주행 영상 상황 분류 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="입력 영상 파일 경로")
    parser.add_argument(
        "--model",
        default="mock",
        choices=["mock", "qwen2_5_vl"],
        help="사용할 모델 (기본: mock)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="로컬 모델 경로 (qwen2_5_vl 사용 시 필요)",
    )
    parser.add_argument(
        "--output",
        default="data/outputs",
        help="결과 저장 디렉터리 (기본: data/outputs)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
