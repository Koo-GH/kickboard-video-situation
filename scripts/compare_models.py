import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def load_json(filepath: Path) -> dict:
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def format_evidence(evidence_list: list) -> str:
    if not evidence_list:
        return "-"
    return "\n".join(f"- {e}" for e in evidence_list)

def create_markdown_report(data: dict, name1: str, name2: str) -> str:
    lines = [
        f"# Model Comparison Report\n",
        f"**Model 1:** `{name1}`\n",
        f"**Model 2:** `{name2}`\n\n",
    ]
    
    for video_name, results in data.items():
        res1 = results.get('model1')
        res2 = results.get('model2')
        
        lines.append(f"## {video_name}\n")
        lines.append("| 항목 | Model 1 | Model 2 |")
        lines.append("|---|---|---|")
        
        if res1 and res2:
            lines.append(f"| **상황 (Situation)** | `{res1.get('primary_situation', 'N/A')}` | `{res2.get('primary_situation', 'N/A')}` |")
            lines.append(f"| **위험도 (Risk)** | {res1.get('risk_level', 'N/A')} | {res2.get('risk_level', 'N/A')} |")
            lines.append(f"| **신뢰도 (Confidence)** | {res1.get('confidence', 'N/A')} | {res2.get('confidence', 'N/A')} |")
            lines.append(f"| **근거 (Evidence)** | {format_evidence(res1.get('evidence', []))} | {format_evidence(res2.get('evidence', []))} |")
            lines.append(f"| **사람 검토 (Review)** | {'Yes' if res1.get('needs_human_review') else 'No'} | {'Yes' if res2.get('needs_human_review') else 'No'} |")
        elif res1:
            lines.append(f"| 상태 | 성공 | **결과 없음** |")
        elif res2:
            lines.append(f"| 상태 | **결과 없음** | 성공 |")
        else:
            lines.append(f"| 상태 | **결과 없음** | **결과 없음** |")
            
        lines.append("\n")
        
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="두 모델의 분석 결과를 나란히 비교합니다.")
    parser.add_argument("--dir1", required=True, help="첫 번째 모델의 결과가 저장된 디렉터리")
    parser.add_argument("--name1", required=True, help="첫 번째 모델의 이름 (출력용)")
    parser.add_argument("--dir2", required=True, help="두 번째 모델의 결과가 저장된 디렉터리")
    parser.add_argument("--name2", required=True, help="두 번째 모델의 이름 (출력용)")
    parser.add_argument("--output", default="data/outputs/comparison_report.md", help="비교 결과를 저장할 파일 경로")
    
    args = parser.parse_args()
    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)
    
    if not dir1.exists() or not dir2.exists():
        console.print("[red]입력된 디렉터리를 찾을 수 없습니다.[/red]")
        return

    # 두 디렉터리 내의 _output.json 파일명 수집
    files1 = {f.name for f in dir1.glob("*_output.json")}
    files2 = {f.name for f in dir2.glob("*_output.json")}
    all_files = sorted(list(files1.union(files2)))
    
    if not all_files:
        console.print("[yellow]분석 결과 파일이 없습니다.[/yellow]")
        return

    comparison_data = {}
    for filename in all_files:
        video_name = filename.replace("_output.json", ".mp4")
        res1 = load_json(dir1 / filename) if filename in files1 else None
        res2 = load_json(dir2 / filename) if filename in files2 else None
        
        comparison_data[video_name] = {
            "model1": res1,
            "model2": res2
        }

    # 콘솔 출력 (테이블 형태)
    console.print(f"\n[bold cyan]▶ {args.name1} vs {args.name2} 비교 결과[/bold cyan]")
    
    for video_name, results in comparison_data.items():
        res1 = results['model1']
        res2 = results['model2']
        
        table = Table(title=f"비디오: {video_name}", show_header=True, header_style="bold magenta")
        table.add_column("항목", style="cyan", width=15)
        table.add_column(args.name1, width=40)
        table.add_column(args.name2, width=40)
        
        if res1 and res2:
            table.add_row("Situation", res1.get('primary_situation', '-'), res2.get('primary_situation', '-'))
            table.add_row("Risk Level", res1.get('risk_level', '-'), res2.get('risk_level', '-'))
            table.add_row("Confidence", str(res1.get('confidence', '-')), str(res2.get('confidence', '-')))
            table.add_row("Evidence", format_evidence(res1.get('evidence', [])), format_evidence(res2.get('evidence', [])))
        elif res1:
            table.add_row("Status", "성공", "[red]결과 없음[/red]")
        elif res2:
            table.add_row("Status", "[red]결과 없음[/red]", "성공")
            
        console.print(table)
        console.print("")

    # 마크다운 저장
    md_content = create_markdown_report(comparison_data, args.name1, args.name2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    console.print(f"[bold green]비교 리포트가 성공적으로 저장되었습니다: {output_path}[/bold green]")

if __name__ == "__main__":
    main()
