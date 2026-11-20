"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="legal-document-lora-tagger", description="Public sector clerks classify documents faster. This tool uses LoRA to fine-tune GPT-2 to tag legal clauses in public records, improving efficiency and reducing manual review time.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    p = sub.add_parser("prepare", help="milestone 1: build the balanced LEDGAR clause dataset")
    p.add_argument("--top-k", type=int, default=6, help="number of clause types (most common first)")
    p.add_argument("--per-class", type=int, default=100, help="training examples per clause type")
    p.add_argument("--eval-per-class", type=int, default=50, help="validation and test examples per type")
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Legal Document LoRA Tagger is ready. Open README.md for your milestones.")
    elif args.command == "prepare":
        from app.core import milestone_1
        milestone_1(args.top_k, args.per_class, args.eval_per_class)


if __name__ == "__main__":
    main()
