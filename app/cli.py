"""Command-line entry point: `python -m app.cli --help`.

Everyday use (no training needed, the adapter is in the repo):
    python -m app.cli tag samples/sample_contract.txt         -> .md + .json + .html next to it

Rebuilding everything (milestones 1-6 and the extras), for a run "top6" or "procurement":
    prepare -> tokens -> inject -> train -> merge -> evaluate -> calibrate
    label / real-eval   hand-label real Indian clauses and test on them
"""
import argparse


def _run_arg(p):
    p.add_argument("--run", default="top6", choices=["top6", "procurement"],
                   help="label set: top6 (course recipe) or procurement (14 types, the shipped adapter)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="legal-document-lora-tagger", description="Public sector clerks classify documents faster. This tool uses LoRA to fine-tune GPT-2 to tag legal clauses in public records, improving efficiency and reducing manual review time.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    p = sub.add_parser("prepare", help="milestone 1: build the balanced LEDGAR clause dataset")
    _run_arg(p)
    p.add_argument("--per-class", type=int, default=100, help="training examples per clause type")
    p.add_argument("--eval-per-class", type=int, default=50, help="validation and test examples per type")
    p = sub.add_parser("tokens", help="milestone 2: add the label tokens and show what changed")
    _run_arg(p)
    p = sub.add_parser("inject", help="milestone 3: inject LoRA and count trainable parameters")
    _run_arg(p)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p = sub.add_parser("train", help="milestone 4: fine-tune with LoRA, save and reload the adapter")
    _run_arg(p)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--epochs", type=float, default=6)
    p.add_argument("--grad-accum", type=int, default=1, help="split each batch of 8 to save memory (same maths)")
    p = sub.add_parser("real-eval", help="evaluate on your hand-labelled Indian clauses")
    _run_arg(p)
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Legal Document LoRA Tagger is ready. Open README.md for your milestones.")
    elif args.command == "prepare":
        from app.core import milestone_1
        milestone_1(args.run, args.per_class, args.eval_per_class)
    elif args.command == "tokens":
        from app.core import milestone_2
        milestone_2(args.run)
    elif args.command == "inject":
        from app.core import milestone_3
        milestone_3(args.run, r=args.r, alpha=args.alpha)
    elif args.command == "train":
        from app.core import milestone_4
        milestone_4(args.run, r=args.r, alpha=args.alpha, epochs=args.epochs, grad_accum=args.grad_accum)
    elif args.command == "real-eval":
        from app.pipeline import real_eval
        real_eval(args.run)


if __name__ == "__main__":
    main()
