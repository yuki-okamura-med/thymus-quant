from __future__ import annotations

import argparse
import json

from .core import analyze


def cmd_analyze(args: argparse.Namespace) -> int:
    masks = args.ensemble if args.ensemble else [args.trq_mask]
    result = analyze(
        args.ct,
        trq_mask=masks,
        study_id=args.study_id,
        segmenter=args.segmenter,
        protocol=args.protocol,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thymus-quant")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Compute TRQ quantitative metrics")
    a.add_argument("--ct", required=True)
    a.add_argument("--trq-mask", required=True)
    a.add_argument("--ensemble", nargs="*", help="Optional additional masks for ensemble QC")
    a.add_argument("--study-id", default=None)
    a.add_argument("--segmenter", default="provided_mask")
    a.add_argument("--protocol", default="okamura2025", choices=["okamura2025", "okamura2025_plus_ptt"])
    a.set_defaults(func=cmd_analyze)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
