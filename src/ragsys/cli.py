"""Command line: ask a question, or evaluate a configuration."""

from __future__ import annotations

import argparse

from ragsys.access import USERS, user_for
from ragsys.configs import CONFIGS
from ragsys.evals import (
    by_type,
    evaluate,
    load_questions,
    split,
    summarize,
)
from ragsys.pipeline import Pipeline


def ask(args: argparse.Namespace) -> None:
    pipe = Pipeline(CONFIGS[args.config])
    result = pipe.run(
        args.question, user_for(args.user, args.tenant)
    )
    print(result.answer.text)
    if args.show:
        for c in result.context:
            print(f"  context: {c.id}")


def run_eval(args: argparse.Namespace) -> None:
    pipe = Pipeline(CONFIGS[args.config])
    questions = load_questions()
    if args.split != "all":
        questions = split(questions, args.split)
    grades = evaluate(pipe, questions)
    for key, value in summarize(grades).items():
        print(f"{key:15} {value}")
    for kind, (ok, n) in by_type(grades).items():
        print(f"  {kind:13} {ok}/{n}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragsys")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument(
        "--config", default="production", choices=CONFIGS
    )
    a.add_argument("--user", default="customer", choices=USERS)
    a.add_argument("--tenant", default="")
    a.add_argument("--show", action="store_true")
    a.set_defaults(fn=ask)
    e = sub.add_parser("eval", help="grade a configuration")
    e.add_argument(
        "--config", default="production", choices=CONFIGS
    )
    e.add_argument(
        "--split", default="all", choices=["all", "dev", "test"]
    )
    e.set_defaults(fn=run_eval)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
