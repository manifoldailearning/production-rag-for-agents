"""Chapter 11: follow-up questions and long-term memory."""

from __future__ import annotations

from ragsys.configs import PRODUCTION
from ragsys.memory import (
    DIALOGUES,
    SESSIONS,
    MemoryStore,
    concatenate,
    contextualize,
    personalize,
)
from ragsys.pipeline import Pipeline


def follow_ups(pipe: Pipeline, mode: str) -> int:
    correct = 0
    for dialogue in DIALOGUES:
        history: list[str] = []
        for turn, (question, expected) in enumerate(dialogue):
            query = question
            if turn and mode == "concatenate":
                query = concatenate(question, history)
            elif turn and mode == "rewrite":
                query = contextualize(question, history)
            text = pipe.run(query).answer.text.lower()
            correct += turn > 0 and expected.lower() in text
            history.append(
                query if mode == "rewrite" else question
            )
    return correct


def sessions(pipe: Pipeline, policy: str | None) -> int:
    correct = 0
    for said, question, expected in SESSIONS:
        store = MemoryStore(policy or "slots")
        for turn, text in enumerate(said if policy else ()):
            store.observe("u1", text, turn)
        query = personalize(question, store.recall("u1", "plan"))
        if expected is None:
            correct += query == question
            continue
        text = pipe.run(query).answer.text.lower()
        correct += expected.lower() in text
    return correct


def main() -> None:
    pipe = Pipeline(PRODUCTION)
    n = sum(len(d) - 1 for d in DIALOGUES)
    print(
        contextualize(
            "What about Business?",
            ["How long are pipeline logs kept on the Team plan?"],
        )
    )
    for mode in ("as asked", "concatenate", "rewrite"):
        print(
            f"  follow-ups, {mode:12} "
            f"{follow_ups(pipe, mode)}/{n}"
        )
    print()
    for policy in (None, "append", "slots"):
        label = policy or "no memory"
        print(
            f"  sessions, {label:10} "
            f"{sessions(pipe, policy)}/{len(SESSIONS)}"
        )


if __name__ == "__main__":
    main()
