"""An atomic claim-file queue, so several GPUs can drain one run cooperatively.

Every run writes one JSON file per example into `<run>/res/NNNN.json` behind an
`O_EXCL` claim in `<run>/claim/NNNN.lock`. Creating the claim is a single atomic
filesystem operation, so two workers cannot take the same example, and losing the
race simply means someone else has it. Re-running a run therefore resumes where it
stopped instead of starting over, and adding a worker to a run already in progress
is safe.

One thing a restart does not recover on its own: an example that was in flight when
its process was killed. Its claim file is still there and it has no result, so every
later worker skips it and the run ends one short. Once nothing is working on the run,
delete the claims that have no matching result and start it again:

    for f in <run>/claim/*.lock; do
        [ -e <run>/res/$(basename "$f" .lock).json ] || rm "$f"
    done

Examples are handed out longest first. A 231-slot CLAPNQ question costs 138 times what
a 19-slot one does in forward passes -- 28,906 against 209 -- and about 360 times in
seconds, so starting the long ones last would leave a single GPU grinding alone for
minutes after the others had finished.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path


class WorkQueue:
    def __init__(self, run_directory: str | Path) -> None:
        self.root = Path(run_directory)
        self.results = self.root / "res"
        self.claims = self.root / "claim"
        self.results.mkdir(parents=True, exist_ok=True)
        self.claims.mkdir(parents=True, exist_ok=True)

    def result_path(self, index: int) -> Path:
        return self.results / f"{index:04d}.json"

    def claim(self, index: int) -> bool:
        """Try to take example `index`. False means done already or taken."""
        if self.result_path(index).exists():
            return False
        try:
            os.close(
                os.open(
                    self.claims / f"{index:04d}.lock",
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            )
            return True
        except FileExistsError:
            return False

    def publish(self, index: int, record: dict) -> None:
        """Write a result so that a reader never sees a half-written file."""
        target = self.result_path(index)
        temporary = target.with_suffix(".json.part")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(record, handle)
        temporary.replace(target)

    def completed(self) -> int:
        return sum(1 for _ in self.results.glob("[0-9]*.json"))

    def load_all(self, expected: int | None = None) -> list[dict]:
        """Every result, or a clear error naming what is missing.

        A partial run must never be scored: an average over 1,200 of 1,814 turns
        is not the number the table prints.
        """
        paths = sorted(self.results.glob("[0-9]*.json"))
        records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if expected is not None and len(records) != expected:
            present = {int(path.stem) for path in paths}
            missing = sorted(set(range(expected)) - present)
            raise RuntimeError(
                f"{self.root} holds {len(records)}/{expected} results; "
                f"missing indices {missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        return records


def longest_first(
    examples: list[dict], budget: Callable[[dict], int]
) -> list[tuple[int, dict]]:
    """Pair each example with its index in file order, heaviest first."""
    order = sorted(range(len(examples)), key=lambda i: -budget(examples[i]))
    return [(i, examples[i]) for i in order]


def drain(
    queue: WorkQueue, ordered: list[tuple[int, dict]]
) -> Iterator[tuple[int, dict]]:
    """Yield only the examples this worker successfully claimed."""
    for index, example in ordered:
        if queue.claim(index):
            yield index, example
