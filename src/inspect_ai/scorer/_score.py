from contextvars import ContextVar

from inspect_ai.solver._task_state import TaskState

from ._metric import Score
from ._scorer import Scorer
from ._target import Target


async def score(state: TaskState) -> list[Score]:
    """Score a TaskState.

    Score a task state from within a solver.

    Args:
      state (TaskState): `TaskState` to submit for scoring

    Returns:
      List of scores (one for each task scorer)

    Raises:
      RuntimerError: If called from outside a task or within
        a task that does not have a scorer.

    """
    from inspect_ai.log._transcript import ScoreEvent, transcript

    scorers = _scorers.get(None)
    target = _target.get(None)
    if scorers is None or target is None:
        raise RuntimeError(
            "The score() function can only be called while executing a task with a scorer."
        )

    scores: list[Score] = []
    for scorer in scorers:
        score = await scorer(state, target)
        scores.append(score)
        transcript()._event(
            ScoreEvent(score=score, target=target.target, intermediate=True)
        )

    return scores


def init_scoring_context(scorers: list[Scorer], target: Target) -> None:
    _scorers.set(scorers)
    _target.set(target)


_scorers: ContextVar[list[Scorer]] = ContextVar("scorers")
_target: ContextVar[Target] = ContextVar("target")
