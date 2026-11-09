"""Incremental local evidence survives interrupted or failed hosted experiments."""
import json
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from evals.grading import AGENT_SCORE, MEMORY_SCORE, SAFETY_SCORE, scores

SCORE_KEYS = (AGENT_SCORE, MEMORY_SCORE, SAFETY_SCORE)
REPORT_TYPE = 'agent-memory-lab-evaluation-v1'


def validate_output(path: Path) -> None:
    if path.suffix != '.json' or path.is_symlink():
        raise ValueError('Output must be a regular .json evaluation report, not a database or symlink.')
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding='utf-8'))
            valid = isinstance(existing, dict) and existing.get('report_type') == REPORT_TYPE
        except (ValueError, UnicodeError):
            valid = False
        if not valid:
            raise ValueError('Refusing to overwrite an existing non-evaluation file.')


def resume_results(path: Path, identity: str, case_ids: list[str] | None = None,
                   repetitions: int = 1) -> list[dict]:
    validate_output(path)
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('identity') != identity:
        raise ValueError('Report identity differs: cases, labels, models, repetitions, or runner code changed.')
    rows = report['cases']
    trials = [(row['id'], row['trial']) for row in rows]
    if len(set(trials)) != len(trials) or any(
            not isinstance(trial, int) or isinstance(trial, bool) or not 1 <= trial <= repetitions
            or (case_ids is not None and identity not in case_ids) for identity, trial in trials):
        raise ValueError('Invalid or duplicate trial records in report.')
    if any(row['status'] not in {'pass', 'fail', 'unlabeled', 'blocked', 'error'} for row in rows):
        raise ValueError('Invalid trial status in report.')
    return [result for result in rows if result['status'] not in {'blocked', 'error'}]


def save_report(path: Path, results: list[dict], total: int, **metadata: object) -> None:
    validate_output(path)
    feedback = [score for result in results for score in scores(result)]
    summary = {key: {'passed': sum(score.score is True for score in feedback if score.key == key),
                     'failed': sum(score.score is False for score in feedback if score.key == key),
                     'scored': sum(score.score is not None for score in feedback if score.key == key)}
               for key in SCORE_KEYS}
    report = {**metadata, 'report_type': REPORT_TYPE, 'counts': dict(Counter(result['status'] for result in results)),
              'total': total, 'remaining': total - len(results), 'scores': summary, 'cases': results}
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', dir=path.parent, prefix='agent-eval-', suffix='.tmp',
                            delete=False, encoding='utf-8') as temporary:
        temporary.write(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
