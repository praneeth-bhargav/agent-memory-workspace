"""Opt-in hosted experiments over synthetic, human-labeled Memory Lab cases."""
import hashlib
import json
import os
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langsmith import Client, tracing_context
from langsmith.schemas import Dataset, Example
from langsmith.utils import LangSmithNotFoundError

from evals.grading import scores
from evals.fixtures import verify_synthetic_inputs
from evals.reporting import save_report

DATASET_PREFIX = 'agent-memory-lab-'
EXPERIMENT_PREFIX = 'agent-memory-lab'
DATASET_DESCRIPTION = 'Synthetic evaluation cases. Reference outputs preserve repository human labels.'
KEY_ERROR = 'Set LANGSMITH_API_KEY before publishing synthetic evaluations to LangSmith.'


def feedback(outputs: dict) -> dict:
    return {'results': [score.model_dump() for score in scores(outputs)]}


def publish(client: Client, cases: list[dict]) -> Dataset:
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    name = DATASET_PREFIX + digest[:16]
    try:
        dataset = client.read_dataset(dataset_name=name)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(dataset_name=name, description=DATASET_DESCRIPTION)
    existing = {str(example.id): example for example in client.list_examples(dataset_id=dataset.id)}
    expected_ids = {str(uuid5(NAMESPACE_URL, f'{dataset.id}/{case["id"]}')) for case in cases}
    if set(existing) - expected_ids:
        raise ValueError('Published dataset differs from repository cases; unexpected examples found.')
    for case in cases:
        example_id = uuid5(NAMESPACE_URL, f'{dataset.id}/{case["id"]}')
        reference = {key: value for key, value in case.items()
                     if key in {'expected_behavior', 'turns', 'deterministic_checks'}}
        prior = existing.get(str(example_id))
        if prior is None:
            client.create_example(dataset_id=dataset.id, example_id=example_id,
                                  inputs={'case': case}, outputs=reference, metadata={'case_id': case['id']})
        elif prior.inputs != {'case': case} or prior.outputs != reference:
            raise ValueError(f'Published example differs from repository case {case["id"]}; refusing to overwrite labels.')
    return dataset


def validated_examples(client: Client, dataset: Dataset, cases: list[dict]) -> list[Example]:
    examples = list(client.list_examples(dataset_id=dataset.id))
    expected = {str(uuid5(NAMESPACE_URL, f'{dataset.id}/{case["id"]}')): case for case in cases}
    if len(examples) != len(expected) or {str(example.id) for example in examples} != set(expected):
        raise ValueError('Published dataset has missing, unexpected, or duplicate examples.')
    for example in examples:
        case = expected[str(example.id)]
        reference = {key: value for key, value in case.items()
                     if key in {'expected_behavior', 'turns', 'deterministic_checks'}}
        if example.inputs != {'case': case} or example.outputs != reference:
            raise ValueError('Published example differs from approved inputs or labels.')
    return examples


def run_experiment(cases: list[dict], target: Callable[..., dict], output: Path,
                   repetitions: int, **metadata: object) -> list[dict]:
    if not (os.getenv('LANGSMITH_API_KEY') or os.getenv('LANGCHAIN_API_KEY')):
        raise ValueError(KEY_ERROR)
    fixtures = verify_synthetic_inputs(cases)
    results: list[dict] = []
    total = len(cases) * repetitions
    save_report(output, results, total=total, **metadata)
    with closing(Client()) as client:
        dataset = publish(client, cases)
        examples = validated_examples(client, dataset, cases)

        def evaluate_case(inputs: dict) -> dict:
            case = next(case for case in cases if case['id'] == inputs['case']['id'])
            if inputs != {'case': case}:
                raise ValueError('Dataset inputs changed after validation; refusing to execute.')
            result = {**target(case, fixtures=fixtures), 'trial': 1 + sum(row['id'] == case['id'] for row in results)}
            results.append(result)
            save_report(output, results, total=total, dataset=dataset.name, **metadata)
            return result

        with tracing_context(enabled=True, client=client):
            experiment = client.evaluate(evaluate_case, data=examples,
                                         evaluators=[feedback], experiment_prefix=EXPERIMENT_PREFIX,
                                         max_concurrency=0, num_repetitions=repetitions, metadata=metadata)
            experiment.wait()
        if len(results) != total:
            raise RuntimeError(f'Incomplete LangSmith experiment: recorded {len(results)} of {total} trials.')
        save_report(output, results, total=total, dataset=dataset.name,
                    experiment_url=experiment.url, **metadata)
        print(f'LangSmith experiment: {experiment.url}', flush=True)
    return results
