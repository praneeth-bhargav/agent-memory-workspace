import json
from pathlib import Path

import pytest
from langsmith import evaluate
from langsmith.schemas import Example
from uuid import uuid4

from evals.langsmith_eval import feedback, publish, run_experiment
from evals.reporting import save_report


def test_feedback_keeps_memory_and_agent_separate_and_safety_blocks_pass():
    result = {'status': 'fail', 'turns': [
        {'grades': {'agent': {'passed': True, 'rationale': 'Answer is correct.'},
                    'memory': {'passed': False, 'rationale': 'Wrong saved owner.'}},
         'checks': [{'name': 'confirmation', 'passed': False, 'rationale': 'Unapproved write.'}]}]}
    scores = {entry['key']: entry['score'] for entry in feedback(outputs=result)['results']}
    assert scores == {'agent_correctness': False, 'memory_correctness': False, 'state_safety': False}
    assert feedback(outputs={'status': 'blocked', 'reason': 'Missing setup'})['results'][0]['score'] is None


def test_incremental_report_keeps_incomplete_coverage(tmp_path):
    path = tmp_path / 'report.json'
    save_report(path, [{'id': 'one', 'status': 'pass', 'turns': []}], total=3)
    report = json.loads(path.read_text())
    assert report['remaining'] == 2
    assert report['counts'] == {'pass': 1}
    assert report['scores']['memory_correctness']['scored'] == 0


def test_sdk_evaluates_locally_without_provider_or_upload():
    example = Example(id=uuid4(), dataset_id=uuid4(), inputs={'case': {'id': 'synthetic'}}, outputs={})
    results = evaluate(lambda inputs: {'status': 'pass', 'turns': [
        {'grades': {'agent': {'passed': True, 'rationale': 'OK'}, 'memory': None}, 'checks': []}]},
        data=[example], evaluators=[feedback], upload_results=False, max_concurrency=0)
    rows = list(results)
    assert len(rows) == 1
    scores = rows[0]['evaluation_results']['results']
    assert any(score.key == 'agent_correctness' and score.score is True for score in scores)
    assert any(score.key == 'memory_correctness' and score.score is None for score in scores)


def test_cloud_requires_key_before_running_target(monkeypatch, tmp_path):
    monkeypatch.delenv('LANGSMITH_API_KEY', raising=False)
    monkeypatch.delenv('LANGCHAIN_API_KEY', raising=False)
    with pytest.raises(ValueError, match='LANGSMITH_API_KEY'):
        run_experiment([{'id': 'test'}], lambda case: pytest.fail('Target must not run'), tmp_path / 'out.json', 1)


def test_publication_keeps_exact_labels_and_uses_immutable_dataset():
    class FakeClient:
        def __init__(self):
            self.datasets = []
            self.examples = []
        def read_dataset(self, **kwargs):
            from langsmith.utils import LangSmithNotFoundError
            raise LangSmithNotFoundError('missing')
        def create_dataset(self, **kwargs):
            self.datasets.append(kwargs)
            return type('Dataset', (), {'id': uuid4(), 'name': kwargs['dataset_name']})()
        def list_examples(self, **kwargs):
            return []
        def create_example(self, **kwargs):
            self.examples.append(kwargs)
            return None
    client = FakeClient()
    case = {'id': 'test', 'setup': 'Synthetic.', 'user_message': 'Hello', 'expected_behavior': 'Exact human label.'}
    dataset = publish(client, [case])
    assert dataset.name.startswith('agent-memory-lab-')
    assert client.examples[0]['inputs']['case'] == case
    assert client.examples[0]['outputs']['expected_behavior'] == 'Exact human label.'
    assert case['expected_behavior'] == 'Exact human label.'


def test_unsupported_state_checks_are_blocked_before_model_call():
    from evals.run_memory_lab import evaluate_case
    result = evaluate_case({'id': 'test', 'setup': 'Synthetic setup', 'user_message': 'Hello',
                            'expected_behavior': 'Human expectation',
                            'deterministic_checks': ['Unsupported database expectation']}, None)
    assert result['status'] == 'blocked'
    assert 'Unsupported deterministic checks' in result['reason']


def test_state_checks_reject_unconfirmed_mock_write():
    from evals.state_checks import check_turn
    before = {'pending': None, 'mock_write_ids': []}
    after = {'pending': None, 'mock_write_ids': ['mock:email']}
    checks = check_turn('send it', before, after)
    assert not checks[0].passed


def test_local_cli_keeps_unlabeled_visible_without_paid_calls(monkeypatch, tmp_path):
    from evals.run_memory_lab import main
    monkeypatch.setattr('evals.run_memory_lab.load_cases', lambda: [
        {'id': 'source-001', 'setup': 'Synthetic', 'user_message': 'Example', 'expected_behavior': ''}])
    monkeypatch.setattr('evals.run_memory_lab.verify_synthetic_inputs', lambda cases: None)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr('sys.argv', ['eval', '--case', 'source-001', '--output', str(tmp_path / 'report.json')])
    main()
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['counts'] == {'unlabeled': 1}
    assert report['remaining'] == 0


def test_partial_turn_labels_are_not_scored():
    from evals.run_memory_lab import labeled
    assert not labeled({'turns': [{'user_message': 'First', 'expected_agent_behavior': 'Human label'},
                                  {'user_message': 'Second'}]})


def test_separate_following_label_is_recognized():
    from evals.run_memory_lab import labeled
    assert labeled({'turns': [{'user_message': 'First'}, {'expected_agent_behavior': 'Human label'}]})


def test_malformed_case_is_rejected_without_target_call():
    from evals.run_memory_lab import evaluate_case
    result = evaluate_case({'id': 'bad', 'setup': 'Synthetic', 'user_message': 123,
                            'expected_behavior': 'Human label'}, None)
    assert result['status'] == 'blocked'
    assert 'Invalid fixture' in result['reason']


def test_resume_validates_identity_and_preserves_completed_trials(tmp_path):
    from evals.reporting import resume_results
    path = tmp_path / 'report.json'
    completed = [{'id': 'one', 'status': 'unlabeled', 'trial': 1}]
    save_report(path, completed, total=2, identity='original')
    assert resume_results(path, 'original') == completed
    with pytest.raises(ValueError, match='identity'):
        resume_results(path, 'changed-labels')


def test_cloud_cli_fails_without_key_before_openai_init(monkeypatch):
    from evals.run_memory_lab import main
    monkeypatch.delenv('LANGSMITH_API_KEY', raising=False)
    monkeypatch.delenv('LANGCHAIN_API_KEY', raising=False)
    monkeypatch.setattr('evals.run_memory_lab.OpenAI', lambda **kwargs: pytest.fail('Must not initialize OpenAI'))
    monkeypatch.setattr('sys.argv', ['eval', '--langsmith'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_eval_transport_honors_existing_ipv4_flag(monkeypatch):
    from evals.run_memory_lab import provider_http_client
    monkeypatch.setenv('AGENT_OPENAI_IPV4_ONLY', 'true')
    with provider_http_client() as client:
        assert client._transport._pool._local_address == '0.0.0.0'
        assert client.timeout.connect == 30


def test_manually_changed_remote_label_is_rejected():
    from types import SimpleNamespace
    from uuid import NAMESPACE_URL, uuid5
    dataset_id = uuid4()
    example_id = uuid5(NAMESPACE_URL, f'{dataset_id}/test')
    case = {'id': 'test', 'setup': 'Synthetic', 'user_message': 'Hi', 'expected_behavior': 'Original human label'}
    class FakeClient:
        def read_dataset(self, **kwargs):
            return SimpleNamespace(id=dataset_id, name='existing')
        def list_examples(self, **kwargs):
            return [SimpleNamespace(id=example_id, inputs={'case': case}, outputs={'expected_behavior': 'Edited remotely'})]
    with pytest.raises(ValueError, match='differs'):
        publish(FakeClient(), [case])


def test_report_output_cannot_replace_database_or_unrelated_json(tmp_path):
    db = tmp_path / 'active.sqlite3'
    db.write_bytes(b'SQLite format 3\x00do not destroy')
    unrelated = tmp_path / 'user.json'
    unrelated.write_text('{"important": true}')
    for path in (db, unrelated):
        original = path.read_bytes()
        with pytest.raises(ValueError):
            save_report(path, [], total=0)
        assert path.read_bytes() == original


def test_resume_rejects_duplicate_or_out_of_range_trials(tmp_path):
    from evals.reporting import resume_results
    path = tmp_path / 'report.json'
    row = {'id': 'one', 'trial': 1, 'status': 'unlabeled'}
    save_report(path, [row, row], total=2, identity='same')
    with pytest.raises(ValueError, match='trial'):
        resume_results(path, 'same', ['one'], 2)


def test_memory_unchanged_check_rejects_transient_replacement():
    from evals.state_checks import check_final, UNCHANGED_MEMORY
    before = {'active_memories': [], 'pending': None, 'outgoing_mock_email_ids': [], 'memory_operation_count': 0}
    after = {**before, 'memory_operation_count': 2}
    assert not check_final([UNCHANGED_MEMORY], before, after)[0].passed


def test_later_provider_failure_keeps_earlier_turn_evidence(monkeypatch):
    from evals.run_memory_lab import evaluate_case
    from memory_layer.agent import AgentEvent
    class FailingAgent:
        def __init__(self, memory, model, saver, sources, *args):
            self.memory, self.sources = memory, sources
        def pending_action(self, chat_id):
            return None
        def stream_answer(self, text, *args):
            if text == 'second':
                raise RuntimeError('sensitive provider detail')
            yield AgentEvent(kind='answer', data='first answer')
    monkeypatch.setattr('evals.run_memory_lab.MemoryAgent', FailingAgent)
    monkeypatch.setattr('evals.run_memory_lab.ChatOpenAI', lambda **kwargs: object())
    result = evaluate_case({'id': 'memory-003', 'setup': 'Synthetic', 'turns': [
        {'user_message': 'first', 'expected_agent_behavior': 'First label'},
        {'user_message': 'second', 'expected_agent_behavior': 'Second label'}]}, object())
    assert result['status'] == 'error'
    assert result['turns'][0]['events'][0]['data'] == 'first answer'
    assert 'sensitive provider detail' not in json.dumps(result)


def test_hosted_guard_rejects_modified_synthetic_source(tmp_path):
    from evals.fixtures import verify_synthetic_inputs
    with pytest.raises(ValueError, match='synthetic'):
        verify_synthetic_inputs([{'id': 'not-an-approved-fixture', 'user_message': 'private'}])


def test_required_memory_grade_cannot_silently_be_unscored():
    from types import SimpleNamespace
    from evals.grading import grade_turn, TurnGrades, Grade
    verdict = TurnGrades(agent=Grade(passed=True, rationale='OK'), memory=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kwargs:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=verdict))]))))
    with pytest.raises(ValueError, match='memory grade'):
        grade_turn('Do not change either memory until clarification.', {}, client, 'test')


def test_hosted_orchestration_saves_trials_and_reuses_validated_examples(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from uuid import NAMESPACE_URL, uuid5
    from contextlib import nullcontext
    case = json.loads(Path('evals/memory_lab/memory.json').read_text())[0]
    dataset = SimpleNamespace(id=uuid4(), name='synthetic-test')
    example = Example(id=uuid5(NAMESPACE_URL, f'{dataset.id}/{case["id"]}'), dataset_id=dataset.id,
                      inputs={'case': case}, outputs={'expected_behavior': case['expected_behavior']})
    closed = []
    class FakeClient:
        def close(self) -> None:
            closed.append(True)
        def read_dataset(self, **kwargs):
            return dataset
        def list_examples(self, **kwargs):
            return [example]
        def evaluate(self, target, **kwargs):
            assert kwargs['data'] == [example]
            for _ in range(kwargs['num_repetitions']):
                result = target(example.inputs)
                assert feedback(outputs=result)['results'][0]['score'] is None
            return SimpleNamespace(wait=lambda: None, url='https://example.invalid/synthetic-experiment')
    monkeypatch.setenv('LANGSMITH_API_KEY', 'test-only-not-a-real-key')
    monkeypatch.setattr('evals.langsmith_eval.Client', FakeClient)
    monkeypatch.setattr('evals.langsmith_eval.tracing_context', lambda **kwargs: nullcontext())
    path = tmp_path / 'report.json'
    results = run_experiment([case], lambda value, fixtures: {'id': value['id'], 'status': 'unlabeled'}, path, 2)
    assert [row['trial'] for row in results] == [1, 2]
    assert json.loads(path.read_text())['remaining'] == 0
    assert json.loads(path.read_text())['experiment_url'].endswith('/synthetic-experiment')
    assert closed == [True]


def test_changed_source_fixture_is_blocked_before_publication(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path
    from evals.fixtures import verify_synthetic_inputs
    root = Path.cwd()
    manifest = json.loads((root / 'evals/synthetic-fixtures.json').read_text())
    for row in manifest:
        destination = tmp_path / row['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / row['path'], destination)
    monkeypatch.setattr('evals.fixtures.ROOT', tmp_path)
    snapshot = verify_synthetic_inputs([])
    original = snapshot.read_text('data/persona.json')
    (tmp_path / 'data/persona.json').write_text('{"private": "not approved"}')
    assert snapshot.read_text('data/persona.json') == original
    with pytest.raises(ValueError, match='synthetic fixtures changed'):
        verify_synthetic_inputs([])


def test_remote_example_ids_and_labels_are_validated_as_one_snapshot():
    from types import SimpleNamespace
    from evals.langsmith_eval import validated_examples
    case = {'id': 'test', 'expected_behavior': 'Human label'}
    dataset = SimpleNamespace(id=uuid4())
    example = Example(id=uuid4(), dataset_id=dataset.id, inputs={'case': case}, outputs={})
    client = SimpleNamespace(list_examples=lambda **kwargs: [example, example])
    with pytest.raises(ValueError, match='duplicate'):
        validated_examples(client, dataset, [case])


def test_local_resume_skips_completed_trial(monkeypatch, tmp_path):
    from evals.run_memory_lab import main
    monkeypatch.setattr('evals.run_memory_lab.load_cases', lambda: [
        {'id': 'source-001', 'setup': 'Synthetic', 'user_message': 'Example', 'expected_behavior': ''}])
    monkeypatch.setattr('evals.run_memory_lab.verify_synthetic_inputs', lambda cases: None)
    path = tmp_path / 'report.json'
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    args = ['eval', '--case', 'source-001', '--repeat', '2', '--output', str(path)]
    monkeypatch.setattr('sys.argv', args)
    main()
    monkeypatch.setattr('evals.run_memory_lab.evaluate_case', lambda *args: pytest.fail('Already completed'))
    monkeypatch.setattr('sys.argv', [*args, '--resume'])
    main()
    report = json.loads(path.read_text())
    assert report['counts'] == {'unlabeled': 2}
    assert report['remaining'] == 0
