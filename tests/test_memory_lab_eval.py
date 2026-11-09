from evals.run_memory_lab import judge_turn


class FakeCompletions:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return type('Response', (), {'choices': [type('Choice', (), {
            'message': type('Message', (), {'parsed': self.verdict})()
        })()]})()


class FakeClient:
    def __init__(self, verdict):
        self.chat = type('Chat', (), {'completions': FakeCompletions(verdict)})()


def test_judge_turn_returns_structured_pass_and_uses_label_as_rubric():
    from evals.run_memory_lab import JudgeVerdict

    client = FakeClient(JudgeVerdict(passed=True, rationale='All required points are present.'))
    actual = {'answer': 'Support summaries go to #ops.', 'state': {'active_memories': []}}

    result = judge_turn('Say support summaries go to #ops.', actual, client)

    assert result['passed'] is True
    assert result['rationale'] == 'All required points are present.'
    assert 'Say support summaries go to #ops.' in client.chat.completions.calls[0]['messages'][1]['content']


def test_judge_turn_preserves_failure_reason():
    from evals.run_memory_lab import JudgeVerdict

    client = FakeClient(JudgeVerdict(passed=False, rationale='The response names the wrong channel.'))

    result = judge_turn('Say support summaries go to #ops.', {'answer': 'Use #random.'}, client)

    assert result == {'passed': False, 'rationale': 'The response names the wrong channel.'}
