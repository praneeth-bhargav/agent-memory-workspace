"""Validate case structure without changing human-authored expectations."""
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_MANIFEST = ROOT / 'evals' / 'synthetic-fixtures.json'
EXPECTED_SETUP_SHA256 = 'e512b9638a1ca2c0148d197caef4c3609c6fe7825f5556f7e4bfc064f12353fe'


class TurnFixture(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    user_message: str | None = None
    expected_agent_behavior: str = ''
    chat_id: str | None = None


class CaseFixture(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    id: str = Field(min_length=1)
    category: str = ''
    setup: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    deterministic_checks: list[str] = Field(default_factory=list)
    user_message: str | None = None
    expected_behavior: str = ''
    turns: list[TurnFixture] | None = None

    @model_validator(mode='after')
    def require_conversation(self) -> 'CaseFixture':
        if (self.user_message is None) == (self.turns is None):
            raise ValueError('Provide exactly one user_message or turns')
        if self.turns is not None and not any(turn.user_message for turn in self.turns):
            raise ValueError('Conversation must include a user message')
        if self.turns and not self.turns[0].user_message:
            raise ValueError('A label must follow a user turn')
        return self


class FixtureFingerprint(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    path: str
    sha256: str


@dataclass(frozen=True)
class FixtureFile:
    path: str
    content: bytes


@dataclass(frozen=True)
class SyntheticFixtures:
    files: tuple[FixtureFile, ...]

    def read_text(self, relative_path: str) -> str:
        return next(row.content.decode('utf-8') for row in self.files if row.path == relative_path)


def verify_synthetic_inputs(cases: list[dict]) -> SyntheticFixtures:
    manifest = [FixtureFingerprint.model_validate(row)
                for row in json.loads(SYNTHETIC_MANIFEST.read_text(encoding='utf-8'))]
    paths = sorted((ROOT / 'data').rglob('*.json')) + sorted((ROOT / 'data').rglob('*.jsonl'))
    paths += sorted((ROOT / 'evals' / 'memory_lab').glob('*.json'))
    approved = {entry.path: entry.sha256 for entry in manifest}
    snapshot = SyntheticFixtures(tuple(FixtureFile(str(path.relative_to(ROOT)), path.read_bytes()) for path in paths))
    observed = {row.path: hashlib.sha256(row.content).hexdigest() for row in snapshot.files}
    if observed != approved or any(path.is_symlink() for path in paths):
        raise ValueError('Approved synthetic fixtures changed; review data before updating synthetic-fixtures.json.')
    known = [case for row in snapshot.files if row.path.startswith('evals/memory_lab/')
             for case in json.loads(row.content)]
    setup_digest = hashlib.sha256(json.dumps(sorted((case['id'], case['setup']) for case in known),
                                              ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if setup_digest != EXPECTED_SETUP_SHA256:
        raise ValueError('Synthetic setup descriptions changed; review evaluator setup before running.')
    for case in cases:
        if case not in known:
            raise ValueError('Only approved repository synthetic cases may be uploaded.')
        CaseFixture.model_validate(case)
    return snapshot


def conversation(case: dict) -> list[TurnFixture]:
    raw = case.get('turns', [{'user_message': case.get('user_message'),
                              'expected_agent_behavior': case.get('expected_behavior', '')}])
    turns: list[TurnFixture] = []
    for value in raw:
        turn = TurnFixture.model_validate(value)
        if turn.user_message is not None:
            turns.append(turn)
        elif turns and turn.expected_agent_behavior:
            turns[-1] = turns[-1].model_copy(update={'expected_agent_behavior': turn.expected_agent_behavior})
    return turns
