import assert from 'node:assert/strict';

const base = process.env.WORKSPACE_QA_URL;
if (!base || !['127.0.0.1', 'localhost'].includes(new URL(base).hostname)) throw new Error('Use an isolated local test server.');
const expected = process.env.EXPECT_SIGNOFF || 'Rohan';
async function chat(text) {
  const response = await fetch(`${base}/api/chat/stream`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
  });
  assert.equal(response.status, 200);
  const events = (await response.text()).trim().split('\n').map(line => JSON.parse(line));
  assert(!events.some(event => event.kind === 'error'), JSON.stringify(events));
  return events;
}
let events = await chat("draft a mail for arjun that I won't be available for next 2 days");
let context = events.findIndex(event => event.kind === 'tool_result' && event.name === 'get_action_context' && event.status === 'success');
let draft = events.findIndex(event => event.kind === 'tool_result' && event.name === 'prepare_mock_email' && event.status === 'success');
assert(context >= 0, JSON.stringify(events));
if (draft < 0) {
  assert(events.some(event => event.kind === 'answer' && event.data.includes(expected)), JSON.stringify(events));
  events = await chat('Prepare that email as a mock action and ask for my confirmation.');
  context = events.findIndex(event => event.kind === 'tool_result' && event.name === 'get_action_context' && event.status === 'success');
  draft = events.findIndex(event => event.kind === 'tool_result' && event.name === 'prepare_mock_email' && event.status === 'success');
}
assert(context >= 0 && draft > context, JSON.stringify(events));
assert.equal(JSON.parse(events[context].data).action, 'email');
const email = JSON.parse(events[draft].data).payload;
assert(email.body.split('\n').some(line => line.trim() === expected || (expected === 'Rohan' && line.trim() === 'Rohan Menon')), email.body);
assert(!/\[(?:your\s+)?(?:name|signature)\]/i.test(email.body), email.body);
const before = await (await fetch(`${base}/api/sources`)).json();
assert(!before.mail.some(thread => thread.messages.some(message => message.body === email.body)));
const confirmed = await chat('yes');
assert(confirmed.some(event => event.kind === 'answer' && event.data.includes('Confirmed mock email')));
const after = await (await fetch(`${base}/api/sources`)).json();
assert(after.mail.some(thread => thread.messages.some(message => message.body === email.body)));
console.log(`PASS email context, ${expected} sign-off, confirmation and persisted mock email`);
