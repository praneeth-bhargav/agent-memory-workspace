import assert from 'node:assert/strict';
import { chromium } from 'playwright-core';

const base = process.env.WORKSPACE_QA_URL;
const chrome = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
if (!base || !['127.0.0.1', 'localhost'].includes(new URL(base).hostname)) {
  throw new Error('Set WORKSPACE_QA_URL to an isolated local test server; this test creates mock records.');
}
const suffix = String(Date.now());
const slackText = `QA workspace Slack ${suffix}`;
const emailSubject = `QA workspace mail ${suffix}`;
const meetingTitle = `QA workspace meeting ${suffix}`;
const memoryText = `QA preferred fruit is starfruit ${suffix}`;
const browser = await chromium.launch({ headless: true, executablePath: chrome });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
let intentionalError = false;
page.on('pageerror', error => errors.push(error.message));
page.on('console', message => { if (message.type() === 'error' && !intentionalError) errors.push(message.text()); });
page.setDefaultTimeout(30000);

async function apiChat(text) {
  const response = await fetch(`${base}/api/chat/stream`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
  });
  assert.equal(response.status, 200, 'chat stream HTTP status');
  const events = (await response.text()).trim().split('\n').map(line => JSON.parse(line));
  assert(!events.some(event => event.kind === 'error'), JSON.stringify(events));
  return events;
}

async function send(text) {
  await page.getByLabel('Your message').fill(text);
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await page.waitForFunction(() => !document.querySelector('.composer textarea').disabled, null, { timeout: 120000 });
}

try {
  const initial = await fetch(`${base}/api/sources`);
  assert.equal(initial.status, 200);
  const snapshot = await initial.json();
  assert.equal(snapshot.contacts.length, 9);
  assert.equal(snapshot.mail.length, 9);
  assert.equal(snapshot.calendar_events.length, 31);
  assert.equal(snapshot.slack_channels.length, 4);
  assert.equal((await fetch(`${base}/api/sources`, { method: 'POST' })).status, 405);

  await page.goto(`${base}/workspace`);
  await page.getByRole('heading', { name: 'Workspace' }).waitFor();
  await page.getByText('Proposal timeline', { exact: true }).waitFor();
  await page.getByText('Proposal timeline', { exact: true }).click();
  await page.getByText(/our board would like delivery pulled in/i).waitFor();
  await page.getByRole('button', { name: /Contacts/ }).click();
  await page.getByText('Maya Rao', { exact: true }).waitFor();
  await page.getByText('Maya Iyer', { exact: true }).waitFor();
  await page.getByRole('button', { name: /Calendar/ }).click();
  await page.getByText('Board prep', { exact: true }).waitFor();
  await page.getByRole('button', { name: /Slack/ }).click();
  await page.getByRole('button', { name: '#ops', exact: true }).click();
  await page.getByText(/Weekly support summary: 128 tickets/).waitFor();
  assert.equal(await page.getByRole('button', { name: /create|add contact|new event/i }).count(), 0);
  await page.screenshot({ path: '/tmp/agent-workspace-desktop.png' });

  const prepared = await apiChat(`Prepare this mock Slack post to #ops exactly: ${slackText}. Ask for confirmation.`);
  assert(prepared.some(event => event.name === 'get_action_context' && event.kind === 'tool_result' && JSON.parse(event.data).action === 'slack'));
  assert(prepared.some(event => event.name === 'prepare_mock_slack_post' && event.kind === 'tool_result'));
  assert(!((await (await fetch(`${base}/api/sources`)).json()).slack_messages.some(item => item.text.includes(slackText))));
  assert((await apiChat('yes')).some(event => event.kind === 'answer' && event.data.includes('Confirmed mock slack')));
  await page.getByText(new RegExp(slackText)).waitFor({ timeout: 20000 });
  console.log('PASS external API/tool action refreshed Slack without reloading');

  await page.getByRole('button', { name: /Mail/ }).click();
  await send(`Prepare a mock email to maya.rao@acme-industries.example with subject ${emailSubject} and body: Synthetic workspace verification. Ask me to confirm.`);
  await page.getByText(/Prepared mock email; nothing has been sent/).waitFor({ timeout: 120000 });
  await page.getByText('Getting email context', { exact: false }).first().waitFor();
  await page.screenshot({ path: '/tmp/agent-action-context-after.png' });
  await page.getByText('Preparing mock email', { exact: false }).first().click();
  await page.getByText(new RegExp(emailSubject)).first().waitFor();
  assert.equal((await (await fetch(`${base}/api/sources`)).json()).mail.some(item => item.subject === emailSubject), false);
  await send('yes');
  await page.getByText(/Confirmed mock email/).waitFor({ timeout: 30000 });
  await page.getByText(emailSubject, { exact: true }).waitFor({ timeout: 20000 });
  assert.equal(await page.locator('.mail-thread .record-title').first().textContent(), emailSubject);
  console.log('PASS UI agent confirmation persisted email and refreshed Mail');

  await page.getByRole('button', { name: /Calendar/ }).click();
  const booking = await apiChat(`Use list_calendar_people, then prepare a mock booking titled ${meetingTitle} for Rohan on May 25 2026 from 12:30 to 13:00 IST. Ask for confirmation after preparing it.`);
  assert(booking.some(event => event.name === 'get_action_context' && event.kind === 'tool_result' && JSON.parse(event.data).action === 'calendar'));
  assert(booking.some(event => event.name === 'prepare_mock_booking' && event.kind === 'tool_result'), JSON.stringify(booking));
  assert((await apiChat('yes')).some(event => event.kind === 'answer' && event.data.includes('Confirmed mock booking')));
  await page.getByText(meetingTitle, { exact: true }).waitFor({ timeout: 20000 });
  console.log('PASS external agent booking refreshed Calendar');

  await send(`Remember that my ${memoryText}.`);
  await page.getByText('Updating memory', { exact: false }).last().waitFor({ timeout: 120000 });
  const activity = page.locator('details.activity').filter({ hasText: 'Updating memory' }).last();
  await activity.locator('summary').click();
  const saved = JSON.parse(await activity.locator('pre').last().textContent());
  assert.equal(saved.status, 'created');
  assert.match(saved.memory, /starfruit/i);
  console.log('PASS streamed memory activity exposes its saved result');

  intentionalError = true;
  const failSources = route => route.fulfill({ status: 503, contentType: 'application/json',
    body: JSON.stringify({ detail: 'Source test failure; check the backend log.' }) });
  await page.route('**/api/sources', failSources);
  await page.getByRole('button', { name: 'Refresh workspace data' }).click();
  await page.getByRole('alert').filter({ hasText: 'Source test failure' }).waitFor();
  await page.unroute('**/api/sources', failSources);
  await page.getByRole('button', { name: 'Retry' }).click();
  await page.getByRole('alert').filter({ hasText: 'Source test failure' }).waitFor({ state: 'hidden' });
  intentionalError = false;
  console.log('PASS source error is visible and retry recovers');

  await page.setViewportSize({ width: 375, height: 812 });
  await page.getByRole('button', { name: /Contacts/ }).click();
  await page.getByText('Maya Rao', { exact: true }).waitFor();
  await page.screenshot({ path: '/tmp/agent-workspace-mobile.png', fullPage: true });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'mobile overflow');
  await page.setViewportSize({ width: 768, height: 1024 });
  await page.screenshot({ path: '/tmp/agent-workspace-tablet.png', fullPage: true });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'tablet overflow');
  assert.deepEqual(errors, [], `browser console/page errors: ${errors.join('; ')}`);
  console.log(`QA_MARKERS slack=${slackText} email=${emailSubject} meeting=${meetingTitle} memory=${memoryText}`);
} finally {
  await browser.close();
}
