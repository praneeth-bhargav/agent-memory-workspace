import assert from 'node:assert/strict';
import { chromium } from 'playwright-core';

const base = process.env.WORKSPACE_QA_URL;
if (!base || !['127.0.0.1', 'localhost'].includes(new URL(base).hostname)) throw new Error('Use an isolated local test server.');
const chrome = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser = await chromium.launch({ headless: true, executablePath: chrome });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const marker = `CHAT-QA-${Date.now()}`;
async function send(text) {
  const input = page.getByLabel('Your message');
  await input.waitFor({ state: 'visible' });
  await page.waitForFunction(() => !document.querySelector('.composer textarea').disabled, null, { timeout: 120000 });
  await input.fill(text);
  await page.getByRole('button', { name: 'Send', exact: true }).click();
  await page.waitForFunction(() => !document.querySelector('.composer textarea').disabled, null, { timeout: 120000 });
}
try {
  await page.goto(`${base}/workspace`);
  await page.getByRole('button', { name: 'New chat', exact: true }).waitFor();
  await page.waitForFunction(() => !document.querySelector('#chat-select').disabled, null, { timeout: 120000 });
  const selector = page.getByLabel('Chats');
  const first = await selector.inputValue();
  await send('Remember that my preferred QA color is teal.');
  await page.getByText('Updating memory', { exact: false }).last().waitFor();
  await send(`Prepare a mock email to arjun@kestrelworks.example with subject ${marker} and body Hello. Ask for confirmation.`);
  await page.getByText(/Prepared mock email; nothing has been sent/).waitFor({ timeout: 120000 });
  await page.screenshot({ path: '/tmp/agent-chat-before-new.png' });
  await page.getByRole('button', { name: 'New chat', exact: true }).click();
  await page.waitForFunction(old => document.querySelector('#chat-select').value !== old && !document.querySelector('#chat-select').disabled, first);
  const second = await selector.inputValue();
  assert.notEqual(first, second);
  assert.equal(await page.getByText(marker, { exact: false }).count(), 0);
  await send('What is my preferred QA color?');
  await page.getByText('Reading memory', { exact: false }).last().waitFor();
  await page.locator('.chat-message.assistant').filter({ hasText: /teal/i }).last().waitFor();
  await send('yes');
  const snapshot = await (await fetch(`${base}/api/sources`)).json();
  assert(!snapshot.mail.some(thread => thread.subject === marker), 'new chat must not confirm the old draft');
  await selector.selectOption(first);
  await page.locator('.chat-message.user').filter({ hasText: marker }).waitFor();
  await page.reload();
  await page.waitForFunction(old => document.querySelector('#chat-select')?.value === old && !document.querySelector('#chat-select').disabled, first);
  await page.locator('.chat-message.user').filter({ hasText: marker }).waitFor();
  await send('yes');
  await page.getByText(/Confirmed mock email/).waitFor();
  const after = await (await fetch(`${base}/api/sources`)).json();
  assert(after.mail.some(thread => thread.subject === marker));
  await page.screenshot({ path: '/tmp/agent-chat-after-reopen.png' });
  await page.setViewportSize({ width: 375, height: 812 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  console.log('PASS new chat, isolated pending confirmation, reopen, refresh, and mobile layout');
} finally {
  await browser.close();
}
