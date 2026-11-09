import assert from 'node:assert/strict';
import { chromium } from 'playwright-core';

const base = process.env.WORKSPACE_QA_URL;
if (!base || !['127.0.0.1', 'localhost'].includes(new URL(base).hostname)) throw new Error('Use an isolated local test server.');
const chrome = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser = await chromium.launch({ headless: true, executablePath: chrome });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
try {
  await page.goto(`${base}${process.env.WORKSPACE_QA_PAGE || '/workspace'}`);
  await page.getByRole('button', { name: 'Persona', exact: true }).click();
  const input = page.getByLabel('Persona memory');
  await input.waitFor();
  assert.match(await input.inputValue(), /Rohan Menon/);
  await page.screenshot({ path: '/tmp/persona-editor-before.png' });
  const updated = `Rohan runs QA Orchard ${Date.now()}.`;
  await input.fill(updated);
  await page.getByRole('button', { name: 'Save persona' }).click();
  await page.getByText('Persona saved.').waitFor();
  await page.reload();
  await page.getByRole('button', { name: 'Persona', exact: true }).click();
  await page.waitForFunction(text => document.querySelector('#persona-memory')?.value === text, updated);
  assert.equal((await (await fetch(`${base}/api/persona`)).json()).text, updated);
  await page.screenshot({ path: '/tmp/persona-editor-after.png' });
  const other = await fetch(`${base}/api/persona`, { method: 'PUT', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: 'Another edit.', expected_text: updated }) });
  assert.equal(other.status, 200);
  await input.fill('Unsaved draft.');
  await page.getByRole('button', { name: 'Save persona' }).click();
  await page.getByRole('alert').filter({ hasText: 'Persona changed since it was loaded' }).waitFor();
  assert.equal(await input.inputValue(), 'Unsaved draft.');
  assert.equal((await (await fetch(`${base}/api/persona`)).json()).text, 'Another edit.');
  await page.setViewportSize({ width: 375, height: 812 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  console.log('PASS persona editor save, reload and mobile layout');
} finally {
  await browser.close();
}
