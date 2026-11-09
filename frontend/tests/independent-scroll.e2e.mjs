import assert from 'node:assert/strict';
import { chromium } from 'playwright-core';

const base = process.env.WORKSPACE_QA_URL || 'http://127.0.0.1:5175/memory-lab.html';
const browser = await chromium.launch({ headless: true,
  executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
try {
  await page.route('**/api/chats', route => route.fulfill({ json: [{ id: 'scroll-qa', title: 'Long chat' }] }));
  await page.route('**/api/chats/scroll-qa', route => route.fulfill({ json: {
    pending: null, events: Array.from({ length: 1000 }, (_, index) => ({
      kind: index % 2 ? 'answer' : 'user', data: `Message ${index}: synthetic scroll check.`, message_id: `m${index}`,
    })),
  } }));
  await page.goto(base);
  await page.locator('.chat-message').last().waitFor();
  await page.screenshot({ path: `/tmp/agent-scroll-${process.env.SCROLL_PHASE || 'check'}.png` });
  const state = await page.evaluate(() => {
    const chat = document.querySelector('.conversation');
    const workspace = document.querySelector('.workspace-panel');
    return { pageScroll: window.scrollY, chatScroll: chat.scrollTop,
      chatHeight: chat.clientHeight, chatTotal: chat.scrollHeight,
      workspaceTop: workspace.getBoundingClientRect().top,
      workspaceBottom: workspace.getBoundingClientRect().bottom, viewport: innerHeight };
  });
  assert.equal(state.pageScroll, 0, 'opening a long chat must not scroll the page');
  assert(state.chatTotal > state.chatHeight && state.chatScroll > 0, 'chat must scroll internally to latest message');
  assert(state.workspaceTop >= 0 && state.workspaceBottom <= state.viewport, 'workspace must remain visible');
  const rightScroll = await page.locator('.source-content').evaluate(node => node.scrollTop);
  await page.locator('.conversation').evaluate(node => { node.scrollTop = 0; });
  assert.equal(await page.locator('.source-content').evaluate(node => node.scrollTop), rightScroll);
  console.log('PASS: 1000-message chat scroll is independent; workspace remains visible');
} finally {
  await browser.close();
}
