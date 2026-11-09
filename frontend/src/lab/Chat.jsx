import React, { useEffect, useRef, useState } from 'react';
import './chat-history.css';
import { parseChatHash, sourceHash } from './source-references.js';

const TOOL_LABELS = {
  memory_read: 'Reading memory', memory_change: 'Updating memory', remember_source_fact: 'Evaluating source memory',
  source_memory_review: 'Reviewing source facts for memory',
  query_sources_sql: 'Checking source records', get_action_context: 'Getting action context', search_contacts: 'Searching contacts',
  search_mail: 'Searching mail', read_mail_thread: 'Reading mail', search_slack: 'Searching Slack',
  read_calendar: 'Reading calendar', list_calendar_people: 'Checking calendars',
  find_available_slots: 'Finding available slots', prepare_mock_email: 'Preparing mock email',
  prepare_mock_slack_post: 'Preparing mock Slack post', prepare_mock_booking: 'Preparing mock booking',
};
const STREAM_URL = '/api/chat/stream';
const CHATS_URL = '/api/chats';
const SELECTED_CHAT = 'agent-selected-chat';

function restoreEvents(events) {
  return events.reduce((items, event) => {
    if (event.kind === 'user') return [...items, { kind: 'user', text: event.data, messageId: event.message_id }];
    if (event.kind === 'answer') return [...items, { kind: 'assistant', text: event.data, references: event.references || [] }];
    if (event.kind === 'error') return [...items, { kind: 'error', text: event.data }];
    if (event.kind === 'tool_start') return [...items, { kind: 'tool', id: event.tool_call_id, name: event.name,
      input: event.data, result: null, status: '' }];
    if (event.kind === 'tool_result') return items.map(item => item.kind === 'tool' && item.id === event.tool_call_id
      ? { ...item, result: event.data, status: event.status } : item);
    return items;
  }, []);
}

function pretty(value) {
  try { return JSON.stringify(JSON.parse(value), null, 2); }
  catch { return value; }
}

function Activity({ item }) {
  let label = TOOL_LABELS[item.name] || item.name;
  if (item.name === 'get_action_context') {
    try {
      const action = JSON.parse(item.input).action;
      if (['email', 'slack', 'calendar'].includes(action)) label = `Getting ${action} context`;
    } catch { /* Keep the generic label for malformed input. */ }
  }
  return <details className="activity"><summary>{label} <span className="activity-status">
    {item.result === null ? 'Working…' : item.status === 'error' ? 'Failed' : 'Done'}</span></summary>
    <div className="activity-detail"><strong>Input</strong><pre>{pretty(item.input)}</pre>
      {item.result !== null && <><strong>Result</strong><pre>{pretty(item.result)}</pre></>}</div>
  </details>;
}

export default function Chat({ onSourceChange }) {
  const [items, setItems] = useState([]);
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [chats, setChats] = useState([]);
  const [chatId, setChatId] = useState('');
  const [chatError, setChatError] = useState('');
  const [pending, setPending] = useState(null);
  const [chatReference, setChatReference] = useState(() => parseChatHash(window.location.hash));
  const drafts = useRef({});
  const endRef = useRef(null);

  useEffect(() => {
    if (chatReference && chatReference.chat_id === chatId) {
      const target = document.getElementById(`chat-message-${chatReference.message_id}`);
      if (target) { target.scrollIntoView({ block: 'center' }); target.focus({ preventScroll: true }); }
    } else endRef.current?.scrollIntoView({ block: 'nearest' });
  }, [items, chatId, chatReference]);

  useEffect(() => {
    const onHashChange = () => setChatReference(parseChatHash(window.location.hash));
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    if (chatReference && chats.some(chat => chat.id === chatReference.chat_id)) setChatId(chatReference.chat_id);
  }, [chatReference, chats]);

  useEffect(() => {
    let active = true;
    async function initialize() {
      try {
        const response = await fetch(CHATS_URL);
        if (!response.ok) throw new Error('Could not load chats.');
        let available = await response.json();
        if (!available.length) {
          const created = await fetch(CHATS_URL, { method: 'POST' });
          if (!created.ok) throw new Error('Could not create a chat.');
          available = [await created.json()];
        }
        if (!active) return;
        const saved = localStorage.getItem(SELECTED_CHAT);
        setChats(available);
        const reference = parseChatHash(window.location.hash);
        setChatId(reference && available.some(chat => chat.id === reference.chat_id) ? reference.chat_id
          : available.some(chat => chat.id === saved) ? saved : available[0].id);
      } catch (error) {
        if (active) { setChatError(error.message || 'Could not load chats.'); setLoading(false); }
      }
    }
    initialize();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!chatId) return;
    const controller = new AbortController();
    setLoading(true);
    setItems([]);
    setPending(null);
    setChatError('');
    localStorage.setItem(SELECTED_CHAT, chatId);
    async function load() {
      try {
        const response = await fetch(`${CHATS_URL}/${encodeURIComponent(chatId)}`, { signal: controller.signal });
        if (!response.ok) throw new Error('Could not load this chat.');
        const history = await response.json();
        setItems(restoreEvents(history.events));
        setPending(history.pending);
      } catch (error) {
        if (error.name !== 'AbortError') setChatError(error.message || 'Could not load this chat.');
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    load();
    return () => controller.abort();
  }, [chatId]);

  function switchChat(id) {
    if (chatReference) {
      window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`);
      setChatReference(null);
    }
    drafts.current = { ...drafts.current, [chatId]: text };
    setText(drafts.current[id] || '');
    setChatId(id);
  }

  async function newChat() {
    setLoading(true);
    setChatError('');
    try {
      const response = await fetch(CHATS_URL, { method: 'POST' });
      if (!response.ok) throw new Error('Could not create a chat.');
      const chat = await response.json();
      setChats(previous => [chat, ...previous]);
      switchChat(chat.id);
    } catch (error) {
      setChatError(error.message || 'Could not create a chat.');
      setLoading(false);
    }
  }

  async function send(event) {
    event.preventDefault();
    const message = text.trim();
    if (!message || busy || loading || !chatId || chatError) return;
    if (chatReference) {
      window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}`);
      setChatReference(null);
    }
    setBusy(true);
    setText('');
    setItems(previous => [...previous, { kind: 'user', text: message }]);
    setChats(previous => previous.map(chat => chat.id === chatId && chat.title === 'New chat'
      ? { ...chat, title: message.slice(0, 60) } : chat));
    let answered = false;
    try {
      const response = await fetch(STREAM_URL, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: message, chat_id: chatId }),
      });
      if (!response.ok || !response.body) throw new Error(`Chat request failed (${response.status}).`);
      setPending(null);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          const update = JSON.parse(line);
          if (update.kind === 'user_id') {
            setItems(previous => previous.map((item, index) => index === previous.length - 1 && item.kind === 'user'
              ? { ...item, messageId: update.message_id } : item));
          } else if (update.kind === 'tool_start') {
            setItems(previous => [...previous, { kind: 'tool', id: update.tool_call_id, name: update.name,
              input: update.data, result: null, status: '' }]);
          } else if (update.kind === 'tool_result') {
            setItems(previous => previous.map(item => item.kind === 'tool' && item.id === update.tool_call_id
              ? { ...item, result: update.data, status: update.status } : item));
            onSourceChange();
          } else if (update.kind === 'answer') {
            answered = true;
            setItems(previous => [...previous, { kind: 'assistant', text: update.data, references: update.references || [] }]);
            onSourceChange();
          } else if (update.kind === 'error') {
            throw new Error(update.data);
          }
        }
        if (done) break;
      }
      if (!answered) throw new Error('The response ended before the assistant finished.');
    } catch (error) {
      setItems(previous => [...previous, { kind: 'error', text: error.message || 'Could not reach the assistant.' }]);
      onSourceChange();
    } finally {
      setBusy(false);
    }
  }

  const userMessageIds = new Set(items.filter(item => item.kind === 'user' && item.messageId).map(item => item.messageId));

  function messageText(item) {
    if (item.kind !== 'assistant') return item.text;
    return item.text.split(/(\[chat:m[0-9]+\])/g).map((part, index) => {
      const match = /^\[chat:(m[0-9]+)\]$/.exec(part);
      if (!match || !userMessageIds.has(match[1])) return part;
      const href = sourceHash({ kind: 'chat', chat_id: chatId, source_id: match[1] });
      return <a key={index} href={href} onClick={event => {
        if (busy) event.preventDefault();
        else if (window.location.hash === href) {
          const target = document.getElementById(`chat-message-${match[1]}`);
          target?.scrollIntoView({ block: 'center' });
          target?.focus({ preventScroll: true });
        }
      }}>{part}</a>;
    });
  }

  return <section className="chat-panel" aria-label="Agent chat">
    <div className="panel-heading"><div><p className="eyebrow">Standalone agent</p><h2>Conversation</h2></div>
      <span className="status-pill"><span aria-hidden="true">●</span> Synthetic</span></div>
    <div className="chat-switcher"><label htmlFor="chat-select">Chats</label>
      <select id="chat-select" value={chatId} disabled={busy || loading || !chats.length}
        onChange={event => switchChat(event.target.value)}>{chats.map(chat => <option key={chat.id} value={chat.id}>{chat.title}</option>)}</select>
      <button type="button" disabled={busy || loading} onClick={newChat}>New chat</button></div>
    {chatError && <p className="chat-load-error" role="alert">{chatError} Reload the page to retry.</p>}
    {pending && <details className="pending-chat"><summary>Pending mock {pending.kind} — review before confirming</summary>
      <pre>{JSON.stringify(pending.payload, null, 2)}</pre></details>}
    <div className="conversation" aria-live="polite" aria-relevant="additions text">
      {!items.length && <div className="chat-welcome"><div className="welcome-mark" aria-hidden="true">✳</div>
        <h3>What can I help with?</h3><p>Ask about your workspace or request a mock action.</p>
        <p className="suggestion">Try “Summarize the Acme proposal thread”</p></div>}
      {items.map((item, index) => item.kind === 'tool' ? <Activity key={index} item={item} /> :
        <article key={index} id={item.messageId ? `chat-message-${item.messageId}` : undefined}
          tabIndex={item.messageId ? -1 : undefined}
          className={`chat-message ${item.kind}${chatReference?.message_id === item.messageId && chatReference?.chat_id === chatId ? ' source-reference-target' : ''}`}
          role={item.kind === 'error' ? 'alert' : undefined}>
          <strong>{item.kind === 'user' ? 'You' : item.kind === 'error' ? 'Request failed' : 'Agent'}</strong>
          <p>{messageText(item)}</p>
          {item.kind === 'assistant' && item.references?.length > 0 && <nav className="message-references" aria-label="Answer sources">
            <strong>Sources</strong>{item.references.map(reference => <a key={`${reference.kind}:${reference.source_id}`}
              href={sourceHash(reference)} onClick={event => { if (busy) event.preventDefault(); }}>{reference.label}</a>)}
          </nav>}
        </article>)}
      {busy && <p className="working" role="status">Agent is working…</p>}<div ref={endRef} />
    </div>
    <form className="composer" onSubmit={send}>
      <label className="sr-only" htmlFor="chat-message">Your message</label>
      <textarea id="chat-message" value={text} onChange={event => setText(event.target.value)}
        onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
          event.preventDefault(); event.currentTarget.form.requestSubmit();
        } }} disabled={busy || loading || !!chatError} placeholder="Ask about mail, Slack, or your calendar…" rows="2" maxLength="10000" />
      <div className="composer-footer"><span>Enter to send · Shift + Enter for a new line</span>
        <button type="submit" disabled={busy || loading || !!chatError || !text.trim()}>{busy ? 'Working…' : 'Send'}</button></div>
    </form>
    <p className="safety-note">Mock actions require confirmation. No real accounts are contacted.</p>
  </section>;
}
