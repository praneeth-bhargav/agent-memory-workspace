import React, { useEffect, useRef, useState } from 'react';
import Persona from './Persona.jsx';
import Memories from './Memories.jsx';
import { sourceElementId } from './source-references.js';
import './source-forms.css';

const SECTIONS = [
  { id: 'mail', label: 'Mail', count: data => data.mail.length },
  { id: 'slack', label: 'Slack', count: data => data.slack_messages.length },
  { id: 'calendar', label: 'Calendar', count: data => data.calendar_events.length },
  { id: 'contacts', label: 'Contacts', count: data => data.contacts.length },
  { id: 'persona', label: 'Persona' },
  { id: 'memories', label: 'Memories' },
];
const TIMEZONE = 'Asia/Kolkata';
const formatter = new Intl.DateTimeFormat('en-IN', {
  timeZone: TIMEZONE, day: 'numeric', month: 'short', year: 'numeric', hour: 'numeric', minute: '2-digit',
});

function displayTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : `${formatter.format(date)} IST`;
}

function IncomingMail({ threads, onRefresh, asOf }) {
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [newThread, setNewThread] = useState(!threads.length);
  async function submit(event) {
    event.preventDefault();
    setError('');
    setSaving(true);
    const form = event.currentTarget;
    const fields = new FormData(form);
    try {
      const response = await fetch('/api/sources/incoming-mail', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...(newThread ? { subject: fields.get('subject') } : { thread_id: fields.get('thread_id') }),
          sender: fields.get('sender'), body: fields.get('body'), timestamp: fields.get('timestamp') }),
      });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Check the supplied details.');
      }
      form.reset();
      onRefresh();
    } catch (failure) { setError(failure.message || 'Could not add incoming mail.'); }
    finally { setSaving(false); }
  }
  return <details className="record"><summary>Add synthetic incoming email</summary>
    <form onSubmit={submit}>
      <label>Thread <select value={newThread ? 'new' : 'existing'} onChange={event => setNewThread(event.target.value === 'new')}>
        <option value="new">New thread</option>{threads.length > 0 && <option value="existing">Existing thread</option>}</select></label>
      {newThread ? <label>Subject <input name="subject" required maxLength="300" /></label> :
        <label>Existing thread <select name="thread_id" required>{threads.map(thread =>
          <option key={thread.thread_id} value={thread.thread_id}>{thread.subject}</option>)}</select></label>}
      <label>Sender (.example) <input name="sender" type="email" required /></label>
      <label>Message <textarea name="body" required maxLength="10000" /></label>
      <label>Timestamp (ISO with timezone) <input name="timestamp" defaultValue={asOf} required /></label>
      {error && <p role="alert">{error}</p>}
      <button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Add incoming email'}</button>
    </form>
  </details>;
}

function MailEditor({ message, threadId, onRefresh }) {
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  async function submit(event) {
    event.preventDefault();
    const fields = new FormData(event.currentTarget);
    const expected = { thread_id: threadId, sender: message.from ?? message.sender, body: message.body,
      timestamp: message.ts, unread: message.unread };
    setError('');
    setSaving(true);
    try {
      const response = await fetch(`/api/sources/incoming-mail/${encodeURIComponent(message.id)}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expected, message: { ...expected, sender: fields.get('sender'), body: fields.get('body'),
          timestamp: fields.get('timestamp'), unread: fields.has('unread') } }),
      });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Check the supplied details.');
      }
      onRefresh();
    } catch (failure) { setError(failure.message || 'Could not update email.'); }
    finally { setSaving(false); }
  }
  return <details><summary>Edit incoming email</summary><form onSubmit={submit} key={JSON.stringify(message)}>
    <label>Sender (.example) <input name="sender" type="email" defaultValue={message.from ?? message.sender} required /></label>
    <label>Message <textarea name="body" defaultValue={message.body} required maxLength="10000" /></label>
    <label>Timestamp (ISO with timezone) <input name="timestamp" defaultValue={message.ts} required /></label>
    <label><input type="checkbox" name="unread" defaultChecked={message.unread} /> Unread</label>
    {error && <p role="alert">{error}</p>}
    <button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Save email'}</button>
  </form></details>;
}

function ThreadSubjectEditor({ thread, onRefresh }) {
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  async function submit(event) {
    event.preventDefault();
    const subject = new FormData(event.currentTarget).get('subject');
    setError(''); setSaving(true);
    try {
      const response = await fetch(`/api/sources/mail-threads/${encodeURIComponent(thread.thread_id)}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expected_subject: thread.subject, subject }),
      });
      if (!response.ok) { const payload = await response.json(); throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Check the subject.'); }
      onRefresh();
    } catch (failure) { setError(failure.message || 'Could not update subject.'); }
    finally { setSaving(false); }
  }
  return <details><summary>Edit thread subject</summary><form onSubmit={submit} key={thread.subject}>
    <label>Subject for all messages in this thread <input name="subject" defaultValue={thread.subject} required maxLength="300" /></label>
    {error && <p role="alert">{error}</p>}<button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Save subject'}</button>
  </form></details>;
}

function Mail({ threads, userEmail, onRefresh, asOf }) {
  const sorted = threads.slice().sort((a, b) => (b.messages.at(-1)?.ts || '').localeCompare(a.messages.at(-1)?.ts || ''));
  return <div className="record-list"><IncomingMail threads={threads} onRefresh={onRefresh} asOf={asOf} />{sorted.map(thread => {
    const latest = thread.messages[thread.messages.length - 1];
    return <details className="record mail-thread" key={thread.thread_id}>
      <summary><span className="record-title">{thread.subject}</span>
        <span className="record-meta">{thread.messages.length} messages · {latest ? displayTime(latest.ts) : 'No messages'}</span>
        {latest && <span className="preview">{latest.body}</span>}</summary>
      <ThreadSubjectEditor thread={thread} onRefresh={onRefresh} />
      <div className="thread-messages">{thread.messages.map(message => <article className="thread-message" id={sourceElementId({ kind: 'mail', source_id: message.id })} key={message.id}>
        <div className="record-row"><strong>{message.from ?? message.sender}</strong><time dateTime={message.ts}>{displayTime(message.ts)}</time></div>
        <div className="record-meta">To: {message.to.join(', ')}{message.cc.length > 0 && ` · Cc: ${message.cc.join(', ')}`}</div>
        <p className="message-body">{message.body}</p>
        {(message.from ?? message.sender).toLowerCase() !== userEmail.toLowerCase() &&
          [...message.to, ...message.cc].some(address => address.toLowerCase() === userEmail.toLowerCase()) &&
          <MailEditor message={message} threadId={thread.thread_id} onRefresh={onRefresh} />}
      </article>)}</div>
    </details>;
  })}</div>;
}

function SlackChannelForm({ channel, onRefresh, onRename }) {
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  async function submit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const fields = new FormData(form);
    const value = { name: fields.get('name'), members: fields.get('members').split(',').map(item => item.trim()).filter(Boolean) };
    setError(''); setSaving(true);
    try {
      const response = await fetch(`/api/sources/slack-channels${channel ? `/${encodeURIComponent(channel.name)}` : ''}`, {
        method: channel ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(channel ? { expected: channel, channel: value } : value),
      });
      if (!response.ok) { const payload = await response.json(); throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Check channel details.'); }
      if (!channel) form.reset();
      else if (value.name !== channel.name) onRename(value.name);
      onRefresh();
    } catch (failure) { setError(failure.message || 'Could not save channel.'); }
    finally { setSaving(false); }
  }
  return <details className="record"><summary>{channel ? `Edit ${channel.name}` : 'Add synthetic Slack channel'}</summary>
    <form onSubmit={submit} key={JSON.stringify(channel)}>
      <label>Channel name <input name="name" defaultValue={channel?.name} placeholder="#team-name" required /></label>
      <label>Members (comma separated) <input name="members" defaultValue={channel?.members.join(', ') || ''} /></label>
      {error && <p role="alert">{error}</p>}
      <button type="submit" disabled={saving}>{saving ? 'Saving…' : channel ? 'Save channel' : 'Add channel'}</button>
    </form>
  </details>;
}

function SlackMessageForm({ message, channel, onRefresh, asOf }) {
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  async function submit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const fields = new FormData(form);
    const value = { channel, author: fields.get('author'), text: fields.get('text'), ts: fields.get('ts') };
    const expected = message && { channel: message.channel, author: message.author, text: message.text, ts: message.ts };
    setError(''); setSaving(true);
    try {
      const response = await fetch(`/api/sources/slack-messages${message ? `/${encodeURIComponent(message.id)}` : ''}`, {
        method: message ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(message ? { expected, message: value } : value),
      });
      if (!response.ok) { const payload = await response.json(); throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Check message details.'); }
      if (!message) form.reset();
      onRefresh();
    } catch (failure) { setError(failure.message || 'Could not save Slack message.'); }
    finally { setSaving(false); }
  }
  return <details className={message ? '' : 'record'}><summary>{message ? 'Edit message' : 'Add synthetic Slack message'}</summary>
    <form onSubmit={submit} key={JSON.stringify(message)}>
      <label>Author <input name="author" defaultValue={message?.author} required maxLength="100" /></label>
      <label>Message <textarea name="text" defaultValue={message?.text} required maxLength="10000" /></label>
      <label>Timestamp (ISO with timezone) <input name="ts" defaultValue={message?.ts || asOf} required /></label>
      {error && <p role="alert">{error}</p>}
      <button type="submit" disabled={saving}>{saving ? 'Saving…' : message ? 'Save message' : 'Add message'}</button>
    </form>
  </details>;
}

function Slack({ channels, messages, onRefresh, activeChannel, onSelectChannel, asOf }) {
  const channel = activeChannel || channels[0]?.name;
  const visible = messages.filter(message => message.channel === channel)
    .slice().sort((a, b) => a.ts.localeCompare(b.ts));
  const selectedChannel = channels.find(item => item.name === channel);
  return <div><SlackChannelForm onRefresh={onRefresh} /><div className="channel-list" aria-label="Slack channels">
    {channels.map(item => <button key={item.name} className="channel" aria-pressed={item.name === channel}
      onClick={() => onSelectChannel(item.name)}>{item.name}</button>)}
  </div>
  {selectedChannel && <><SlackChannelForm channel={selectedChannel} onRefresh={onRefresh} onRename={onSelectChannel} />
    <SlackMessageForm channel={channel} onRefresh={onRefresh} asOf={asOf} /></>}
  {!visible.length ? <p className="empty-state">No messages in this channel.</p> : <div className="record-list">
    {visible.map((message, index) => <article className="record" id={sourceElementId({ kind: 'slack', source_id: message.id })} key={`${message.channel}-${message.ts}-${index}`}>
      <div className="record-row"><strong>{message.author}</strong><time dateTime={message.ts}>{displayTime(message.ts)}</time></div>
      <p className="message-body">{message.text}</p>
      <SlackMessageForm message={message} channel={channel} onRefresh={onRefresh} />
    </article>)}
  </div>}</div>;
}

function CalendarForm({ event, people, onRefresh }) {
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  async function submit(submission) {
    submission.preventDefault();
    const form = submission.currentTarget;
    const fields = new FormData(form);
    const value = { person: fields.get('person'), title: fields.get('title'), start: fields.get('start'),
      end: fields.get('end'), all_day: fields.has('all_day') };
    const expected = event && { person: event.person, title: event.title, start: event.start, end: event.end, all_day: event.all_day };
    setError('');
    setSaving(true);
    try {
      const response = await fetch(`/api/sources/calendar-events${event ? `/${encodeURIComponent(event.id)}` : ''}`, {
        method: event ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(event ? { expected, event: value } : value),
      });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Check the supplied details and timezone.');
      }
      if (!event) form.reset();
      onRefresh();
    } catch (failure) { setError(failure.message || 'Could not save calendar event.'); }
    finally { setSaving(false); }
  }
  return <details className={event ? '' : 'record'}><summary>{event ? 'Edit event' : 'Add synthetic event'}</summary>
    <form onSubmit={submit} key={JSON.stringify(event)}>
      <label>Calendar <select name="person" defaultValue={event?.person} required>{people.map(person =>
        <option key={person.key} value={person.key}>{person.name} ({person.email})</option>)}</select></label>
      <label>Title <input name="title" defaultValue={event?.title} required maxLength="300" /></label>
      <label>Start (ISO with timezone) <input name="start" defaultValue={event?.start} placeholder="2026-05-18T10:00:00+05:30" required /></label>
      <label>End (ISO with timezone) <input name="end" defaultValue={event?.end} placeholder="2026-05-18T11:00:00+05:30" required /></label>
      <label><input type="checkbox" name="all_day" defaultChecked={event?.all_day} /> All day</label>
      {error && <p role="alert">{error}</p>}
      <button type="submit" disabled={saving || !people.length}>{saving ? 'Saving…' : event ? 'Save event' : 'Add event'}</button>
    </form>
  </details>;
}

function Calendar({ events, people, onRefresh }) {
  const sorted = events.slice().sort((a, b) => a.start.localeCompare(b.start));
  return <div className="record-list"><CalendarForm people={people} onRefresh={onRefresh} />{sorted.map((event, index) => {
    const person = people.find(item => item.key === event.person);
    return <article className="record event-record" id={sourceElementId({ kind: 'calendar', source_id: event.id })} key={`${event.person}-${event.start}-${event.title}-${index}`}>
      <div className="event-date">{new Date(event.start).toLocaleDateString('en-IN', { timeZone: TIMEZONE, day: '2-digit', month: 'short' })}</div>
      <div><strong className="record-title">{event.title}</strong><div className="record-meta">{person?.name || event.person}</div>
        <div className="record-meta">{displayTime(event.start)} – {displayTime(event.end)}{event.all_day && ' · All day'}</div>
        <CalendarForm event={event} people={people} onRefresh={onRefresh} /></div>
    </article>;
  })}</div>;
}

function ContactEditor({ contact, onRefresh }) {
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  async function submit(event) {
    event.preventDefault();
    setError('');
    setSaving(true);
    const fields = Object.fromEntries(new FormData(event.currentTarget).entries());
    const expected = { name: contact.name, email: contact.email, org: contact.org, role: contact.role };
    try {
      const response = await fetch(`/api/sources/contacts/${encodeURIComponent(contact.id)}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expected, contact: fields }),
      });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Check the supplied details.');
      }
      onRefresh();
    } catch (failure) { setError(failure.message || 'Could not update contact.'); }
    finally { setSaving(false); }
  }
  return <details><summary>Edit contact</summary><form onSubmit={submit} key={JSON.stringify(contact)}>
    <label>Name <input name="name" defaultValue={contact.name} required maxLength="200" /></label>
    <label>Email (.example) <input name="email" type="email" defaultValue={contact.email} required /></label>
    <label>Organisation <input name="org" defaultValue={contact.org} maxLength="200" /></label>
    <label>Role <input name="role" defaultValue={contact.role} maxLength="200" /></label>
    {error && <p role="alert">{error}</p>}
    <button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Save contact'}</button>
  </form></details>;
}

function Contacts({ contacts, onRefresh }) {
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  async function submit(event) {
    event.preventDefault();
    setError('');
    setSaving(true);
    const form = event.currentTarget;
    const fields = new FormData(form);
    try {
      const response = await fetch('/api/sources/contacts', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(Object.fromEntries(fields.entries())) });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Check the supplied details.');
      }
      form.reset();
      onRefresh();
    } catch (failure) { setError(failure.message || 'Could not add contact.'); }
    finally { setSaving(false); }
  }
  return <div className="record-list"><details className="record"><summary>Add synthetic contact</summary>
    <form onSubmit={submit}>
      <label>Name <input name="name" required maxLength="200" /></label>
      <label>Email (.example) <input name="email" type="email" required /></label>
      <label>Organisation <input name="org" maxLength="200" /></label>
      <label>Role <input name="role" maxLength="200" /></label>
      {error && <p role="alert">{error}</p>}
      <button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Add contact'}</button>
    </form>
  </details>{contacts.map(contact => <article className="record contact-record" id={sourceElementId({ kind: 'contact', source_id: contact.id })} key={contact.id}>
    <span className="avatar" aria-hidden="true">{contact.name.split(' ').map(part => part[0]).slice(0, 2).join('')}</span>
    <div><strong className="record-title">{contact.name}</strong><div className="record-meta">{contact.role} · {contact.org}</div>
      <div className="contact-email">{contact.email}</div><ContactEditor contact={contact} onRefresh={onRefresh} /></div>
  </article>)}</div>;
}

export default function Sources({ data, error, onRefresh, activeReference }) {
  const [section, setSection] = useState('mail');
  const [slackChannel, setSlackChannel] = useState('');
  const focusedReference = useRef('');
  const targetSection = activeReference?.kind === 'contact' ? 'contacts' : activeReference?.kind;

  useEffect(() => {
    if (activeReference) setSection(targetSection);
  }, [activeReference?.kind, activeReference?.source_id, targetSection]);

  useEffect(() => {
    if (activeReference?.kind !== 'slack' || !data) return;
    const message = data.slack_messages.find(item => item.id === activeReference.source_id);
    if (message) setSlackChannel(message.channel);
  }, [activeReference?.kind, activeReference?.source_id, data !== null]);

  useEffect(() => {
    if (!activeReference) { focusedReference.current = ''; return; }
    if (section !== targetSection) return;
    const key = `${activeReference.kind}:${activeReference.source_id}`;
    if (focusedReference.current === key) return;
    let target;
    let highlightTimer;
    const frame = window.requestAnimationFrame(() => {
      target = document.getElementById(sourceElementId(activeReference));
      if (!target) return;
      for (let parent = target.parentElement; parent; parent = parent.parentElement) {
        if (parent instanceof HTMLDetailsElement) parent.open = true;
      }
      focusedReference.current = key;
      target.classList.add('source-reference-target');
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      highlightTimer = window.setTimeout(() => target?.classList.remove('source-reference-target'), 2200);
    });
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(highlightTimer);
      target?.classList.remove('source-reference-target');
    };
  }, [activeReference, section, targetSection, data !== null, slackChannel]);

  return <section className="workspace-panel" aria-label="Source data">
    <div className="panel-heading"><div><p className="eyebrow">Shared mock data</p><h2>Workspace</h2></div>
      <button className="refresh" type="button" onClick={onRefresh} aria-label="Refresh workspace data">Refresh</button></div>
    <p className="panel-description">Manage synthetic contacts, incoming email, calendar events and Slack directly</p>
    <nav className="source-nav" aria-label="Source categories">{SECTIONS.map(item => <button key={item.id}
      type="button" aria-pressed={section === item.id} onClick={() => setSection(item.id)}>
      {item.label} {data && item.count && <span className="count">{item.count(data)}</span>}</button>)}</nav>
    {error && <p className="source-error" role="alert">{error} <button type="button" onClick={onRefresh}>Retry</button></p>}
    {section === 'persona' && <div className="source-content">
      <h3>Persona</h3><Persona />
    </div>}
    {section === 'memories' && <div className="source-content">
      <h3>Saved memories</h3><Memories />
    </div>}
    {!['persona', 'memories'].includes(section) && (!data ? !error && <p className="empty-state" role="status">Loading workspace data…</p> : <div className="source-content">
      <h3>{SECTIONS.find(item => item.id === section)?.label}</h3>
      {section === 'mail' && <Mail threads={data.mail} userEmail={data.user_email} onRefresh={onRefresh} asOf={data.as_of} />}
      {section === 'slack' && <Slack channels={data.slack_channels} messages={data.slack_messages} onRefresh={onRefresh} asOf={data.as_of}
        activeChannel={slackChannel} onSelectChannel={setSlackChannel} />}
      {section === 'calendar' && <Calendar events={data.calendar_events} people={data.calendar_people} onRefresh={onRefresh} />}
      {section === 'contacts' && <Contacts contacts={data.contacts} onRefresh={onRefresh} />}
    </div>)}
  </section>;
}
