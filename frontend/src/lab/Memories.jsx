import React, { useCallback, useEffect, useState } from 'react';
import { sourceHash } from './source-references.js';

const MEMORIES_URL = '/api/memories';

function errorText(payload, fallback) {
  return typeof payload?.detail === 'string' ? payload.detail : fallback;
}

export default function Memories() {
  const [memories, setMemories] = useState([]);
  const [error, setError] = useState('');
  const [editing, setEditing] = useState('');
  const [saving, setSaving] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(MEMORIES_URL, { cache: 'no-store' });
      if (!response.ok) throw new Error(errorText(await response.json(), 'Could not load memories.'));
      setMemories(await response.json());
      setError('');
    } catch (failure) {
      setError(failure.message || 'Could not load memories.');
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  async function save(event, memory) {
    event.preventDefault();
    const text = new FormData(event.currentTarget).get('text');
    setSaving(true);
    setError('');
    try {
      const response = await fetch(`${MEMORIES_URL}/${encodeURIComponent(memory.id)}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expected_text: memory.text, text }),
      });
      if (!response.ok) throw new Error(errorText(await response.json(), 'Could not update memory.'));
      setEditing('');
      await refresh();
    } catch (failure) {
      setError(failure.message || 'Could not update memory.');
    } finally {
      setSaving(false);
    }
  }

  async function forget(memory) {
    if (!window.confirm('Forget this memory? The original chat and source records will remain in history.')) return;
    setError('');
    try {
      const response = await fetch(`${MEMORIES_URL}/${encodeURIComponent(memory.id)}`, {
        method: 'DELETE', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expected_text: memory.text }),
      });
      if (!response.ok) throw new Error(errorText(await response.json(), 'Could not forget memory.'));
      await refresh();
    } catch (failure) {
      setError(failure.message || 'Could not forget memory.');
    }
  }

  if (error && memories.length === 0) return <div className="record-list"><p className="source-error" role="alert">{error}
    <button type="button" onClick={refresh}>Retry</button></p></div>;

  return <div className="record-list" aria-label="Saved memories">
    {error && <p role="alert" className="source-error">{error}</p>}
    {memories.length === 0 && !error && <p className="empty-state">No saved memories yet.</p>}
    {memories.map(memory => <article className="record memory-record" key={memory.id}>
      <p className="memory-text">{memory.text}</p>
      {memory.provenance.length > 0 && <ul className="memory-provenance" aria-label="Memory sources">
        {memory.provenance.map(source => <li key={`${memory.id}:${source.kind}:${source.source_id}`}>
          <a href={sourceHash(source)}>{source.kind === 'chat' ? `Chat message ${source.source_id}` : `${source.kind} source ${source.source_id}`}</a>
        </li>)}
      </ul>}
      {memory.is_persona ? <p className="record-meta">Persona is fixed and read-only.</p> : <div className="memory-actions">
        <button type="button" onClick={() => setEditing(editing === memory.id ? '' : memory.id)}>
          {editing === memory.id ? 'Cancel edit' : 'Edit'}
        </button>
        <button className="forget-memory" type="button" onClick={() => forget(memory)}>Forget</button>
      </div>}
      {editing === memory.id && <form className="memory-edit" onSubmit={event => save(event, memory)} key={memory.text}>
        <label htmlFor={`memory-text-${memory.id}`}>Memory text</label>
        <textarea id={`memory-text-${memory.id}`} name="text" defaultValue={memory.text} required maxLength="10000" />
        <button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Save memory'}</button>
      </form>}
    </article>)}
  </div>;
}
