import React, { useEffect, useState } from 'react';
import './persona.css';

const PERSONA_URL = '/api/persona';

export default function Persona() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch(PERSONA_URL, { signal: controller.signal });
        if (!response.ok) throw new Error(`Could not load persona (${response.status}).`);
        const persona = await response.json();
        setText(persona.text || '');
      } catch (failure) {
        if (failure.name !== 'AbortError') setError(failure.message || 'Could not load persona.');
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    load();
    return () => controller.abort();
  }, []);

  return <div className="persona-editor">
    <label htmlFor="persona-memory">Persona memory (read-only)</label>
    <textarea id="persona-memory" value={text} readOnly />
    {loading && <p role="status">Loading persona…</p>}
    {error && <p className="persona-error" role="alert">{error}</p>}
  </div>;
}
