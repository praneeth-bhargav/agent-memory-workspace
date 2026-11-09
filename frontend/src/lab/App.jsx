import React, { useCallback, useEffect, useRef, useState } from 'react';
import Chat from './Chat.jsx';
import Sources from './Sources.jsx';
import { parseSourceHash } from './source-references.js';
import './references.css';

const REFRESH_INTERVAL_MS = 5000;
const SOURCES_URL = '/api/sources';

export default function App() {
  const [sources, setSources] = useState(null);
  const [sourceError, setSourceError] = useState('');
  const [activeReference, setActiveReference] = useState(() => parseSourceHash(window.location.hash));
  const latestRequest = useRef(0);
  const mounted = useRef(false);

  const refresh = useCallback(async () => {
    const request = ++latestRequest.current;
    try {
      const response = await fetch(SOURCES_URL, { cache: 'no-store' });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail || `Workspace request failed (${response.status}).`);
      }
      const payload = await response.json();
      if (mounted.current && request === latestRequest.current) {
        setSources(payload);
        setSourceError('');
      }
    } catch (error) {
      if (mounted.current && request === latestRequest.current) {
        setSourceError(error.message || 'Could not load workspace data.');
      }
    }
  }, []);

  useEffect(() => {
    const onHashChange = () => setActiveReference(parseSourceHash(window.location.hash));
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    mounted.current = true;
    refresh();
    const timer = window.setInterval(() => { if (!document.hidden) refresh(); }, REFRESH_INTERVAL_MS);
    const onVisibilityChange = () => { if (!document.hidden) refresh(); };
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => { mounted.current = false; window.clearInterval(timer); document.removeEventListener('visibilitychange', onVisibilityChange); };
  }, [refresh]);

  return <div className="app-shell">
    <header className="app-header"><div className="brand"><span className="brand-mark" aria-hidden="true">✳</span>
      <div><strong>Agent</strong><span>Memory workspace</span></div></div>
      <div className="header-meta"><span className="scenario-label">Scenario · 18 May 2026</span>
        <span className="environment">Synthetic workspace</span></div></header>
    <main className="app-main"><Chat onSourceChange={refresh} />
      <Sources data={sources} error={sourceError} onRefresh={refresh} activeReference={activeReference} /></main>
  </div>;
}
