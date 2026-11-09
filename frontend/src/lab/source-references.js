const SOURCE_KINDS = new Set(['mail', 'slack', 'calendar', 'contact']);

export function sourceElementId(reference) {
  return `source-${reference.kind}-${encodeURIComponent(reference.source_id)}`;
}

export function parseSourceHash(hash) {
  const match = /^#source\/(mail|slack|calendar|contact)\/([^/]+)$/.exec(hash);
  if (!match) return null;
  try { return { kind: match[1], source_id: decodeURIComponent(match[2]) }; }
  catch { return null; }
}

export function parseChatHash(hash) {
  const match = /^#chat\/([^/]+)\/(m[0-9]+)$/.exec(hash);
  if (!match) return null;
  try { return { chat_id: decodeURIComponent(match[1]), message_id: match[2] }; }
  catch { return null; }
}

export function sourceHash(reference) {
  if (reference.kind === 'chat' && reference.chat_id && /^m[0-9]+$/.test(reference.source_id)) {
    return `#chat/${encodeURIComponent(reference.chat_id)}/${reference.source_id}`;
  }
  if (!SOURCE_KINDS.has(reference.kind) || !reference.source_id) return '#';
  return `#source/${reference.kind}/${encodeURIComponent(reference.source_id)}`;
}
