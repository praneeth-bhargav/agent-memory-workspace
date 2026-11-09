"""FTS indexes for synthetic source message search."""
import re
import sqlite3

MAX_TERMS = 20
MAX_QUERY_LENGTH = 500
MAX_RESULTS = 50

SCHEMA = '''
CREATE VIRTUAL TABLE IF NOT EXISTS slack_fts USING fts5(channel, author, text, tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS mail_fts USING fts5(subject, sender, body, recipients, tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS slack_fts_insert AFTER INSERT ON slack_messages BEGIN
  INSERT INTO slack_fts(rowid, channel, author, text) VALUES (new.rowid, new.channel_id, new.author_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS slack_fts_delete AFTER DELETE ON slack_messages BEGIN
  DELETE FROM slack_fts WHERE rowid=old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS slack_fts_update AFTER UPDATE ON slack_messages BEGIN
  DELETE FROM slack_fts WHERE rowid=old.rowid;
  INSERT INTO slack_fts(rowid, channel, author, text) VALUES (new.rowid, new.channel_id, new.author_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS mail_fts_insert AFTER INSERT ON mail_messages BEGIN
  INSERT INTO mail_fts(rowid,subject,sender,body,recipients)
  SELECT new.rowid,t.subject,new.sender,new.body,'' FROM mail_threads t WHERE t.id=new.thread_id;
END;
CREATE TRIGGER IF NOT EXISTS mail_fts_delete AFTER DELETE ON mail_messages BEGIN
  DELETE FROM mail_fts WHERE rowid=old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS mail_fts_update AFTER UPDATE ON mail_messages BEGIN
  DELETE FROM mail_fts WHERE rowid=old.rowid;
  INSERT INTO mail_fts(rowid,subject,sender,body,recipients)
  SELECT new.rowid,t.subject,new.sender,new.body,coalesce((SELECT group_concat(address,' ') FROM mail_recipients WHERE message_id=new.id),'')
  FROM mail_threads t WHERE t.id=new.thread_id;
END;
CREATE TRIGGER IF NOT EXISTS mail_fts_subject AFTER UPDATE OF subject ON mail_threads BEGIN
  UPDATE mail_fts SET subject=new.subject WHERE rowid IN (SELECT rowid FROM mail_messages WHERE thread_id=new.id);
END;
CREATE TRIGGER IF NOT EXISTS mail_fts_recipient_insert AFTER INSERT ON mail_recipients BEGIN
  UPDATE mail_fts SET recipients=(SELECT coalesce(group_concat(address,' '),'') FROM mail_recipients WHERE message_id=new.message_id)
  WHERE rowid=(SELECT rowid FROM mail_messages WHERE id=new.message_id);
END;
CREATE TRIGGER IF NOT EXISTS mail_fts_recipient_delete AFTER DELETE ON mail_recipients BEGIN
  UPDATE mail_fts SET recipients=(SELECT coalesce(group_concat(address,' '),'') FROM mail_recipients WHERE message_id=old.message_id)
  WHERE rowid=(SELECT rowid FROM mail_messages WHERE id=old.message_id);
END;
CREATE TRIGGER IF NOT EXISTS mail_fts_recipient_update AFTER UPDATE ON mail_recipients BEGIN
  UPDATE mail_fts SET recipients=(SELECT coalesce(group_concat(address,' '),'') FROM mail_recipients WHERE message_id=old.message_id)
  WHERE rowid=(SELECT rowid FROM mail_messages WHERE id=old.message_id);
  UPDATE mail_fts SET recipients=(SELECT coalesce(group_concat(address,' '),'') FROM mail_recipients WHERE message_id=new.message_id)
  WHERE rowid=(SELECT rowid FROM mail_messages WHERE id=new.message_id);
END;
'''


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    with conn:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute("SELECT 1 FROM source_migrations WHERE name='source_fts_v1'").fetchone():
            return
        conn.execute('DELETE FROM slack_fts')
        conn.execute('INSERT INTO slack_fts(rowid,channel,author,text) SELECT rowid,channel_id,author_id,text FROM slack_messages')
        conn.execute('DELETE FROM mail_fts')
        conn.execute('''INSERT INTO mail_fts(rowid,subject,sender,body,recipients)
            SELECT m.rowid,t.subject,m.sender,m.body,coalesce((SELECT group_concat(address,' ') FROM mail_recipients r WHERE r.message_id=m.id),'')
            FROM mail_messages m JOIN mail_threads t ON t.id=m.thread_id''')
        conn.execute("INSERT INTO source_migrations(name) VALUES ('source_fts_v1')")


def expression(query: str) -> str:
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError('Search query is too long.')
    terms = re.findall(r'\w+', query, flags=re.UNICODE)
    if len(terms) > MAX_TERMS:
        raise ValueError('Search query has too many terms.')
    return ' AND '.join('"' + term.replace('"', '""') + '"' for term in terms)
