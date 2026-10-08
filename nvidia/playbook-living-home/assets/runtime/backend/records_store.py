"""Local records for one new household; contains no sample seeding."""
from __future__ import annotations
import datetime as dt
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value):
        raise ValueError("Invalid record identifier")
    return value

class PropertyStore:
    def __init__(self, root, property_id):
        self.property_id = identifier(property_id)
        self.root = Path(root)
        self.db_path = self.root / "property.sqlite"

    @contextmanager
    def connect(self):
        self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path, timeout=20)
        db.row_factory = sqlite3.Row
        db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY, asset_id TEXT, kind TEXT, title TEXT, body TEXT,
                uri TEXT, updated_at TEXT, is_demo INTEGER, hash TEXT, metadata TEXT,
                embedding TEXT
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY, kind TEXT, created_at TEXT, payload TEXT
            );
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS deliveries (
                id TEXT PRIMARY KEY, artifact_id TEXT, destination TEXT,
                state TEXT, receipt TEXT, updated_at TEXT
            );
        """)
        try:
            with db:
                yield db
        finally:
            db.close()

    def setting(self, key, value=None):
        with self.connect() as db:
            if value is not None:
                db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json.dumps(value)))
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_source(self, source):
        sid = identifier(source["source_id"])
        body = source["body"]
        title = source["title"]
        digest = hashlib.sha256((title + "\n" + body).encode()).hexdigest()
        metadata = {key: value for key, value in source.items() if key not in {"body", "embedding"}}
        metadata.update(property_id=self.property_id, content_hash=digest)
        with self.connect() as db:
            old = db.execute("SELECT hash FROM sources WHERE id=?", (sid,)).fetchone()
            changed = old is None or old[0] != digest
            db.execute("""INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET asset_id=excluded.asset_id,kind=excluded.kind,
                title=excluded.title,body=excluded.body,uri=excluded.uri,updated_at=excluded.updated_at,
                is_demo=excluded.is_demo,hash=excluded.hash,metadata=excluded.metadata,
                embedding=CASE WHEN sources.hash=excluded.hash THEN sources.embedding ELSE NULL END""",
                (sid, source.get("asset_id"), source["source_kind"], title, body,
                 source["source_uri"], source.get("source_updated_at", now()),
                 int(source.get("is_demo", False)), digest, json.dumps(metadata), None))
        self._write_note(sid, title, body, metadata)
        return changed

    def _write_note(self, sid, title, body, metadata):
        folder = self.root / "vault" / "Generated"
        folder.mkdir(parents=True, exist_ok=True)
        note = (f"# {title}\n\n" + ""
                + f"Source: {metadata['source_uri']}\n\n" + body + "\n\n"
                + f"<!-- source_id: {sid}; sha256: {metadata['content_hash']} -->\n")
        if "curated_attribution" in metadata:
            note += ("\n## Curated annotation (separate from source text)\n\n```json\n"
                     + json.dumps(metadata["curated_attribution"], indent=2)
                     + "\n```\n")
        target = folder / (sid + ".md")
        temp = target.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temp.write_text(note, encoding="utf-8")
        temp.replace(target)

    def mark_source_deleted(self, sid, reason="Source removed or access revoked"):
        with self.connect() as db:
            row = db.execute("SELECT metadata FROM sources WHERE id=?", (identifier(sid),)).fetchone()
            if row:
                metadata = json.loads(row[0])
                metadata.update(is_deleted=True, deleted_reason=reason, deleted_at=now())
                db.execute("UPDATE sources SET metadata=?,embedding=NULL WHERE id=?", (json.dumps(metadata), sid))
        note = self.root / "vault" / "Generated" / (identifier(sid) + ".md")
        if note.exists():
            archive = self.root / "vault" / "Removed"
            archive.mkdir(parents=True, exist_ok=True)
            note.replace(archive / note.name)

    def sources(self, asset_id=None, include_deleted=False):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM sources" + (" WHERE asset_id=?" if asset_id else ""),
                              (asset_id,) if asset_id else ()).fetchall()
        return [dict(row) for row in rows if include_deleted or not json.loads(row["metadata"]).get("is_deleted")]

    def set_embedding(self, sid, embedding):
        with self.connect() as db:
            db.execute("UPDATE sources SET embedding=? WHERE id=?", (json.dumps(embedding), sid))

    def search(self, query, limit=5, embedding=None):
        terms = set(re.findall(r"[\w-]+", query.lower()))
        ranked = []
        semantic = False
        for row in self.sources():
            text = (row["title"] + " " + row["body"]).lower()
            words = set(re.findall(r"[\w-]+", text))
            exact = len(terms & words) / max(len(terms), 1)
            score = exact
            if embedding and row["embedding"]:
                other = json.loads(row["embedding"])
                if len(other) == len(embedding):
                    norm = (sum(x*x for x in other) * sum(x*x for x in embedding)) ** .5
                    cosine = sum(a*b for a,b in zip(other, embedding)) / max(norm, 1e-12)
                    score = .35 * exact + .65 * cosine
                    semantic = True
            if score > 0:
                item = json.loads(row["metadata"])
                item.update(title=row["title"], body=row["body"][:8000], score=round(score, 4))
                ranked.append(item)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return {"ok": True, "query": query, "method": "hybrid" if semantic else "keyword",
                "sources": ranked[:max(1, min(limit, 12))], "property_id": self.property_id}

    def artifact(self, kind, payload, record_id=None):
        aid = identifier(record_id or (kind + "-" + uuid.uuid4().hex))
        payload = dict(payload, id=aid, property_id=self.property_id)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?)", (aid, kind, now(), json.dumps(payload)))
        return payload

    def get_artifact(self, aid):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM artifacts WHERE id=?", (identifier(aid),)).fetchone()
        return json.loads(row[0]) if row else None

    def recent(self, kind=None, limit=8):
        with self.connect() as db:
            if kind:
                rows = db.execute("SELECT payload FROM artifacts WHERE kind=? ORDER BY created_at DESC LIMIT ?", (kind, limit)).fetchall()
            else:
                rows = db.execute("SELECT payload FROM artifacts ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delivery(self, artifact_id, destination, state=None, receipt=None):
        did = hashlib.sha256((artifact_id + "|" + destination).encode()).hexdigest()
        with self.connect() as db:
            if state is not None:
                db.execute("INSERT OR REPLACE INTO deliveries VALUES (?,?,?,?,?,?)",
                           (did, artifact_id, destination, state, json.dumps(receipt), now()))
            row = db.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone()
        return dict(row) if row else None
