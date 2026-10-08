"""Bounded reads of saved report evidence; never generate or publish content."""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import re
import sqlite3
from urllib.parse import urlsplit


REPORT_TYPES = {"general", "health", "maintenance", "inventory"}
_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}")
_ORDER = ("COALESCE(julianday(json_extract(payload,'$.generated_at')),"
          "julianday(json_extract(payload,'$.created_at')),julianday(created_at)) DESC,id DESC")
_TIME_BASIS = "saved generated_at, then saved created_at, then ledger updated_at fallback"
_MAX_BODY = 40000
_MAX_PACKET_BYTES = 12000


def _text(value, limit=200):
    # Omit over-bound metadata rather than make a shortened value look exact.
    return value if isinstance(value, str) and len(value) <= limit else None


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Use an observed report or incident identifier")
    return value


def _validate(report_type, incident_id):
    if report_type is not None and report_type not in REPORT_TYPES:
        raise ValueError("Unsupported recorded report_type")
    if incident_id is not None:
        _identifier(incident_id)


def _where(report_type, incident_id):
    clauses, values = ["kind='report'", "json_valid(payload)", "json_type(payload)='object'"], []
    for key, value in (("report_type", report_type), ("incident_id", incident_id)):
        if value is not None:
            clauses.append(f"json_extract(payload,'$.{key}')=?")
            values.append(value)
    return " AND ".join(clauses), values


@contextmanager
def _read_db(store):
    # PropertyStore.connect initializes schemas and directories. This read surface
    # deliberately bypasses it, including when no ledger has ever been created.
    db = sqlite3.connect(store.db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        yield db
    finally:
        db.close()


def _envelope(report_id, report_type, incident_id):
    return {"ok": True, "available": False,
            "selection": {"requested_report_id": report_id, "report_type": report_type,
                          "incident_id": incident_id, "order": _TIME_BASIS,
                          "scope": "exact saved metadata; no title or body classification"},
            "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "missing_evidence": [],
            "retrieval_effects": {"model_calls": 0, "artifact_writes": 0, "external_actions": 0}}


def _meta(row, item):
    created, generated = _text(item.get("created_at"), 64), _text(item.get("generated_at"), 64)
    declared = _text(item.get("report_type"), 40)
    return {"id": row["id"], "title": _text(item.get("title")),
            "report_type": declared, "incident_id": _text(item.get("incident_id"), 160),
            "created_at": created, "generated_at": generated,
            "status": _text(item.get("status"), 40),
            "is_demo": item.get("is_demo") if isinstance(item.get("is_demo"), bool) else None,
            "record_updated_at": _text(row["created_at"], 64)}


def _url(value):
    value = _text(value, 1000)
    if not value or any(ord(c) < 32 for c in value):
        return None
    try:
        parsed = urlsplit(value)
        return value if parsed.scheme in {"http", "https"} and parsed.netloc and not parsed.username else None
    except ValueError:
        return None


def _links(db, report_id, item):
    receipt = item.get("drive_receipt")
    receipt = receipt if isinstance(receipt, dict) else {}
    deliveries = db.execute("SELECT destination,state,receipt,updated_at FROM deliveries "
                            "WHERE artifact_id=? ORDER BY updated_at DESC LIMIT 4", (report_id,)).fetchall()
    observed = []
    for row in deliveries[:3]:
        try:
            value = json.loads(row["receipt"] or "{}")
        except (ValueError, TypeError):
            value = {}
        value = value if isinstance(value, dict) else {}
        observed.append({"destination": _text(row["destination"], 200),
                         "state": _text(row["state"], 40), "updated_at": _text(row["updated_at"], 64),
                         "message_id": _text(value.get("message_id"), 80),
                         "channel_id": _text(value.get("channel_id"), 80),
                         "url": _url(value.get("url") or value.get("webViewLink"))})
    return {"download_url": "/property/report/" + report_id,
            "drive_url": _url(receipt.get("webViewLink") or receipt.get("url")),
            "deliveries": observed, "deliveries_truncated": len(deliveries) > 3}


def catalog(store, *, report_type=None, incident_id=None, offset=0, limit=5):
    """Discover metadata only. 'general' is not an inferred health subtype."""
    _validate(report_type, incident_id)
    if type(offset) is not int or not 0 <= offset <= 10000 or type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Use offset 0..10000 and limit 1..20")
    result = _envelope(None, report_type, incident_id)
    result.update(reports=[], page={"offset": offset, "next_offset": None, "total": 0})
    if not store.db_path.is_file():
        result["missing_evidence"].append("Saved report ledger unavailable")
        return result
    where, values = _where(report_type, incident_id)
    with _read_db(store) as db:
        count = db.execute("SELECT COUNT(*) FROM artifacts WHERE " + where, values).fetchone()[0]
        rows = db.execute("SELECT id,created_at,payload FROM artifacts WHERE " + where +
                          " ORDER BY " + _ORDER + " LIMIT ? OFFSET ?", values + [limit, offset]).fetchall()
        result["page"]["total"] = count
        for row in rows:
            result["reports"].append(_meta(row, json.loads(row["payload"])))
            if _bytes(result) > _MAX_PACKET_BYTES:
                result["reports"].pop()
                break
    end = offset + len(result["reports"])
    result["page"]["next_offset"] = end if end < count else None
    result["available"] = bool(result["reports"])
    if not result["available"]:
        result["missing_evidence"].append("No saved report matches the requested metadata/page")
        if rows:
            result.update(ok=False)
            result["page"]["next_offset"] = None
            result["missing_evidence"].append("Saved metadata exceeds the bounded report packet")
    return result


def _bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def read(store, report_id="latest", *, report_type=None, incident_id=None, body_offset=0, body_limit=4000):
    _identifier(report_id)
    _validate(report_type, incident_id)
    if type(body_offset) is not int or not 0 <= body_offset <= _MAX_BODY:
        raise ValueError("Use body_offset 0..40000")
    if type(body_limit) is not int or not 1 <= body_limit <= 4000:
        raise ValueError("Use body_limit 1..4000")
    result = _envelope(report_id, report_type, incident_id)
    result["report"] = None
    if not store.db_path.is_file():
        result["missing_evidence"].append("Saved report ledger unavailable")
        return result
    where, values = _where(report_type, incident_id)
    if report_id != "latest":
        where += " AND id=?"
        values.append(report_id)
    with _read_db(store) as db:
        row = db.execute("SELECT id,created_at,payload FROM artifacts WHERE " + where +
                         " ORDER BY " + _ORDER + " LIMIT 1", values).fetchone()
        if not row:
            result["missing_evidence"].append("No saved report matches the exact ID/metadata selector")
            return result
        item = json.loads(row["payload"])
        value = _meta(row, item)
        value["source_dates"] = {key: _text(item.get(key), 64)
                                 for key in ("observed_at", "source_observed_at", "source_updated_at")}
        snapshot = item.get("inventory_snapshot")
        if isinstance(snapshot, dict):
            value["source_dates"]["inventory_observed_at"] = _text(snapshot.get("observed_at"), 64)
        if item.get("report_type") == "inventory":
            value["inventory_snapshot_id"] = _text(item.get("inventory_snapshot_id"), 80)
        value["provenance"] = {
            "source": "local_property_report_ledger", "ledger_kind": "report",
            "author": _text(item.get("author"), 80),
            "content_scope": "exact saved authored text; not independent fact verification",
            "classification_scope": "stored report_type only; legacy missing type stays unknown",
            "incident_binding_scope": "recorded incident ID only; source revision not verified",
            "retrieval_refreshes_evidence": False, "record_updated_at_is_observation_time": False,
            "created_at_and_generated_at_are_observation_times": False,
        }
        if item.get("report_type") == "inventory":
            value["provenance"]["inventory_binding_scope"] = (
                "recorded snapshot ID only; use inventory_report_read for source binding and current freshness checks"
            )
        value["links"] = _links(db, row["id"], item)
    result["report"] = value
    if not any(value["source_dates"].values()):
        result["missing_evidence"].append("Source observation dates are not recorded in report metadata")
    if value["report_type"] is None:
        result["missing_evidence"].append("Report type was not recorded; no type inferred from prose")
    body = item.get("body")
    if not isinstance(body, str) or len(body) > _MAX_BODY:
        value.update(body=None, body_page=None)
        result["missing_evidence"].append("Saved body unavailable or exceeds the 40000-character read bound")
        return result
    if body_offset > len(body):
        raise ValueError("body_offset exceeds the saved body length")
    end = min(len(body), body_offset + body_limit)
    value["body"] = body[body_offset:end]
    value["body_page"] = {"offset": body_offset, "next_offset": end if end < len(body) else None,
                          "total_characters": len(body), "complete": body_offset == 0 and end == len(body),
                          "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}
    # Bound the *serialized* packet too, including escaped control characters.
    # Pagination preserves all body bytes; no ellipsis or rewritten summary.
    while _bytes(result) > _MAX_PACKET_BYTES and end > body_offset:
        end = body_offset + (end - body_offset) // 2
        value["body"] = body[body_offset:end]
        value["body_page"].update(next_offset=end if end < len(body) else None,
                                   complete=body_offset == 0 and end == len(body))
    if _bytes(result) > _MAX_PACKET_BYTES or (end == body_offset and end < len(body)):
        result.update(ok=False, report=None)
        result["missing_evidence"].append("Saved metadata exceeds the bounded report packet")
        return result
    result["available"] = True
    return result
