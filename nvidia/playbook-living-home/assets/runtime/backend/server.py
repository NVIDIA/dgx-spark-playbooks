"""Loopback-only household API. Run explicitly with --data-dir and --port.

Every request requires the new household's api-token.txt bearer token. The API
never loads reference-household configuration and never imports/reset seeds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'workflows'))
from household_state import HouseholdRunStore
from household_plans import HouseholdPlanStore
from records_store import PropertyStore, now, identifier
import saved_reports


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HomeAssistant:
    def __init__(self, env_path):
        config = {}
        for line in Path(env_path).read_text(encoding='utf-8-sig').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
        self.configure(config.get('HA_URL', ''), config.get('HA_TOKEN', ''))

    @classmethod
    def connection(cls, url, token):
        instance = cls.__new__(cls)
        instance.configure(url, token)
        return instance

    def configure(self, url, token):
        if not isinstance(url, str) or not isinstance(token, str):
            raise ValueError('Enter a Home Assistant address and access token')
        self.url, self.token = url.rstrip('/'), token
        url = urllib.parse.urlsplit(self.url)
        if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment or url.path:
            raise ValueError('Configure a Home Assistant origin without credentials in the URL')
        if not self.token or any(c.isspace() for c in self.token):
            raise ValueError('Configure the household Home Assistant token')
        self.opener = urllib.request.build_opener(NoRedirect())

    def __call__(self, method, path, body=None, *, timeout=8):
        permitted = path in {'/api/states', '/api/config'} and method == 'GET'
        permitted |= bool(re.fullmatch(r'/api/config/automation/config/livinghome_[a-f0-9]{32}_[0-9]{1,2}', path)) and method in {'GET', 'POST'}
        permitted |= bool(re.fullmatch(r'/api/services/(?:light|switch|fan|climate|cover|media_player)/[a-z_]+', path)) and method == 'POST'
        if not permitted:
            raise ValueError('Unsupported Home Assistant request')
        request = urllib.request.Request(self.url + path, method=method,
            data=json.dumps(body, allow_nan=False).encode() if body is not None else None,
            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json', 'Accept': 'application/json'})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError('Home Assistant response exceeded the size bound')
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            raise ValueError('Home Assistant request returned HTTP ' + str(code)) from None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            raise ValueError('Home Assistant response unavailable; inspect current state before retrying a write') from None


def text(value, name, limit=40000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or '\x00' in value:
        raise ValueError('Invalid ' + name)
    return value


def fields(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError('Unexpected or missing request fields')


class HouseholdAPI:
    def __init__(self, data_directory, ha=None, *, verify_timeout=8):
        self.root = Path(data_directory).resolve(strict=True)
        settings = json.loads((self.root / 'home-settings.json').read_text(encoding='utf-8-sig'))
        self.run_store = HouseholdRunStore(self.root / 'household.json')
        state = self.run_store._load()
        if settings.get('propertyId') != state['property_id']:
            raise ValueError('Household settings do not match the saved identity')
        self.api_token = (self.root / 'api-token.txt').read_text(encoding='utf-8-sig').strip()
        if len(self.api_token) < 32 or any(c.isspace() for c in self.api_token):
            raise ValueError('A private API token with at least 32 characters is required')
        self.ha = ha or HomeAssistant(self.root / '.env')
        self.plans = HouseholdPlanStore(self.root, self.ha, self.run_store, settings['entityIds'], verify_timeout=verify_timeout)
        self.records = PropertyStore(self.root / 'property', state['property_id'])
        self.lock = threading.RLock()

    def route(self, method, path, query, body):
        if method == 'GET' and path == '/healthz':
            return {'ok': True, 'property_id': self.records.property_id, 'service': 'living-home-household'}
        if path == '/device-plans/capabilities' and method == 'GET':
            return self.plans.capabilities()
        if path == '/device-plans/status' and method == 'GET':
            return self.plans.status(query.get('draft_id', ''))
        if path in {'/device-plans/preview', '/device-plans/apply', '/device-plans/execute'} and method == 'POST':
            return getattr(self.plans, path.rsplit('/', 1)[-1])(body)
        if path == '/property/status' and method == 'GET':
            return {'ok': True, 'property_id': self.records.property_id, 'sources': len(self.records.sources()),
                'recent_reports': self.records.recent('report'), 'recent_incidents': self.records.recent('incident'),
                'google': {'connected': False}, 'record_scope': 'This household only'}
        if path == '/property/search' and method == 'GET':
            return self.records.search(text(query.get('q'), 'query', 2000), int(query.get('limit', 5)))
        if path.startswith('/property/asset/') and method == 'GET':
            aid = identifier(path.rsplit('/', 1)[-1])
            return {'ok': True, 'asset_id': aid, 'sources': [dict(json.loads(r['metadata']), body=r['body']) for r in self.records.sources(aid)]}
        if path == '/property/source' and method == 'POST':
            return self.import_source(body)
        if path == '/property/incidents' and method == 'POST':
            return self.import_incident(body)
        if path.startswith('/property/incident/') and method == 'GET':
            return self.incident(path.rsplit('/', 1)[-1])
        if path.startswith('/property/maintenance/') and method == 'GET':
            segments = path.split('/')
            if len(segments) not in {4, 5} or len(segments) == 5 and segments[4] != 'brief':
                raise ValueError('Unsupported maintenance evidence route')
            result = self.incident(segments[3])
            result['evidence_revision'] = hashlib.sha256(json.dumps(result['incident'], sort_keys=True).encode()).hexdigest()
            result['available'] = True
            result['missing_evidence'] = ['Only supplied household observations and linked records are available']
            return result
        if path == '/property/report' and method == 'POST':
            return self.report(body)
        if path.startswith('/property/reports/') and method == 'GET':
            fields(query, {'report_type', 'incident_id', 'body_offset', 'body_limit'})
            options = {k: int(v) if k in {'body_offset', 'body_limit'} else v for k, v in query.items()}
            return saved_reports.read(self.records, path.rsplit('/', 1)[-1], **options)
        if path == '/property/repair-draft' and method == 'POST':
            return self.draft(body)
        raise ValueError('Unsupported household API operation')

    def import_source(self, body):
        fields(body, {'source_id', 'asset_id', 'source_kind', 'title', 'body', 'source_uri', 'source_updated_at'},
               {'source_id', 'asset_id', 'source_kind', 'title', 'body', 'source_uri'})
        source = dict(body)
        for field in ('source_id', 'asset_id'):
            identifier(text(source[field], field, 160))
        for field in ('source_kind', 'title', 'source_uri'):
            text(source[field], field, 2000)
        text(source['body'], 'body')
        if 'source_updated_at' in source:
            stamp = dt.datetime.fromisoformat(source['source_updated_at'].replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                raise ValueError('Source date needs a timezone')
        source['is_demo'] = False
        with self.lock:
            changed = self.records.put_source(source)
        return {'ok': True, 'source_id': source['source_id'], 'changed': changed, 'source_authority': 'User-supplied household record'}

    def import_incident(self, body):
        fields(body, {'asset_id', 'title', 'observations', 'observed_at', 'source_ids'}, {'asset_id', 'title', 'observations', 'observed_at', 'source_ids'})
        identifier(text(body['asset_id'], 'asset_id', 160))
        text(body['title'], 'title', 200)
        text(body['observations'], 'observations', 10000)
        stamp = dt.datetime.fromisoformat(text(body['observed_at'], 'observed_at', 80).replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            raise ValueError('Observation date needs a timezone')
        ids = body['source_ids']
        if not isinstance(ids, list) or len(ids) > 30 or any(not isinstance(s, str) for s in ids):
            raise ValueError('Provide at most 30 source IDs')
        with self.lock:
            available = {r['id'] for r in self.records.sources(body['asset_id'])}
            if not set(ids) <= available:
                raise ValueError('Linked sources must exist for this asset')
            result = self.records.artifact('incident', dict(body, created_at=now(), status='open', observation_author='user'))
        return {'ok': True, 'incident': result}

    def incident(self, incident_id):
        if incident_id == 'current':
            rows = self.records.recent('incident', limit=2)
            if len(rows) != 1:
                raise ValueError('Choose the exact incident ID from property status')
            item = rows[0]
        else:
            item = self.records.get_artifact(identifier(incident_id))
        if not item or 'observations' not in item:
            raise ValueError('Household incident not found')
        records = {r['id']: r for r in self.records.sources(item['asset_id'])}
        return {'ok': True, 'incident': item, 'sources': [dict(json.loads(records[s]['metadata']), body=records[s]['body']) for s in item['source_ids'] if s in records]}

    def report(self, body, *, author='agent_submitted'):
        fields(body, {'title', 'body', 'report_type', 'incident_id', 'maintenance_revision', 'publish_to_drive', 'format', 'idempotency_key'}, {'title', 'body'})
        title, content = text(body['title'], 'title', 200), text(body['body'], 'body')
        kind = body.get('report_type', 'general')
        if kind not in {'general', 'health', 'maintenance'} or body.get('publish_to_drive', False) is not False or body.get('format', 'markdown') != 'markdown':
            raise ValueError('This connection saves local Markdown reports only')
        with self.lock:
            incident_id = body.get('incident_id')
            if incident_id:
                if incident_id == 'current':
                    raise ValueError('Use an exact observed incident ID')
                self.incident(incident_id)
            if kind == 'maintenance':
                current = self.route('GET', '/property/maintenance/' + text(incident_id, 'incident_id', 160), {}, None)
                if current['evidence_revision'] != body.get('maintenance_revision'):
                    raise ValueError('Maintenance evidence changed; read it again')
            elif 'maintenance_revision' in body:
                raise ValueError('maintenance_revision requires maintenance report_type')
            payload = {'title': title, 'body': content, 'created_at': now(), 'generated_at': now(),
                'status': 'saved_local', 'author': author, 'report_type': kind, 'incident_id': incident_id,
                'publication_format': 'markdown', 'local_format': 'markdown', 'is_demo': False}
            # Store the collector timestamp separately from the model's authoring time.
            if kind == 'health':
                snapshot_path = self.root / 'health' / 'snapshots' / 'latest.json'
                if snapshot_path.exists():
                    snap = json.loads(snapshot_path.read_text('utf-8'))
                    payload['source_observed_at'] = snap.get('checked_at')
            fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            identity = text(body['idempotency_key'], 'idempotency_key', 200) if 'idempotency_key' in body else json.dumps([title, content, kind, incident_id], ensure_ascii=False)
            artifact_id = 'report-' + hashlib.sha256(identity.encode()).hexdigest()[:24]
            payload['request_fingerprint'] = fingerprint
            existing = self.records.get_artifact(artifact_id)
            if existing and existing.get('request_fingerprint') != fingerprint:
                raise ValueError('Report idempotency conflict; use the saved report receipt')
            result = existing or self.records.artifact('report', payload, artifact_id)
            return {'ok': True, 'report': result}

    def draft(self, body):
        fields(body, {'asset_id', 'incident_id', 'subject', 'body', 'recipient', 'save_to_gmail', 'maintenance_revision', 'idempotency_key'}, {'asset_id', 'incident_id', 'subject', 'body'})
        if body.get('save_to_gmail', False) is not False:
            raise ValueError('This connection saves local drafts only')
        incident = self.incident(body['incident_id'])['incident']
        if body['asset_id'] != incident['asset_id']:
            raise ValueError('Draft asset does not match its incident')
        subject, content = text(body['subject'], 'subject', 200), text(body['body'], 'body', 10000)
        recipient = body.get('recipient')
        if recipient is not None:
            text(recipient, 'recipient', 300)
        with self.lock:
            fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            identity = text(body['idempotency_key'], 'idempotency_key', 200) if 'idempotency_key' in body else json.dumps([body['incident_id'], subject, content, recipient])
            aid = 'draft-' + hashlib.sha256(identity.encode()).hexdigest()[:24]
            existing = self.records.get_artifact(aid)
            if existing and existing.get('request_fingerprint') != fingerprint:
                raise ValueError('Draft idempotency conflict; use the saved draft receipt')
            payload = {k: v for k, v in body.items() if k != 'idempotency_key'}
            result = existing or self.records.artifact('draft', dict(payload, created_at=now(), status='saved_local', sent=False, request_fingerprint=fingerprint), aid)
        return {'ok': True, 'draft': result, 'sent': False}


def make_server(api, port=8081):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'LivingHome'

        def log_message(self, *args):
            pass  # Do not log paths, tokens, observations or source text.

        def do_GET(self):
            self.handle_request()

        def do_POST(self):
            self.handle_request()

        def handle_request(self):
            self.connection.settimeout(15)
            expected = 'Bearer ' + api.api_token
            if not hmac.compare_digest(self.headers.get('Authorization', '').encode('utf-8'), expected.encode('utf-8')):
                return self.respond(401, {'ok': False, 'error': 'Household API authentication required'})
            if self.headers.get('Origin'):
                return self.respond(403, {'ok': False, 'error': 'Browser cross-origin API requests are not enabled'})
            try:
                url = urllib.parse.urlsplit(self.path)
                if url.scheme or url.netloc or url.fragment or '%' in url.path:
                    raise ValueError('Use a canonical relative API path')
                pairs = urllib.parse.parse_qsl(url.query, keep_blank_values=True)
                if len(dict(pairs)) != len(pairs):
                    raise ValueError('Duplicate query field')
                body = None
                if self.command == 'POST':
                    if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                        raise ValueError('Use application/json')
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 256 * 1024:
                        raise ValueError('Request body must be at most 256 KiB')
                    body = json.loads(self.rfile.read(size))
                with api.lock:
                    value = api.route(self.command, url.path, dict(pairs), body)
                self.respond(200, value)
            except (ValueError, KeyError, TypeError) as exc:
                # Validation details are useful, but never return transport credentials.
                message = str(exc).replace(api.api_token, '[redacted]')
                ha_token = getattr(api.ha, 'token', '')
                if ha_token:
                    message = message.replace(ha_token, '[redacted]')
                self.respond(400, {'ok': False, 'error': message[:400]})
            except Exception:
                self.respond(500, {'ok': False, 'error': 'Household operation unavailable; inspect saved receipts before retrying a write'})

        def respond(self, code, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8081)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Choose a local port from 1024 to 65535')
    service = make_server(HouseholdAPI(args.data_dir), args.port)
    print('Living Home household API listening on loopback port ' + str(args.port), flush=True)
    service.serve_forever()


if __name__ == '__main__':
    main()
