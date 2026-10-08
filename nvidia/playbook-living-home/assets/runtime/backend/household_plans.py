"""Household-scoped controls and reviewed native Home Assistant automations."""
from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

from plan_journal import PlanJournal, ATTRIBUTES, _services, _keys, _text, _number, _timestamp, _utc


class HouseholdPlanStore(PlanJournal):
    def __init__(self, data_directory, ha, run_store, entity_ids, *, now=_utc, verify_timeout=8):
        self.ledger = Path(data_directory) / 'device-plans.json'
        self.ha, self.run_store = ha, run_store
        self.entity_ids = frozenset(entity_ids)
        self.now, self.verify_timeout = now, verify_timeout
        self.media_verify_timeout = verify_timeout
        self.calendar_provider = lambda: {'available': False}
        self.lock = threading.RLock()

    def _context(self):
        state = self.run_store._load()
        if state.get('device_profile') != 'physical':
            raise ValueError('A configured household is required')
        return {'run_id': state['run_id'], 'device_profile': 'physical'}

    def capabilities(self, **unused):
        with self.run_store.lock, self.lock:
            context, rows = self._context(), self._rows()
            entities = []
            for eid in sorted(self.entity_ids):
                row = rows.get(eid, {})
                attrs = {k: copy.deepcopy(v) for k, v in row.get('attributes', {}).items() if k in ATTRIBUTES}
                domain = eid.split('.', 1)[0]
                members = self._group_members(eid, rows)
                safe_group = members is not None
                services = _services(eid, attrs) if domain in {'light', 'switch', 'fan', 'climate', 'cover', 'media_player'} else {}
                if not safe_group:
                    services = {}
                available = row.get('state') not in {None, 'unknown', 'unavailable'}
                entities.append({'entity_id': eid, 'name': attrs.get('friendly_name', eid),
                    'state': row.get('state', 'unknown'), 'attributes': attrs,
                    'available': available, 'read_only': not bool(services), 'services': services,
                    'is_virtual': False, 'group_members': members if safe_group else [],
                    'trigger_available': available, 'trigger_attributes': sorted(k for k, v in attrs.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)),
                    'placement': {'available': False, 'missing_evidence': ['Area assignment is not available in this connection']},
                    'source': 'Selected Home Assistant entity',
                    'observation_semantics': {'state_basis': 'reported_entity_state', 'physical_operation_evidence': 'not_collected'}})
            return {'ok': True, **context, 'entities': entities, 'observed_at': self.now().isoformat(),
                'source': 'Home Assistant /api/states', 'scope': 'Only explicitly selected household entities',
                'limits': {'actions': 20, 'automations': 10, 'triggers_per_automation': 10, 'draft_lifetime_seconds': 900},
                'missing_evidence': ['Home Assistant state does not establish physical operation or equipment condition'],
                'timing': 'Explicit state change or absolute time. Intent must be supplied by the user.'}

    def _group_members(self, eid, rows, ancestors=frozenset()):
        """Expand only fully selected, acyclic groups, including nested groups."""
        if eid in ancestors or eid not in self.entity_ids or eid not in rows:
            return None
        members = rows[eid].get('attributes', {}).get('entity_id', [])
        if not isinstance(members, list) or any(not isinstance(m, str) for m in members):
            return None
        leaves = set()
        for member in members:
            if member.split('.', 1)[0] != eid.split('.', 1)[0]:
                return None
            nested = self._group_members(member, rows, ancestors | {eid})
            if nested is None:
                return None
            leaves.update(nested or [member])
        return sorted(leaves)

    def _validate_action(self, action, catalog):
        _keys(action, {'entity_id', 'service', 'data'}, {'entity_id', 'service'})
        eid = _text(action['entity_id'], 'entity_id', 200)
        entity = catalog.get(eid)
        if not entity or not entity['available'] or entity['read_only']:
            raise ValueError('Selected entity is unavailable or read-only: ' + eid)
        service = _text(action['service'], 'service', 100)
        contract = entity['services'].get(service)
        if contract is None:
            raise ValueError('Service is not advertised for ' + eid)
        data = action.get('data', {})
        _keys(data, contract['fields'], contract['required'])
        for key, value in data.items():
            field = contract['fields'][key]
            if field['type'] == 'number':
                _number(value, field['minimum'], field['maximum'])
            elif field['type'] == 'string' and value not in field['enum']:
                raise ValueError('Value is outside observed options')
            elif field['type'] == 'array':
                if not isinstance(value, list) or len(value) != 3:
                    raise ValueError('RGB color requires three components')
                for v in value:
                    _number(v, 0, 255)
        if 'rgb_color' in data and 'color_temp_kelvin' in data:
            raise ValueError('Choose one color representation')
        for member in entity['group_members']:
            if not catalog.get(member, {}).get('available'):
                raise ValueError('An authorized group member is unavailable')
        return {'entity_id': eid, 'service': service, 'data': copy.deepcopy(data)}

    def _native_action(self, action):
        # Freeze selected leaf targets in saved rules. A later change to a HA
        # group's membership must not expand an already authorized automation.
        members = self._group_members(action['entity_id'], self._rows())
        if members is None:
            raise ValueError('Device group membership changed; review a new plan')
        return {'action': action['service'], 'target': {'entity_id': members or action['entity_id']}, 'data': copy.deepcopy(action['data'])}

    def _save_automations(self, draft):
        additions = []
        has_time = any(t['kind'] == 'time' for a in draft['plan']['automations'] for t in a['triggers'])
        timezone = ZoneInfo(self.ha('GET', '/api/config', timeout=8)['time_zone']) if has_time else None
        for index, auto in enumerate(draft['plan']['automations']):
            triggers, conditions = [], []
            for ti, trigger in enumerate(auto['triggers']):
                if trigger['kind'] == 'state':
                    triggers.append({'trigger': 'state', 'id': 'state_' + str(ti), **{k: v for k, v in trigger.items() if k != 'kind'}})
                else:
                    instant = _timestamp(trigger['at'])
                    tid = 'time_' + str(ti)
                    triggers.append({'trigger': 'time', 'id': tid, 'at': instant.astimezone(timezone).strftime('%H:%M:%S')})
                    # This guard is generated from a validated instant, never user template text.
                    conditions.append({'condition': 'template', 'value_template': "{{ trigger.id != '" + tid + "' or ((as_timestamp(now()) - " + str(instant.timestamp()) + ") | abs < 60) }}"})
            native = {'id': 'livinghome_' + draft['draft_id'] + '_' + str(index), 'alias': auto['name'],
                'description': 'Living Home reviewed plan ' + draft['draft_id'],
                'triggers': triggers, 'conditions': conditions, 'actions': [self._native_action(a) for a in auto['actions']], 'mode': 'single'}
            # HA's configuration API validates and saves this exact new ID, then reloads it.
            # The durable applying journal was written by apply() before this call.
            result = self.ha('POST', '/api/config/automation/config/' + native['id'], native, timeout=20)
            if not isinstance(result, dict) or result.get('result') != 'ok':
                raise ValueError('Home Assistant did not acknowledge the automation definition')
            observed = self.ha('GET', '/api/config/automation/config/' + native['id'], timeout=8)
            if observed != native:
                raise ValueError('Saved automation differs from the reviewed definition; inspect it before retrying')
            additions.append(native)
        deadline = time.monotonic() + self.verify_timeout
        while True:
            rows, evidence = self._rows(), []
            for native in additions:
                match = next((r for eid, r in rows.items() if eid.startswith('automation.') and r.get('attributes', {}).get('id') == native['id']), None)
                evidence.append({'automation_id': native['id'], 'name': native['alias'], 'entity_id': match.get('entity_id') if match else None,
                    'loaded': bool(match), 'enabled': bool(match and match.get('state') == 'on'), 'executed': False,
                    'last_triggered': match.get('attributes', {}).get('last_triggered') if match else None,
                    'definition': native, 'definition_readback_matches': True, 'source': 'Native Home Assistant automation'})
            if all(e['loaded'] and e['enabled'] for e in evidence) or time.monotonic() >= deadline:
                return evidence
            time.sleep(0.2)
