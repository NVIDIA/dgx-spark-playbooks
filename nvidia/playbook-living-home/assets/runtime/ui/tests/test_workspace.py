import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import Workspace, make_ui
from household_plans import HouseholdPlanStore
from initialize_household import initialize


class GroupScopeTests(unittest.TestCase):
    def store(self, ids):
        return HouseholdPlanStore(Path(tempfile.gettempdir()) / 'unused-scope-test', None, None, ids)

    def test_nested_unselected_member_is_refused(self):
        rows = {'light.outer': {'attributes': {'entity_id': ['light.inner']}},
                'light.inner': {'attributes': {'entity_id': ['light.private']}}, 'light.private': {'attributes': {}}}
        self.assertIsNone(self.store(['light.outer', 'light.inner'])._group_members('light.outer', rows))

    def test_selected_nested_group_expands_to_leaves_only(self):
        rows = {'light.outer': {'attributes': {'entity_id': ['light.inner']}},
                'light.inner': {'attributes': {'entity_id': ['light.lamp']}}, 'light.lamp': {'attributes': {}}}
        self.assertEqual(self.store(rows)._group_members('light.outer', rows), ['light.lamp'])

    def test_group_cycle_and_cross_domain_members_are_refused(self):
        rows = {'light.a': {'attributes': {'entity_id': ['light.b']}}, 'light.b': {'attributes': {'entity_id': ['light.a']}}}
        self.assertIsNone(self.store(rows)._group_members('light.a', rows))
        rows['light.b']['attributes']['entity_id'] = ['switch.c']
        rows['switch.c'] = {'attributes': {}}
        self.assertIsNone(self.store(rows)._group_members('light.a', rows))

    def test_native_rule_pins_leaf_targets_instead_of_group_id(self):
        rows = {'light.group': {'attributes': {'entity_id': ['light.a', 'light.b']}},
                'light.a': {'attributes': {}}, 'light.b': {'attributes': {}}}
        store = self.store(rows)
        with patch.object(store, '_rows', return_value=rows):
            action = store._native_action({'entity_id': 'light.group', 'service': 'light.turn_on', 'data': {}})
        rows['light.group']['attributes']['entity_id'].append('light.other')
        self.assertEqual(action['target']['entity_id'], ['light.a', 'light.b'])


class BrowserBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(Path(self.temp.name) / 'household', 0)
        self.server, self.token = make_ui(self.workspace, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.workspace.close()

    def request(self, path, headers=None, data=None):
        try:
            return urllib.request.urlopen(urllib.request.Request(self.url + path, headers=headers or {}, data=data))
        except urllib.error.HTTPError as error:
            return error

    def test_static_shell_contains_no_session_or_credentials(self):
        with self.request('/') as response:
            content = response.read().decode()
            self.assertNotIn(self.token, content)
            self.assertIn("frame-ancestors 'none'", response.headers['Content-Security-Policy'])

    def test_unauthenticated_status_and_wrong_token_are_rejected(self):
        with self.request('/api/status') as response:
            self.assertEqual(response.code, 401)
        with self.request('/api/status', {'Authorization': 'Bearer wrong'}) as response:
            self.assertEqual(response.code, 401)

    def test_foreign_origin_and_host_are_refused_even_with_auth(self):
        auth = {'Authorization': 'Bearer ' + self.token}
        for extra in ({'Origin': 'https://example.invalid'}, {'Host': 'attacker.invalid'}):
            with self.request('/api/status', dict(auth, **extra)) as response:
                self.assertEqual(response.code, 403)

    def test_authenticated_status_is_new_and_has_no_household_created(self):
        with self.request('/api/status', {'Authorization': 'Bearer ' + self.token}) as response:
            self.assertFalse(json.load(response)['configured'])
        self.assertFalse(self.workspace.data_dir.exists())

    def test_create_cannot_select_an_unobserved_entity(self):
        self.workspace.pending = {'entities': {'light.approved'}, 'at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc)}
        with self.assertRaisesRegex(ValueError, 'Select at least one'):
            self.workspace.create({'entityIds': ['light.other'], 'timeZone': 'UTC'})
        self.assertFalse(self.workspace.data_dir.exists())

    def test_bad_token_does_not_leave_a_partial_household(self):
        cfg = {'schemaVersion': 1, 'haMode': 'existing', 'haUrl': 'http://127.0.0.1:18123',
               'entityIds': ['light.own'], 'timeZone': 'UTC', 'openclawProfile': 'test', 'google': {'enabled': False}}
        with self.assertRaises(ValueError):
            initialize(cfg, self.workspace.data_dir, 'invalid\ncredential')
        self.assertFalse(self.workspace.data_dir.exists())


if __name__ == '__main__':
    unittest.main()
