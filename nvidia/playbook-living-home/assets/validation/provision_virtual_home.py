"""Create credentials only in the isolated loopback Home Assistant lab."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import secrets
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'workflows'))
from initialize_household import initialize, protect_private_directory

ORIGIN = 'http://127.0.0.1:18123'
PRIVATE = ROOT / 'validation' / 'private'


def request(path, body=None, token=None, form=False):
    headers = {'Accept': 'application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    data = None
    if body is not None:
        headers['Content-Type'] = 'application/x-www-form-urlencoded' if form else 'application/json'
        data = (urllib.parse.urlencode(body) if form else json.dumps(body)).encode()
    with urllib.request.urlopen(urllib.request.Request(ORIGIN + path, data=data, headers=headers), timeout=40) as response:
        return json.load(response)


async def long_lived_token(access):
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ORIGIN + '/api/websocket') as websocket:
            assert (await websocket.receive_json())['type'] == 'auth_required'
            await websocket.send_json({'type': 'auth', 'access_token': access})
            assert (await websocket.receive_json())['type'] == 'auth_ok'
            await websocket.send_json({'id': 1, 'type': 'auth/long_lived_access_token', 'client_name': 'Living Home Validation', 'lifespan': 1})
            result = await websocket.receive_json()
            if not result.get('success'):
                raise RuntimeError('Lab access token creation failed')
            return result['result']


def main():
    PRIVATE.mkdir(exist_ok=True)
    protect_private_directory(PRIVATE)
    credentials_file = PRIVATE / 'ha-credentials.json'
    if credentials_file.exists():
        print(json.dumps({'status': 'already_provisioned', 'credentialPath': str(credentials_file)}))
        return
    steps = request('/api/onboarding')
    if any(s['step'] == 'user' and s['done'] for s in steps):
        raise RuntimeError('Lab owner already exists without this test credential receipt; no account was changed')
    credentials = {'username': 'livinghome-validation', 'password': secrets.token_urlsafe(32), 'client_id': ORIGIN + '/'}
    credentials_file.write_text(json.dumps(credentials, indent=2), encoding='utf-8')
    user = request('/api/onboarding/users', {**credentials, 'name': 'Living Home Validation', 'language': 'en'})
    token = request('/auth/token', {'grant_type': 'authorization_code', 'code': user['auth_code'], 'client_id': credentials['client_id']}, form=True)
    access = token['access_token']
    credentials.update(refresh_token=token['refresh_token'])
    credentials_file.write_text(json.dumps(credentials, indent=2), encoding='utf-8')
    request('/api/onboarding/core_config', {}, access)
    request('/api/onboarding/analytics', {}, access)
    request('/api/onboarding/integration', {'client_id': credentials['client_id'], 'redirect_uri': ORIGIN + '/?auth_callback=1'}, access)
    access = asyncio.run(long_lived_token(access))
    credentials['access_token'] = access
    credentials_file.write_text(json.dumps(credentials, indent=2), encoding='utf-8')
    config = {'schemaVersion': 1, 'haMode': 'existing', 'haUrl': ORIGIN,
        'entityIds': ['light.lab_desk_lamp', 'switch.lab_desk_plug', 'binary_sensor.lab_hallway_motion', 'sensor.lab_sensor_battery'],
        'timeZone': 'America/Los_Angeles', 'openclawProfile': 'living-home-validation', 'google': {'enabled': False}}
    result = initialize(config, PRIVATE / 'household', access)
    states = request('/api/states', token=access)
    selected = [{'entity_id': r['entity_id'], 'state': r['state']} for r in states if r['entity_id'] in config['entityIds']]
    result.update(haVersion=request('/api/config', token=access)['version'], selectedDevices=selected,
        scope='Real Home Assistant in an isolated WSL2 container, with software-backed devices')
    (ROOT / 'validation/evidence/ha-provisioning.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
