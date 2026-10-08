"""Deploy only reviewed application sources to the Linux validation guest."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path('/mnt/c/LivingHome-installer')
LAB = Path('/opt/livinghome-validation')
APP = LAB / 'app'
APP.mkdir(exist_ok=False)
allowed = [
    *(SOURCE / 'runtime/backend').glob('*.py'),
    SOURCE / 'runtime/health/collect_home_status.py',
    SOURCE / 'workflows/initialize_household.py',
    SOURCE / 'workflows/household_state.py',
    *(SOURCE / 'runtime/openclaw-adapter/dist').glob('*.js'),
    SOURCE / 'runtime/openclaw-adapter/package.json',
]
inventory = []
for source in allowed:
    relative = source.relative_to(SOURCE)
    target = APP / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    inventory.append({'path': str(relative), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
sys.path.insert(0, str(APP / 'workflows'))
from initialize_household import initialize

credential = json.loads((SOURCE / 'validation/private/ha-credentials.json').read_text())
config = json.loads((SOURCE / 'validation/private/household/home-settings.json').read_text())
config = {k: config[k] for k in ('schemaVersion', 'haMode', 'haUrl', 'entityIds', 'timeZone', 'openclawProfile', 'google')}
data = LAB / 'household'
result = initialize(config, data, credential['access_token'])
with (LAB / 'backend.log').open('a') as log:
    process = subprocess.Popen([sys.executable, str(APP / 'runtime/backend/server.py'), '--data-dir', str(data), '--port', '18081'],
        cwd=APP, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(LAB / 'backend.pid').write_text(str(process.pid))
result.update(backendPid=process.pid, backendPort=18081, files=inventory,
    installationMethod='Source deployment in isolated WSL2 Linux guest; not Windows installer execution')
(SOURCE / 'validation/evidence/guest-provisioning.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'files': len(inventory), 'backendPid': process.pid, 'backendPort': 18081, 'guestData': str(data)}))
