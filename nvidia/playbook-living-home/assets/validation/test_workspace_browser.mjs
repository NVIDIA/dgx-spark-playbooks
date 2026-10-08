// Real browser -> packaged app -> isolated real Home Assistant with virtual entities.
// Credentials are read only from an explicitly supplied private lab file.
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { readFile, mkdir, writeFile, access } from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const [bundle, privateRoot, evidenceRoot] = process.argv.slice(2);
if (!bundle || !privateRoot || !evidenceRoot) throw Error('Provide the extracted package, private lab directory and evidence directory');
const credential = JSON.parse(await readFile(path.join(privateRoot, 'ha-credentials.json'), 'utf8'));
const run = randomUUID().slice(0, 8);
const dataDir = path.join(privateRoot, 'ux-household-' + run);
const connectionFile = path.join(privateRoot, 'ui-session-' + run, 'connection.json');
await mkdir(evidenceRoot, { recursive: true });
const results = [];
let child, browser, page, connection, automationId;
const haOrigin = 'http://127.0.0.1:18123';
async function ha(route, body) {
  const response = await fetch(haOrigin + route, { method: body ? 'POST' : 'GET',
    headers: { Authorization: 'Bearer ' + credential.access_token, 'Content-Type': 'application/json' }, ...(body ? {body: JSON.stringify(body)} : {}) });
  assert(response.ok, 'The isolated HA request failed: ' + response.status);
  return response.json();
}
async function waitFor(check, label, timeout = 20000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (await check()) return; await new Promise(r => setTimeout(r, 150)); }
  throw Error('Timed out: ' + label);
}
function passed(name) { results.push({ name, status: 'PASS' }); process.stdout.write('PASS ' + name + '\n'); }
async function startApp() {
  child = spawn(path.join(bundle, 'runtime/python/python.exe'), [path.join(bundle, 'runtime/ui/app.py'), '--data-dir', dataDir,
    '--port', '0', '--api-port', '0', '--no-browser', '--connection-file', connectionFile],
    { cwd: bundle, windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '';
  child.stderr.on('data', b => { stderr += b.toString(); });
  await waitFor(async () => {
    if (child.exitCode !== null) throw Error('Packaged application exited during startup: ' + stderr.replaceAll(credential.access_token, '[redacted]'));
    try { connection = JSON.parse(await readFile(connectionFile, 'utf8')); return true; } catch { return false; }
  }, 'packaged application startup');
  await page.goto(connection.url + '/#' + connection.session);
}
async function stopApp() {
  await page.getByRole('button', { name: 'Quit', exact: true }).click();
  await waitFor(async () => child.exitCode !== null, 'application shutdown');
  assert.equal(child.exitCode, 0);
}
async function workspace(pathname, body) {
  const response = await fetch(connection.url + pathname, { method: body ? 'POST' : 'GET',
    headers: { Authorization: 'Bearer ' + connection.session, 'Content-Type': 'application/json' }, ...(body ? {body: JSON.stringify(body)} : {}) });
  assert(response.ok);
  return response.json();
}
try {
  await ha('/api/services/input_boolean/turn_off', {entity_id: ['input_boolean.lab_lamp_power', 'input_boolean.lab_desk_power']});
  browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--disable-gpu'] });
  page = await browser.newPage({ viewport: { width: 1360, height: 1000 } });
  const browserErrors = [];
  page.on('pageerror', error => browserErrors.push(error.message));
  await startApp();
  await page.getByRole('heading', { name: 'Start with your devices.' }).waitFor();
  await page.screenshot({ path: path.join(evidenceRoot, '01-connect.png'), fullPage: true });
  assert.equal((await workspace('/api/status')).configured, false);
  passed('Packaged ARM64 runtime opens first run without a pre-existing household');
  await page.getByLabel('Home Assistant address').fill(haOrigin);
  await page.getByLabel('Access token', {exact:true}).fill('invalid-lab-test-token');
  await page.getByRole('button', { name: 'Connect to Home Assistant' }).click();
  await page.locator('#notice.error').waitFor();
  assert.equal((await workspace('/api/status')).configured, false);
  passed('Invalid HA token shows a recoverable error without creating household data');
  await page.getByLabel('Access token', {exact:true}).fill(credential.access_token);
  await page.getByRole('button', {name:'Connect to Home Assistant'}).click();
  await page.getByRole('heading', {name:'A home that’s yours.'}).waitFor();
  assert.equal(await page.locator('#ha-token').inputValue(), '');
  assert.equal(await page.locator('#device-selection input:checked').count(), 0);
  for (const eid of ['light.lab_desk_lamp','switch.lab_desk_plug','sensor.lab_sensor_battery','binary_sensor.lab_hallway_motion']) {
    await page.locator('input[value="' + eid + '"]').check();
  }
  await page.screenshot({path:path.join(evidenceRoot, '02-select.png'),fullPage:true});
  await page.getByRole('button', {name:'Open my home',exact:true}).click();
  await page.getByRole('heading', {name:'Your selected devices'}).waitFor();
  let status=await workspace('/api/status');
  assert.equal(status.capabilities.entities.length, 4);
  assert.equal(status.reports.length, 0);
  passed('Device selection creates a fresh household with exactly four chosen entities');
  const lampCard = page.locator('#device-cards article').filter({has:page.getByText('light.lab_desk_lamp',{exact:true})});
  await lampCard.getByRole('button',{name:'Turn on',exact:true}).click();
  await page.locator('#review-dialog[open]').waitFor();
  assert.equal((await ha('/api/states')).find(x=>x.entity_id==='light.lab_desk_lamp').state,'off');
  passed('Browser review makes no device change before approval');
  await page.getByRole('button',{name:'Approve and apply',exact:true}).click();
  await page.locator('#review-dialog').waitFor({state:'hidden'});
  assert.equal((await ha('/api/states')).find(x=>x.entity_id==='light.lab_desk_lamp').state,'on');
  passed('Browser approval applies the lamp action and confirms HA readback');
  await page.screenshot({path:path.join(evidenceRoot,'03-devices.png'),fullPage:true});
  await page.getByRole('button',{name:'Automations',exact:true}).click();
  await page.getByLabel('Rule name').fill('Workspace browser rule ' + run);
  await page.getByLabel('When this device').selectOption('switch.lab_desk_plug');
  await page.getByLabel('Changes to this state').fill('on');
  await page.getByLabel('Change this device').selectOption('light.lab_desk_lamp');
  await page.getByRole('button',{name:'Review rule',exact:true}).click();
  await page.locator('#review-dialog[open]').waitFor();
  await page.screenshot({path:path.join(evidenceRoot,'04-rule-review.png'),fullPage:true});
  await page.getByRole('button',{name:'Approve and apply',exact:true}).click();
  await page.locator('#review-dialog').waitFor({state:'hidden'});
  let rule=(await ha('/api/states')).find(e=>e.entity_id.startsWith('automation.') && e.attributes.friendly_name==='Workspace browser rule '+run);
  assert(rule && rule.state==='on'); automationId=rule.entity_id;
  await ha('/api/services/light/turn_off',{entity_id:'light.lab_desk_lamp'});
  await ha('/api/services/switch/turn_on',{entity_id:'switch.lab_desk_plug'});
  await waitFor(async()=>(await ha('/api/states')).find(e=>e.entity_id==='light.lab_desk_lamp').state==='on','real automation trigger');
  rule=(await ha('/api/states')).find(e=>e.entity_id===automationId);assert(rule.attributes.last_triggered);
  await page.getByRole('button',{name:'Check trigger status'}).click();
  await page.getByText(/last triggered/).waitFor();
  passed('Browser-built native automation fires from an actual virtual-device state transition');
  await page.getByRole('button',{name:'Reports',exact:true}).click();
  await page.getByRole('button',{name:'Check devices and save report'}).click();
  await page.locator('#report-reader').waitFor();
  assert.match(await page.locator('#report-body').innerText(),/18(?:\.0)?% battery reported/);
  status=await workspace('/api/status'); const reportId=status.reports[0].id;
  assert.equal(status.reports[0].author,'status_collector');
  assert(status.reports[0].source_observed_at);
  passed('Browser saves and reads a fresh local battery/availability report with observation time');
  await page.getByRole('button',{name:'Prepare schedule',exact:true}).click();
  await page.locator('#schedule-result').waitFor();
  assert.match(await page.locator('#schedule-message').innerText(),/create the job disabled/);
  passed('Schedule UX distinguishes prepared configuration from an installed/enabled cron job');
  await page.screenshot({path:path.join(evidenceRoot,'05-reports.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(evidenceRoot,'06-mobile.png'),fullPage:true});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  passed('Workspace fits a 390px viewport without horizontal overflow');
  await page.setViewportSize({width:1360,height:1000});
  await stopApp(); await startApp();
  await page.getByRole('heading',{name:'Your selected devices'}).waitFor();
  assert.equal((await workspace('/api/status')).reports[0].id,reportId);
  assert.equal((await workspace('/api/status')).capabilities.entities.length,4);
  passed('Restart preserves household identity, selected devices and saved reports');
  assert.deepEqual(browserErrors,[]);
  passed('No browser JavaScript errors in the tested workflow');
} catch(error) {
  results.push({name:'Browser integration',status:'FAIL',detail:error.message.replaceAll(credential.access_token,'[redacted]')});
  if(page)await page.screenshot({path:path.join(evidenceRoot,'failure.png'),fullPage:true});
  process.exitCode=1;
} finally {
  if(automationId)await ha('/api/services/automation/turn_off',{entity_id:automationId});
  await ha('/api/services/input_boolean/turn_off',{entity_id:['input_boolean.lab_lamp_power','input_boolean.lab_desk_power']});
  if(child&&child.exitCode===null){try{await stopApp();}catch{child.kill();}}
  if(browser)await browser.close();
  const summary={completedAt:new Date().toISOString(),environment:'Packaged Windows ARM64 app + headless Edge + real HA in isolated WSL2',
    modelConversationTested:false,cronExecuted:false,physicalDevicesTested:false,fullInstallerTested:false,
    results,cleanup:{automationDisabled:automationId||null,virtualLamp:'off',virtualPlug:'off'}};
  await writeFile(path.join(evidenceRoot,'browser-results.json'),JSON.stringify(summary,null,2));
  process.stdout.write(JSON.stringify({passed:results.filter(x=>x.status==='PASS').length,failed:results.filter(x=>x.status==='FAIL').length,evidence:evidenceRoot})+'\n');
}
