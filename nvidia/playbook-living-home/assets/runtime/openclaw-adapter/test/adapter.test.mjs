import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { HomeClient, validateConfig } from '../dist/client.js';
import { createTools } from '../dist/tools.js';

const TOKEN = 'fake-local-backend-token-012345678901234567890';
const snapshot = { schema_version: 1, status: 'available', checked_at: '2026-10-04T00:00:00+00:00',
  coverage: { entity_selection: 'explicit_entity_ids', physical_devices: 'not_tested' }, inventory: [] };

function fixture(run) {
  const root = mkdtempSync(path.join(tmpdir(), 'living-home-adapter-'));
  const data = path.join(root, 'data'); mkdirSync(data);
  const collector = path.join(root, 'collector.py'); writeFileSync(collector, '# fake, never executed\n');
  const apiTokenFile = path.join(data, 'api-token.txt'); writeFileSync(apiTokenFile, TOKEN + '\n');
  const config = { baseUrl: 'http://127.0.0.1:19345', propertyBasePath: '/house/property',
    pythonExecutable: process.execPath, collectorPath: collector, dataDirectory: data, apiTokenFile };
  return Promise.resolve().then(() => run(config)).finally(() => rmSync(root, { recursive: true, force: true }));
}
const response = value => new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } });
const invoke = (client, name, callId, args, signal) => createTools(client).find(item => item.name === name).execute(callId, args, signal);

test('trusted config accepts a configured loopback port and validates all local paths', () => fixture(config => {
  assert.equal(validateConfig(config).baseUrl, 'http://127.0.0.1:19345');
  assert.equal(validateConfig({ ...config, baseUrl: 'http://[::1]:32123/' }).baseUrl, 'http://[::1]:32123');
  for (const change of [
    { baseUrl: 'https://external.example:8081' }, { baseUrl: 'http://localhost:19345' },
    { baseUrl: 'http://user:private@127.0.0.1:19345' }, { baseUrl: 'http://127.0.0.1:19345/path' },
    { propertyBasePath: '/property/../private' }, { pythonExecutable: 'relative/python' },
    { dataDirectory: config.collectorPath }, { apiTokenFile: path.join(config.dataDirectory, 'absent') },
    { collectorPath: process.execPath }, { unexpected: 'secret' },
  ]) assert.throws(() => validateConfig({ ...config, ...change }), error => !error.message.includes('private') && !error.message.includes('secret'));
}));

test('health executes fixed argv without a shell and scrubs inherited HA credentials', () => fixture(async config => {
  let call;
  const previousUrl = process.env.HA_URL, previousToken = process.env.HA_TOKEN;
  process.env.HA_URL = 'http://unrelated-house.invalid'; process.env.HA_TOKEN = ['unrelated', 'home', 'secret'].join('-');
  try {
    const client = new HomeClient(config, { execFile: (file, args, options, callback) => {
      call = { file, args, options }; callback(null, JSON.stringify({ ...snapshot, note: TOKEN }), 'ignored-secret-stderr');
    } });
    const result = await invoke(client, 'living_home_property', 'health-runner-test', { operation: 'health_snapshot' });
    assert.equal(result.details.status, 'available');
    assert.equal(result.details.note, '[redacted]');
    assert.equal(call.file, config.pythonExecutable);
    assert.deepEqual(call.args, [config.collectorPath, '--env-file', path.join(config.dataDirectory, '.env'),
      '--config', path.join(config.dataDirectory, 'health-config.json'), '--output-dir', path.join(config.dataDirectory, 'health', 'snapshots')]);
    assert.equal(call.options.cwd, config.dataDirectory); assert.equal(call.options.windowsHide, true);
    assert.equal(call.options.shell, false); assert.equal(call.options.timeout, 35000);
    assert.equal(call.options.env.HA_URL, undefined); assert.equal(call.options.env.HA_TOKEN, undefined);
    assert.equal(call.options.env.PYTHONUTF8, '1');
    assert(!JSON.stringify(result).includes(TOKEN)); assert(!JSON.stringify(result).includes('stderr'));
  } finally {
    if (previousUrl === undefined) delete process.env.HA_URL; else process.env.HA_URL = previousUrl;
    if (previousToken === undefined) delete process.env.HA_TOKEN; else process.env.HA_TOKEN = previousToken;
  }
}));

test('health rejects model supplied executable or path fields before starting anything', () => fixture(async config => {
  let starts = 0;
  const client = new HomeClient(config, { execFile: () => { starts++; } });
  const result = await invoke(client, 'living_home_property', 'health-path-injection', { operation: 'health_snapshot', collectorPath: '/untrusted/path' });
  assert.equal(result.isError, true); assert.equal(starts, 0);
  assert(!JSON.stringify(result).includes('/untrusted/path'));
}));

test('health cancellation avoids pre-aborted startup and forwards a live signal', () => fixture(async config => {
  let starts = 0, options;
  const client = new HomeClient(config, { execFile: (_file, _args, opts, callback) => {
    starts++; options = opts;
    opts.signal.addEventListener('abort', () => callback(new Error('private subprocess failure ' + TOKEN), '', TOKEN), { once: true });
  } });
  const cancelled = new AbortController(); cancelled.abort();
  assert.equal((await client.health(cancelled.signal)).ok, false); assert.equal(starts, 0);
  const controller = new AbortController(); const pending = client.health(controller.signal); controller.abort();
  const result = await pending;
  assert.equal(options.signal, controller.signal); assert.equal(starts, 1); assert.equal(result.ok, false);
  assert(result.error.includes('cancelled')); assert(!JSON.stringify(result).includes(TOKEN));
}));

test('health runner failures and malformed stdout never expose process errors or stderr', () => fixture(async config => {
  for (const [error, stdout] of [[new Error(TOKEN), TOKEN], [null, TOKEN], [null, '[]'], [null, '{}']]) {
    const client = new HomeClient(config, { execFile: (_file, _args, _options, callback) => callback(error, stdout, TOKEN) });
    const result = await client.health();
    assert.equal(result.ok, false); assert(!JSON.stringify(result).includes(TOKEN));
  }
}));

test('backend calls use token file and configurable property prefix with encoded search text', () => fixture(async config => {
  const calls = [];
  const client = new HomeClient(config, { fetch: async (url, options) => { calls.push({ url, options }); return response({ ok: true, echoed: TOKEN }); } });
  const result = await invoke(client, 'living_home_property', 'search-routing', { operation: 'search', query: 'owned / room', limit: 3 });
  assert.equal(result.details.echoed, '[redacted]');
  const url = new URL(calls[0].url); assert.equal(url.origin, config.baseUrl);
  assert.equal(url.pathname, '/house/property/search'); assert.equal(url.searchParams.get('q'), 'owned / room');
  assert.equal(calls[0].options.headers.Authorization, 'Bearer ' + TOKEN);
  assert.equal(calls[0].options.redirect, 'error'); assert.equal(calls[0].options.method, 'GET');
  writeFileSync(config.apiTokenFile, 'rotated-' + TOKEN);
  await client.request(client.propertyRoute('/status'));
  assert.equal(calls[1].options.headers.Authorization, 'Bearer rotated-' + TOKEN);
}));

test('missing or short API token fails before a backend request without leaking content', () => fixture(async config => {
  let calls = 0;
  const client = new HomeClient(config, { fetch: async () => { calls++; return response({}); } });
  for (const token of ['short-secret', 'invalid\n' + TOKEN]) {
    writeFileSync(config.apiTokenFile, token);
    const result = await invoke(client, 'living_home_property', 'invalid-token-' + calls, { operation: 'status' });
    assert.equal(result.isError, true); assert(!JSON.stringify(result).includes(token));
  }
  assert.equal(calls, 0);
}));

test('only supported tools and operations are registered, with no demo or cloud actions', () => fixture(async config => {
  let calls = 0;
  const client = new HomeClient(config, { fetch: async () => { calls++; return response({}); } });
  const tools = createTools(client);
  assert.deepEqual(tools.map(tool => tool.name), ['living_home_fast', 'living_home_property']);
  for (const operation of ['demo_v2', 'reset', 'share', 'present_prepared', 'inventory_snapshot', 'ingest', 'analyze']) {
    const result = await invoke(client, 'living_home_property', 'retired-' + operation, { operation });
    assert.equal(result.isError, true);
  }
  assert.equal(calls, 0);
}));

test('API transport and rejection errors are redacted and writes are never retried', () => fixture(async config => {
  let calls = 0;
  const client = new HomeClient(config, { fetch: async () => { calls++; throw new Error(TOKEN); } });
  const result = await client.request(client.propertyRoute('/report'), { title: 'Local', body: 'Observed text' });
  assert.equal(result.ok, false); assert.equal(result.outcome_unverified, true); assert.equal(calls, 1);
  assert(!JSON.stringify(result).includes(TOKEN));
  const rejected = new HomeClient(config, { fetch: async () => new Response(JSON.stringify({ error: TOKEN }), { status: 401 }) });
  const failed = await rejected.request(rejected.propertyRoute('/status'));
  assert.equal(failed.http_status, 401); assert(!JSON.stringify(failed).includes(TOKEN));
}));

test('requests refuse unsupported and noncanonical backend routes', () => fixture(async config => {
  let calls = 0;
  const client = new HomeClient(config, { fetch: async () => { calls++; return response({}); } });
  for (const route of ['/demo/v2/action', '/property/report', '//other-host/status', '/house/property/reports/..',
    '/house/property/status#secret', '/house/property/reports/report-1?body_limit=9999',
    '/house/property/reports/latest?report_type=health&report_type=general', '/house/property/incident/%2e%2e']) {
    await assert.rejects(client.request(route));
  }
  assert.equal(calls, 0);
}));

test('plans preserve authored effects, observed run guards and actual authorization', () => fixture(async config => {
  const calls = [];
  const client = new HomeClient(config, { fetch: async (url, options) => { calls.push({ url, options }); return response({ ok: true, draft_id: 'draft-test', plan_hash: 'a'.repeat(64) }); } });
  const plan = { name: 'Owned light', actions: [{ entity_id: 'light.owned', service: 'light.turn_on', data: { brightness: 120 } }] };
  const preview = { operation: 'plan_preview', expected_run_id: 'owned-run', device_profile: 'physical', plan };
  assert.equal((await invoke(client, 'living_home_fast', 'plan-preview-contract', preview)).details.ok, true);
  assert.deepEqual(JSON.parse(calls[0].options.body), { expected_run_id: 'owned-run', device_profile: 'physical', plan });
  const apply = { operation: 'plan_apply', expected_run_id: 'owned-run', device_profile: 'physical', draft_id: 'draft-test', plan_hash: 'a'.repeat(64),
    authorization: { confirmed: true, user_request: 'Apply this reviewed owned-light plan' } };
  const first = await invoke(client, 'living_home_fast', 'plan-apply-contract', apply);
  const second = await invoke(client, 'living_home_fast', 'plan-apply-contract', apply);
  assert.deepEqual(first, second); assert.equal(calls.length, 2);
  const written = JSON.parse(calls[1].options.body);
  assert.equal(written.authorization.user_request, apply.authorization.user_request);
  assert.match(written.idempotency_key, /^openclaw:[a-f0-9]{64}$/);
  assert.equal((await invoke(client, 'living_home_fast', 'plan-apply-contract', { ...apply, plan_hash: 'b'.repeat(64) })).isError, true);
  assert.equal(calls.length, 2);
}));

test('physical plans require consent and immediate execute refuses future rules', () => fixture(async config => {
  let calls = 0;
  const client = new HomeClient(config, { fetch: async () => { calls++; return response({ ok: true }); } });
  const action = { entity_id: 'light.owned', service: 'light.turn_off' };
  const base = { operation: 'plan_execute', expected_run_id: 'owned-run', device_profile: 'physical', plan: { name: 'Turn off', actions: [action] },
    authorization: { confirmed: true, user_request: 'Turn off this light now' } };
  const variants = [
    { ...base, device_profile: 'virtual' }, { ...base, authorization: { confirmed: false, user_request: 'preview only' } },
    { ...base, plan: { name: 'Future', automations: [{ name: 'At time', triggers: [{ kind: 'time', at: '2027-01-01T00:00:00Z' }], actions: [action] }] } },
    { ...base, plan: { ...base.plan, temporary_window: { starts_at: '2027-01-01T00:00:00Z', ends_at: '2027-01-02T00:00:00Z' } } },
  ];
  for (let index = 0; index < variants.length; index++) assert.equal((await invoke(client, 'living_home_fast', 'reject-plan-' + index, variants[index])).isError, true);
  assert.equal(calls, 0);
  assert.equal((await invoke(client, 'living_home_fast', 'immediate-contract', base)).details.ok, true);
  assert.equal(calls, 1);
}));

test('local reports and repair drafts reject cloud flags and preserve authored prose', () => fixture(async config => {
  const calls = [];
  const client = new HomeClient(config, { fetch: async (url, options) => { calls.push({ url, options }); return response({ ok: true, report_id: 'report-local' }); } });
  const report = { operation: 'report', title: 'Selected entity status', body: 'Only selected entities checked. Physical operation not tested.', report_type: 'health' };
  assert.equal((await invoke(client, 'living_home_property', 'local-health-report', { ...report, publish_to_drive: true })).isError, true);
  assert.equal(calls.length, 0);
  assert.equal((await invoke(client, 'living_home_property', 'local-health-report', report)).details.ok, true);
  const payload = JSON.parse(calls[0].options.body);
  assert.equal(payload.body, report.body); assert.equal(payload.publish_to_drive, false); assert.equal(payload.format, 'markdown');
  assert.equal(new URL(calls[0].url).pathname, '/house/property/report');
  const draft = { operation: 'repair_draft', asset_id: 'asset-owned', incident_id: 'incident-owned', subject: 'Review needed', body: 'Cause is not established.' };
  assert.equal((await invoke(client, 'living_home_property', 'local-repair-draft', { ...draft, save_to_gmail: true })).isError, true);
  assert.equal((await invoke(client, 'living_home_property', 'local-repair-draft', draft)).details.ok, true);
  assert.equal(JSON.parse(calls[1].options.body).save_to_gmail, false);
}));

test('report paging and source record IDs cannot supply filesystem paths', () => fixture(async config => {
  const calls = [];
  const client = new HomeClient(config, { fetch: async (url) => { calls.push(url); return response({ ok: true }); } });
  const result = await invoke(client, 'living_home_property', 'read-health-report', { operation: 'report_read', report_type: 'health', body_offset: 10, body_limit: 200 });
  assert.equal(result.details.ok, true);
  const url = new URL(calls[0]); assert.equal(url.pathname, '/house/property/reports/latest');
  assert.equal(url.searchParams.get('report_type'), 'health'); assert.equal(url.searchParams.get('body_limit'), '200');
  for (const args of [{ operation: 'asset', asset_id: '../../private' }, { operation: 'report_read', report_id: 'https://external.invalid' },
    { operation: 'report_read', incident_id: 'current' }]) assert.equal((await invoke(client, 'living_home_property', 'invalid-record-' + args.operation, args)).isError, true);
  assert.equal(calls.length, 1);
}));

test('manifest and generated tool configuration agree on two tool names', async () => {
  const manifest = JSON.parse(readFileSync(new URL('../openclaw.plugin.json', import.meta.url), 'utf8'));
  assert.equal(manifest.id, 'living-home');
  assert.deepEqual(manifest.contracts.tools, createTools().map(tool => tool.name));
  assert.deepEqual(manifest.configSchema.required, ['baseUrl', 'pythonExecutable', 'collectorPath', 'dataDirectory', 'apiTokenFile']);
});
