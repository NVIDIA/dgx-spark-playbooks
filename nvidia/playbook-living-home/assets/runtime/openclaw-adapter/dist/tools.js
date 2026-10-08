import { createHash } from 'node:crypto';
import { HomeClient } from './client.js';
import { FAST_SCHEMA, PROPERTY_SCHEMA } from './schemas.js';
import { ContractError, exactId, fields, pick, requiredText, validate,           } from './validation.js';

const writes = new Map                                                 ();
const MAX_WRITE_BINDINGS = 1000;

async function once(callId        , args      , run                                           )                {
  if (typeof callId !== 'string' || !callId.trim()) throw new ContractError('A write requires a host-provided tool call ID');
  const hash = createHash('sha256').update(JSON.stringify(args)).digest('hex');
  const prior = writes.get(callId);
  if (prior) {
    if (prior.hash !== hash) throw new ContractError('Tool call ID was already used for different arguments');
    return prior.result;
  }
  if (writes.size >= MAX_WRITE_BINDINGS) throw new ContractError('Write safety ledger is full; inspect saved receipts before restarting the plugin');
  const key = 'openclaw:' + createHash('sha256').update(callId).digest('hex');
  const result = run(key);
  writes.set(callId, { hash, result });
  return result;
}

function timestamp(value        )          {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(Z|[+-]\d{2}:\d{2})$/.test(value)
      || !Number.isFinite(Date.parse(value))) return false;
  const day = new Date(value.slice(0, 10) + 'T00:00:00Z');
  return day.toISOString().slice(0, 10) === value.slice(0, 10);
}

function checkPlan(plan      , immediate         )       {
  if (!plan || !(plan.actions?.length || plan.automations?.length))
    throw new ContractError('Author explicit actions or automations; no device effects are inferred from the plan name');
  if (immediate && (!plan.actions?.length || plan.automations?.length
      || Object.hasOwn(plan, 'temporary_window') || Object.hasOwn(plan, 'calendar_event')))
    throw new ContractError('plan_execute accepts immediate actions only; future rules require preview and apply');
  for (const action of [...(plan.actions || []), ...(plan.automations || []).flatMap((rule      ) => rule.actions)]) {
    if (!/^[a-z_][a-z0-9_]*\.[a-z0-9_]+$/.test(action.entity_id) || !/^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$/.test(action.service))
      throw new ContractError('Plan actions require exact entity IDs and advertised domain.service names');
  }
  for (const rule of plan.automations || []) for (const trigger of rule.triggers) {
    if (trigger.kind === 'time' && !timestamp(trigger.at)) throw new ContractError('Time triggers require valid absolute ISO timestamps with a timezone offset');
    if (trigger.calendar_reference && !plan.calendar_event) throw new ContractError('Calendar references require the exact observed calendar_event binding');
  }
  if (plan.temporary_window && (!timestamp(plan.temporary_window.starts_at) || !timestamp(plan.temporary_window.ends_at)))
    throw new ContractError('Temporary windows require valid absolute ISO timestamps with timezone offsets');
}

function authorize(args      )       {
  if (args.authorization?.confirmed !== true) throw new ContractError('Execution requires actual user authorization for these exact effects');
  requiredText(args.authorization, 'user_request');
}

async function fast(client            , args      , callId        , signal              )                {
  if (args.operation === 'capabilities') {
    fields(args, ['operation']);
    return client.request('/device-plans/capabilities', undefined, 30000, signal);
  }
  if (args.operation === 'plan_status') {
    fields(args, ['operation', 'draft_id']);
    return client.request('/device-plans/status?' + new URLSearchParams({ draft_id: exactId(args.draft_id) }), undefined, 30000, signal);
  }
  requiredText(args, 'expected_run_id', 'device_profile');
  if (args.operation === 'plan_apply') {
    fields(args, ['operation', 'expected_run_id', 'device_profile', 'draft_id', 'plan_hash', 'authorization']);
    exactId(args.draft_id); requiredText(args, 'plan_hash'); authorize(args);
    return once(callId, args, key => client.request('/device-plans/apply', {
      ...pick(args, ['expected_run_id', 'device_profile', 'draft_id', 'plan_hash', 'authorization']), idempotency_key: key,
    }, 180000, signal));
  }
  const execute = args.operation === 'plan_execute';
  fields(args, ['operation', 'expected_run_id', 'device_profile', 'plan', ...(execute ? ['authorization'] : [])]);
  checkPlan(args.plan, execute);
  if (execute) {
    authorize(args);
    return once(callId, args, key => client.request('/device-plans/execute', {
      ...pick(args, ['expected_run_id', 'device_profile', 'plan', 'authorization']), idempotency_key: key,
    }, 180000, signal));
  }
  return client.request('/device-plans/preview', pick(args, ['expected_run_id', 'device_profile', 'plan']), 45000, signal);
}

async function property(client            , args      , callId        , signal              )                {
  const allowed                           = {
    status: [], health_snapshot: [], search: ['query', 'limit'], asset: ['asset_id'], incident: ['incident_id'],
    maintenance_read: ['incident_id', 'detail'], report_read: ['report_id', 'report_type', 'incident_id', 'body_offset', 'body_limit'],
    report: ['title', 'body', 'report_type', 'incident_id', 'maintenance_revision', 'publish_to_drive', 'format'],
    repair_draft: ['asset_id', 'incident_id', 'subject', 'body', 'recipient', 'maintenance_revision', 'save_to_gmail'],
  };
  fields(args, ['operation', ...allowed[args.operation]]);
  const route = (suffix        ) => client.propertyRoute(suffix);
  if (args.operation === 'health_snapshot') return client.health(signal);
  if (args.operation === 'status') return client.request(route('/status'), undefined, 30000, signal);
  if (args.operation === 'search') {
    requiredText(args, 'query');
    return client.request(route('/search') + '?' + new URLSearchParams({ q: args.query, limit: String(args.limit ?? 10) }), undefined, 30000, signal);
  }
  if (['asset', 'incident'].includes(args.operation)) {
    const id = exactId(args[args.operation + '_id']);
    return client.request(route('/' + args.operation + '/' + id), undefined, 30000, signal);
  }
  if (args.operation === 'maintenance_read') {
    const id = exactId(args.incident_id ?? 'current', true);
    return client.request(route('/maintenance/' + id + (args.detail === 'full' ? '' : '/brief')), undefined, 30000, signal);
  }
  if (args.operation === 'report_read') {
    const id = exactId(args.report_id ?? 'latest');
    if (args.incident_id !== undefined) exactId(args.incident_id);
    const query = new URLSearchParams();
    for (const key of ['report_type', 'incident_id', 'body_offset', 'body_limit']) if (key in args) query.set(key, String(args[key]));
    return client.request(route('/reports/' + id) + (query.size ? '?' + query : ''), undefined, 30000, signal);
  }
  if (args.publish_to_drive === true || args.save_to_gmail === true)
    throw new ContractError('External publication and Gmail drafts are unavailable; this release saves local records only');
  const draft = args.operation === 'repair_draft';
  requiredText(args, ...(draft ? ['asset_id', 'incident_id', 'subject', 'body'] : ['title', 'body']));
  if (draft) { exactId(args.asset_id); exactId(args.incident_id); }
  if (args.incident_id !== undefined) exactId(args.incident_id);
  if (args.report_type === 'maintenance') {
    requiredText(args, 'incident_id', 'maintenance_revision');
    if (args.body.length > 6000) throw new ContractError('Maintenance report body exceeds 6000 characters');
  } else if (!draft && args.maintenance_revision !== undefined) throw new ContractError('maintenance_revision requires a maintenance report');
  if (args.recipient !== undefined && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(args.recipient))
    throw new ContractError('Draft recipient must be an observed or user-provided email address');
  return once(callId, args, key => client.request(route(draft ? '/repair-draft' : '/report'), {
    ...pick(args, allowed[args.operation]), ...(draft ? { save_to_gmail: false } : { publish_to_drive: false, format: 'markdown' }),
    idempotency_key: key,
  }, 90000, signal));
}

export function createTools(client             )         {
  return [FAST_SCHEMA, PROPERTY_SCHEMA].map(schema => ({
    ...schema, label: schema.name, executionMode: 'sequential',
    async execute(callId        , args      , signal              ) {
      try {
        validate(schema.parameters, args);
        if (!client) throw new ContractError('Living Home plugin deployment configuration is required');
        const value = schema.name === FAST_SCHEMA.name ? await fast(client, args, callId, signal) : await property(client, args, callId, signal);
        return { content: [{ type: 'text', text: JSON.stringify(value) }], details: value, ...(value.ok === false ? { isError: true } : {}) };
      } catch (error) {
        const value = { ok: false, error: error instanceof ContractError ? error.message : 'Living Home operation unavailable or cancelled; inspect current evidence before retrying' };
        return { content: [{ type: 'text', text: JSON.stringify(value) }], details: value, isError: true };
      }
    },
  }));
}


//# sourceURL=tools.ts