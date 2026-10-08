                                            

const text = (maxLength        , minLength = 1)       => ({ type: 'string', minLength, maxLength });
const id = text(160);
const physical = { type: 'string', enum: ['physical'] };
const authorization = { type: 'object', properties: { confirmed: { type: 'boolean' }, user_request: text(2000) },
  required: ['confirmed', 'user_request'], additionalProperties: false };
const action = {
  type: 'object', properties: {
    entity_id: text(300), service: text(100),
    data: { type: 'object', properties: {
      brightness: { type: 'number', minimum: 1, maximum: 255 },
      brightness_pct: { type: 'number', minimum: 0, maximum: 100 },
      color_temp_kelvin: { type: 'number', minimum: 1000, maximum: 10000 },
      rgb_color: { type: 'array', minItems: 3, maxItems: 3, items: { type: 'integer', minimum: 0, maximum: 255 } },
      percentage: { type: 'number', minimum: 0, maximum: 100 },
      position: { type: 'number', minimum: 0, maximum: 100 },
      temperature: { type: 'number', minimum: -1000, maximum: 1000 },
      hvac_mode: text(100), source: text(300), option: text(300),
      value: { type: 'number', minimum: -100000, maximum: 100000 },
    }, additionalProperties: false },
  }, required: ['entity_id', 'service'], additionalProperties: false,
};
const trigger = { oneOf: [
  { type: 'object', properties: { kind: { type: 'string', enum: ['state'] }, entity_id: text(300), to: text(300), attribute: text(100) },
    required: ['kind', 'entity_id', 'to'], additionalProperties: false },
  { type: 'object', properties: {
    kind: { type: 'string', enum: ['time'] }, at: text(100),
    calendar_reference: { type: 'object', properties: {
      boundary: { type: 'string', enum: ['start', 'end'] }, offset_seconds: { type: 'number', minimum: -31622400, maximum: 31622400 },
    }, required: ['boundary', 'offset_seconds'], additionalProperties: false },
  }, required: ['kind', 'at'], additionalProperties: false },
] };
const plan = { type: 'object', properties: {
  name: text(120),
  actions: { type: 'array', maxItems: 20, items: action },
  automations: { type: 'array', maxItems: 10, items: {
    type: 'object', properties: { name: text(120), triggers: { type: 'array', minItems: 1, maxItems: 10, items: trigger },
      actions: { type: 'array', minItems: 1, maxItems: 20, items: action } },
    required: ['name', 'triggers', 'actions'], additionalProperties: false,
  } },
  temporary_window: { type: 'object', properties: { starts_at: text(100), ends_at: text(100) },
    required: ['starts_at', 'ends_at'], additionalProperties: false },
  placement_requirements: { type: 'array', minItems: 1, maxItems: 40, items: {
    type: 'object', properties: { entity_id: text(200), area_id: text(200) }, required: ['entity_id', 'area_id'], additionalProperties: false,
  } },
  calendar_event: { type: 'object', properties: { event_id: text(300), event_start: text(100), event_revision: text(300) },
    required: ['event_id', 'event_start', 'event_revision'], additionalProperties: false },
}, required: ['name'], additionalProperties: false };

export const FAST_SCHEMA       = {
  name: 'living_home_fast',
  description: 'Read current capabilities and observed run_id. Author explicit plans only for advertised entities and services with device_profile=physical. plan_preview saves a proposal without executing it. plan_apply requires the exact observed draft_id, plan_hash, expected_run_id and actual user authorization. plan_execute is only for explicitly authorized immediate actions; no future rules. Inspect uncertain writes with plan_status. Never invent consent, change profiles, reset a home, or infer device operation from a health snapshot.',
  parameters: { type: 'object', properties: {
    operation: { type: 'string', enum: ['capabilities', 'plan_preview', 'plan_apply', 'plan_execute', 'plan_status'] },
    expected_run_id: id, device_profile: physical, draft_id: id,
    plan_hash: { type: 'string', minLength: 64, maxLength: 64, pattern: '^[a-f0-9]{64}$' },
    plan, authorization,
  }, required: ['operation'], additionalProperties: false },
};

export const PROPERTY_SCHEMA       = {
  name: 'living_home_property',
  description: 'Read local property records, source evidence and reports. health_snapshot collects current status for explicitly selected Home Assistant entities; it does not test physical operation. Use report/report_read for authorized local health or property reports. Read and preserve coverage gaps, stale observations and unknown causes. maintenance_read retrieves saved evidence and never diagnoses from missing data. repair_draft saves authored local text and never sends it. External sharing, Google publication, Gmail drafts, inventory automation and manual discovery are unavailable in this release.',
  parameters: { type: 'object', properties: {
    operation: { type: 'string', enum: ['status', 'health_snapshot', 'search', 'asset', 'incident', 'maintenance_read', 'report', 'report_read', 'repair_draft'] },
    query: text(2000), limit: { type: 'integer', minimum: 1, maximum: 25 }, asset_id: id, incident_id: id,
    report_id: id, report_type: { type: 'string', enum: ['general', 'health', 'maintenance'] },
    body_offset: { type: 'integer', minimum: 0, maximum: 40000 }, body_limit: { type: 'integer', minimum: 1, maximum: 4000 },
    detail: { type: 'string', enum: ['brief', 'full'] }, title: text(200), body: text(40000), subject: text(200), recipient: text(500),
    maintenance_revision: { type: 'string', minLength: 64, maxLength: 64, pattern: '^[a-f0-9]{64}$' },
    publish_to_drive: { type: 'boolean' }, save_to_gmail: { type: 'boolean' }, format: { type: 'string', enum: ['markdown'] },
  }, required: ['operation'], additionalProperties: false },
};


//# sourceURL=schemas.ts