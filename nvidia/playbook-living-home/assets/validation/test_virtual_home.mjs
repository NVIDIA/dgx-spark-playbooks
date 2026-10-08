import fs from 'node:fs';
import { createTools } from '/opt/livinghome-validation/app/runtime/openclaw-adapter/dist/tools.js';
import { HomeClient } from '/opt/livinghome-validation/app/runtime/openclaw-adapter/dist/client.js';

const data = '/opt/livinghome-validation/household';
const origin = 'http://127.0.0.1:18081';
const apiToken = fs.readFileSync(data + '/api-token.txt', 'utf8').trim();
const haToken = fs.readFileSync(data + '/.env', 'utf8').split('\n').find(x => x.startsWith('HA_TOKEN=')).slice(9).trim();
const haOrigin = 'http://127.0.0.1:18123';
const config = {baseUrl:origin, pythonExecutable:'/usr/bin/python3', collectorPath:'/opt/livinghome-validation/app/runtime/health/collect_home_status.py',dataDirectory:data,apiTokenFile:data+'/api-token.txt'};
const client = new HomeClient(config);
const definitions = createTools(client);
const results = [];
const started = new Date().toISOString();
let sequence = 0;
const assert = (condition, detail) => { if(!condition) throw new Error(detail); };
const sleep = ms => new Promise(r=>setTimeout(r,ms));
async function api(path, body) {
  const response = await fetch(origin+path,{method:body===undefined?'GET':'POST',headers:{Authorization:'Bearer '+apiToken,'Content-Type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)})});
  const value = await response.json();
  if(!response.ok) throw new Error(JSON.stringify(value));
  return value;
}
async function ha(path, body) {
  const response = await fetch(haOrigin+path,{method:body===undefined?'GET':'POST',headers:{Authorization:'Bearer '+haToken,'Content-Type':'application/json'},...(body===undefined?{}:{body:JSON.stringify(body)})});
  const value = await response.json();
  if(!response.ok) throw new Error('HA HTTP '+response.status);
  return value;
}
async function tool(name,args,success=true) {
  const result = await definitions.find(t=>t.name===name).execute('lab-'+Date.now()+'-'+(++sequence),args);
  const value = result.details;
  if(success && (result.isError || value.ok===false)) throw new Error(JSON.stringify(value));
  return value;
}
const fast = (args,success=true)=>tool('living_home_fast',args,success);
const property = (args,success=true)=>tool('living_home_property',args,success);
async function caseRun(name,run) {
  try { const detail=await run(); results.push({name,status:'PASS',detail}); console.log('PASS '+name); }
  catch(error) { results.push({name,status:'FAIL',detail:String(error.message).replaceAll(apiToken,'[redacted]').replaceAll(haToken,'[redacted]')});console.log('FAIL '+name+': '+results.at(-1).detail); }
}
async function waitState(id,state,seconds=15) {
  const end = Date.now()+seconds*1000;
  do { const value=await ha('/api/states/'+id); if(value.state===state) return value; await sleep(250); } while(Date.now()<end);
  throw new Error('State '+state+' was not observed for '+id);
}
const consent = {confirmed:true,user_request:'Authorized isolated test using only the virtual devices in LivingHome-Validation.'};
let capabilities, immediateDraft, automationDraft, automationReceipt, snapshot, savedReport, incident;
const rules=[];
const originalHealthConfig=fs.readFileSync(data+'/health-config.json','utf8');
try {
await caseRun('unauthenticated backend rejects reads',async()=>{assert((await fetch(origin+'/healthz')).status===401,'Expected401');});
await caseRun('new household starts with empty records',async()=>{const p=await property({operation:'status'});assert(p.sources===0&&p.recent_reports.length===0&&p.recent_incidents.length===0,'Household was not empty');return {property_id:p.property_id};});
await caseRun('portable plugin reads exactly four selected HA entities',async()=>{
  capabilities=await fast({operation:'capabilities'}); const ids=capabilities.entities.map(e=>e.entity_id).sort();
  assert(JSON.stringify(ids)===JSON.stringify(['binary_sensor.lab_hallway_motion','light.lab_desk_lamp','sensor.lab_sensor_battery','switch.lab_desk_plug']),'Wrong selected entity scope');return {entities:ids};
});
const context={expected_run_id:capabilities?.run_id,device_profile:'physical'};
const lampAction={entity_id:'light.lab_desk_lamp',service:'light.turn_on',data:{}};
await caseRun('preview leaves virtual lamp off',async()=>{
  await ha('/api/services/light/turn_off',{entity_id:'light.lab_desk_lamp'});
  immediateDraft=await fast({operation:'plan_preview',...context,plan:{name:'Lamp test',actions:[lampAction]}});
  assert((await ha('/api/states/light.lab_desk_lamp')).state==='off','Preview changed the lamp');
});
await caseRun('unselected target is refused',async()=>{
  const r=await fast({operation:'plan_preview',...context,plan:{name:'Scope check',actions:[{entity_id:'light.some_other_home',service:'light.turn_on',data:{}}]}},false);assert(r.ok===false,'Target outside selection accepted');
});
await caseRun('apply without confirmation is refused',async()=>{
  const r=await fast({operation:'plan_apply',...context,draft_id:immediateDraft.draft_id,plan_hash:immediateDraft.plan_hash,authorization:{...consent,confirmed:false}},false);assert(r.ok===false,'Unconfirmed write accepted');assert((await ha('/api/states/light.lab_desk_lamp')).state==='off','Refused write changed state');
});
await caseRun('approved plan switches lamp on and checks HA state',async()=>{
  const r=await fast({operation:'plan_apply',...context,draft_id:immediateDraft.draft_id,plan_hash:immediateDraft.plan_hash,authorization:consent});assert(r.status==='applied'&&r.actions[0].verified,'Action not verified');await waitState('light.lab_desk_lamp','on');return {status:r.status,verification_scope:r.actions[0].verification_scope};
});
await caseRun('native automation definition is saved and loaded',async()=>{
  await ha('/api/services/switch/turn_off',{entity_id:'switch.lab_desk_plug'});
  await ha('/api/services/light/turn_off',{entity_id:'light.lab_desk_lamp'});
  automationDraft=await fast({operation:'plan_preview',...context,plan:{name:'Desk workspace intent fixture',automations:[{name:'Lab desk plug lights lamp',triggers:[{kind:'state',entity_id:'switch.lab_desk_plug',to:'on'}],actions:[lampAction]}]}});
  automationReceipt=await fast({operation:'plan_apply',...context,draft_id:automationDraft.draft_id,plan_hash:automationDraft.plan_hash,authorization:consent});
  const rule=automationReceipt.automations[0];if(rule?.entity_id)rules.push(rule.entity_id);assert(rule?.loaded&&rule.enabled&&rule.definition_readback_matches,'Rule not active or not matched');return {automation_id:rule.automation_id,entity_id:rule.entity_id};
});
await caseRun('real HA state transition triggers the saved automation',async()=>{
  await ha('/api/services/switch/turn_on',{entity_id:'switch.lab_desk_plug'});
  await waitState('light.lab_desk_lamp','on');
  const status=await fast({operation:'plan_status',draft_id:automationDraft.draft_id});assert(status.current_automations?.some(r=>r.last_triggered),'No last-triggered evidence');return {last_triggered:status.current_automations[0].last_triggered};
});
await caseRun('health snapshot identifies low battery and treats off as available',async()=>{
  await ha('/api/services/light/turn_off',{entity_id:'light.lab_desk_lamp'});
  snapshot=await property({operation:'health_snapshot'});
  assert(snapshot.home_assistant.states_collected&&snapshot.summary.selected_entities===4,'Missing fresh coverage');
  assert(snapshot.summary.reported_low_batteries===1,'18 percent battery not flagged');
  assert(snapshot.inventory.find(e=>e.entity_id==='light.lab_desk_lamp').availability==='available','Off lamp called unavailable');
  return {checked_at:snapshot.checked_at,summary:snapshot.summary};
});
await caseRun('unavailable battery sensor stays unavailable',async()=>{
  await ha('/api/services/input_boolean/turn_off',{entity_id:'input_boolean.lab_sensor_online'});
  await waitState('sensor.lab_sensor_battery','unavailable');
  const r=await property({operation:'health_snapshot'});assert(r.summary.unavailable_or_unknown_entities===1,'Unavailable sensor not counted');assert(r.summary.reported_low_batteries===0,'Unavailable value reused as current battery');
});
await caseRun('missing selected entity stays unknown',async()=>{
  const cfg=JSON.parse(originalHealthConfig);cfg.entity_ids.push('sensor.lab_missing');fs.writeFileSync(data+'/health-config.json',JSON.stringify(cfg));
  const r=await property({operation:'health_snapshot'});assert(r.summary.missing_selected_entities===1,'Missing entity hidden');
  fs.writeFileSync(data+'/health-config.json',originalHealthConfig);
});
await caseRun('local health report saves and retrieves matching ID',async()=>{
  await ha('/api/services/input_boolean/turn_on',{entity_id:'input_boolean.lab_sensor_online'});
  await waitState('sensor.lab_sensor_battery','18.0');snapshot=await property({operation:'health_snapshot'});
  const body='Integration test report authored by the test harness, not an AI conversation. Checked '+snapshot.checked_at+'. Selected '+snapshot.summary.selected_entities+' entities. Low batteries: '+snapshot.summary.reported_low_batteries+'. Physical operation was not tested.';
  const written=await property({operation:'report',title:'Home Device Status validation',body,report_type:'health',publish_to_drive:false,format:'markdown'});savedReport=written.report;
  const read=await property({operation:'report_read',report_id:'latest',report_type:'health'});assert(read.available&&read.report.id===savedReport.id&&read.report.body===body,'Saved report mismatch');return {report_id:savedReport.id,source_observed_at:read.report.source_dates.source_observed_at};
});
await caseRun('report reads do not refresh evidence',async()=>{
  const file=data+'/health/snapshots/latest.json';const before=fs.readFileSync(file,'utf8');await property({operation:'report_read',report_id:savedReport.id,report_type:'health'});assert(fs.readFileSync(file,'utf8')===before,'Read triggered collection');
});
await caseRun('report durable idempotency refuses changed payload',async()=>{
  const body={title:'Idempotency test',body:'Local test record.',report_type:'general',publish_to_drive:false,format:'markdown',idempotency_key:'validation-report-once'};
  const first=await api('/property/report',body),second=await api('/property/report',body);assert(first.report.id===second.report.id,'Duplicate report');
  let rejected=false;try{await api('/property/report',{...body,body:'Changed content'})}catch{rejected=true}assert(rejected,'Idempotency conflict accepted');
});
await caseRun('own source and new maintenance incident can be imported',async()=>{
  await api('/property/source',{source_id:'lab-manual',asset_id:'lab-lamp',source_kind:'manual',title:'Validation lamp instructions',body:'The test lamp is an on/off control.',source_uri:'local:validation-lamp-instructions'});
  const imported=await api('/property/incidents',{asset_id:'lab-lamp',title:'Lab lamp observation',observations:'The lamp state was off when checked. No physical defect is established.',observed_at:new Date().toISOString(),source_ids:['lab-manual']});incident=imported.incident;
  const read=await property({operation:'maintenance_read',incident_id:incident.id,detail:'full'});assert(read.available&&read.sources[0].source_id==='lab-manual'&&read.evidence_revision.length===64,'Incident source binding missing');return {incident_id:incident.id};
});
await caseRun('repair draft stays local and unsent',async()=>{
  const r=await property({operation:'repair_draft',asset_id:'lab-lamp',incident_id:incident.id,subject:'Review lamp observation',body:'Please review the supplied observation; no cause has been established.',save_to_gmail:false});assert(r.sent===false&&r.draft.status==='saved_local','Draft was not local');
});
await caseRun('external report publication is refused',async()=>{const r=await property({operation:'report',title:'No cloud write',body:'Scope test',publish_to_drive:true},false);assert(r.ok===false,'Cloud publication accepted');});
} finally {
  fs.writeFileSync(data+'/health-config.json',originalHealthConfig);
  for(const entity_id of rules) await ha('/api/services/automation/turn_off',{entity_id});
  await ha('/api/services/light/turn_off',{entity_id:'light.lab_desk_lamp'});
  await ha('/api/services/switch/turn_off',{entity_id:'switch.lab_desk_plug'});
  await ha('/api/services/input_boolean/turn_on',{entity_id:'input_boolean.lab_sensor_online'});
  const report={started_at:started,finished_at:new Date().toISOString(),environment:'ARM64 Linux WSL2 with real Home Assistant 2026.9.4 and virtual entities',path:'Actual portable plugin -> authenticated household API -> real Home Assistant REST/config/services',windows_installer_executed:false,model_conversation_tested:false,cron_tested:false,physical_devices_tested:false,results,cleanup:{automations_disabled:rules,virtual_lamp:'off',virtual_plug:'off',sensor_available:true}};
  fs.writeFileSync('/mnt/c/LivingHome-installer/validation/evidence/virtual-home-results.json',JSON.stringify(report,null,2));
  console.log(JSON.stringify({passed:results.filter(r=>r.status==='PASS').length,failed:results.filter(r=>r.status==='FAIL').length,total:results.length}));
  process.exitCode=results.some(r=>r.status==='FAIL')?1:0;
}
