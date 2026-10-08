import { execFile } from 'node:child_process';
import { readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { ContractError, redact,           } from './validation.js';

const MAX_BYTES = 8 * 1024 * 1024;
export const CONFIG_SCHEMA       = {
  type: 'object', properties: {
    baseUrl: { type: 'string' }, propertyBasePath: { type: 'string', default: '/property' },
    pythonExecutable: { type: 'string' }, collectorPath: { type: 'string' },
    dataDirectory: { type: 'string' }, apiTokenFile: { type: 'string' },
  }, required: ['baseUrl', 'pythonExecutable', 'collectorPath', 'dataDirectory', 'apiTokenFile'], additionalProperties: false,
};

                             
                                                                      
                                                                     
  

export function validateConfig(input      )                {
  if (!input || typeof input !== 'object' || Array.isArray(input)
      || Object.keys(input).some(key => !Object.hasOwn(CONFIG_SCHEMA.properties, key)))
    throw new ContractError('Living Home plugin configuration contains unsupported fields');
  let origin     ;
  try { origin = new URL(input.baseUrl); } catch { throw new ContractError('Configure a loopback Living Home backend origin'); }
  if (!['http:', 'https:'].includes(origin.protocol) || !['127.0.0.1', '[::1]'].includes(origin.hostname)
      || origin.username || origin.password || origin.pathname !== '/' || origin.search || origin.hash)
    throw new ContractError('Living Home backend must be a loopback origin without credentials, paths or queries');
  const propertyBasePath = input.propertyBasePath ?? '/property';
  if (typeof propertyBasePath !== 'string' || !/^\/[A-Za-z0-9_-]+(?:\/[A-Za-z0-9_-]+)*$/.test(propertyBasePath))
    throw new ContractError('Configure a canonical property API path prefix');
  for (const field of ['pythonExecutable', 'collectorPath', 'dataDirectory', 'apiTokenFile']) {
    if (typeof input[field] !== 'string' || !path.isAbsolute(input[field]) || /[\r\n\0]/.test(input[field]))
      throw new ContractError('Plugin executable, collector, data and token paths must be absolute local paths');
    try {
      const status = statSync(input[field]);
      if (field === 'dataDirectory' ? !status.isDirectory() : !status.isFile()) throw new Error();
    } catch { throw new ContractError('A configured Living Home local file or data directory is unavailable'); }
  }
  if (!input.collectorPath.endsWith('.py')) throw new ContractError('Configure the installed Python health collector');
  return { baseUrl: origin.origin, propertyBasePath, pythonExecutable: input.pythonExecutable,
    collectorPath: input.collectorPath, dataDirectory: input.dataDirectory, apiTokenFile: input.apiTokenFile };
}

function readToken(file        )         {
  try {
    const token = readFileSync(file, 'utf8').trim();
    if (token.length < 32 || token.length > 8192 || /\s/.test(token)) throw new Error();
    return token;
  } catch { throw new ContractError('Living Home backend API token file is missing or invalid'); }
}

function allowedRoute(config               , route        , writing         )          {
  if (/[#\\\r\n]/.test(route) || !route.startsWith('/') || route.includes('//')) return false;
  const [pathname, rawQuery, extra] = route.split('?');
  if (extra !== undefined || rawQuery === '' || pathname.includes('%')) return false;
  const query = new URLSearchParams(rawQuery || '');
  const keys = [...query.keys()];
  if (keys.some(key => query.getAll(key).length !== 1)) return false;
  if (writing) return !rawQuery && (['/device-plans/preview', '/device-plans/apply', '/device-plans/execute'].includes(pathname)
    || [config.propertyBasePath + '/report', config.propertyBasePath + '/repair-draft'].includes(pathname));
  if (pathname === '/device-plans/capabilities') return !rawQuery;
  if (pathname === '/device-plans/status') return keys.length === 1 && keys[0] === 'draft_id'
    && /^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}$/.test(query.get('draft_id') || '');
  if (pathname === config.propertyBasePath + '/status') return !rawQuery;
  if (pathname === config.propertyBasePath + '/search') return keys.length === 2 && keys.includes('q') && keys.includes('limit')
    && (query.get('q') || '').length >= 1 && (query.get('q') || '').length <= 2000
    && /^([1-9]|1[0-9]|2[0-5])$/.test(query.get('limit') || '');
  const suffix = pathname.slice(config.propertyBasePath.length);
  if (!pathname.startsWith(config.propertyBasePath + '/')) return false;
  if (/^\/(?:asset|incident)\/[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}$/.test(suffix)) return !rawQuery;
  if (/^\/maintenance\/[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}(?:\/brief)?$/.test(suffix)) return !rawQuery;
  if (/^\/reports\/[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}$/.test(suffix)) {
    for (const [key, value] of query) {
      if (key === 'report_type' && ['general', 'health', 'maintenance'].includes(value)) continue;
      if (key === 'incident_id' && /^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,159}$/.test(value) && value !== 'current') continue;
      if (key === 'body_offset' && /^(0|[1-9][0-9]*)$/.test(value) && Number(value) <= 40000) continue;
      if (key === 'body_limit' && /^[1-9][0-9]*$/.test(value) && Number(value) <= 4000) continue;
      return false;
    }
    return true;
  }
  return false;
}

export class HomeClient {
  config               ;
  fetcher              ;
  executor                 ;
  constructor(input      , dependencies       = {}) {
    this.config = validateConfig(input);
    this.fetcher = dependencies.fetch || fetch;
    this.executor = dependencies.execFile || execFile;
  }
  propertyRoute(suffix        )         { return this.config.propertyBasePath + suffix; }

  async request(route        , payload       , timeoutMs = 30000, signal              )                {
    if (!allowedRoute(this.config, route, payload !== undefined)) throw new ContractError('Unsupported Living Home API route');
    const token = readToken(this.config.apiTokenFile);
    const combined = signal ? AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]) : AbortSignal.timeout(timeoutMs);
    try {
      combined.throwIfAborted();
      const response = await this.fetcher(this.config.baseUrl + route, {
        method: payload === undefined ? 'GET' : 'POST',
        headers: { Accept: 'application/json', Authorization: 'Bearer ' + token,
          ...(payload === undefined ? {} : { 'Content-Type': 'application/json' }) },
        ...(payload === undefined ? {} : { body: JSON.stringify(payload) }), redirect: 'error', signal: combined,
      });
      const reader = response.body?.getReader();
      if (!reader) throw new Error();
      const chunks               = [];
      let count = 0;
      try {
        while (true) {
          const next = await reader.read();
          if (next.done) break;
          count += next.value.length;
          if (count > MAX_BYTES) throw new Error();
          chunks.push(next.value);
        }
      } finally { await reader.cancel(); }
      const result = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      if (!result || typeof result !== 'object' || Array.isArray(result)) throw new Error();
      if (!response.ok) return { ok: false, error: 'Living Home backend rejected the request', http_status: response.status,
        outcome_unverified: payload !== undefined, retry_policy: 'Read current evidence before retrying; no automatic retry was made.' };
      return redact(result, token);
    } catch {
      return { ok: false, error: combined.aborted ? 'Living Home request cancelled or timed out' : 'Living Home backend response unavailable',
        outcome_unverified: payload !== undefined, retry_policy: 'Read current evidence before retrying; no automatic retry was made.' };
    }
  }

  health(signal              )                { return collectHealth(this.config, signal, this.executor); }
}

export function collectHealth(config               , signal              , executor                  = execFile)                {
  if (signal?.aborted) return Promise.resolve({ ok: false, error: 'Health evidence collection cancelled' });
  const args = [config.collectorPath, '--env-file', path.join(config.dataDirectory, '.env'),
    '--config', path.join(config.dataDirectory, 'health-config.json'),
    '--output-dir', path.join(config.dataDirectory, 'health', 'snapshots')];
  const childEnv = { ...process.env, PYTHONUTF8: '1' };
  // This installed household's .env is authoritative; unrelated host credentials
  // must never select another Home Assistant instance for this plugin.
  delete childEnv.HA_URL;
  delete childEnv.HA_TOKEN;
  return new Promise(resolve => {
    try {
      executor(config.pythonExecutable, args, { cwd: config.dataDirectory, windowsHide: true, shell: false,
        timeout: 35000, maxBuffer: MAX_BYTES, encoding: 'utf8', signal, env: childEnv }, (error, stdout) => {
        if (error) {
          resolve({ ok: false, error: signal?.aborted ? 'Health evidence collection cancelled'
            : 'Health evidence collector failed or timed out; inspect local setup and credentials' });
          return;
        }
        try {
          const result = JSON.parse(stdout          );
          if (!result || typeof result !== 'object' || Array.isArray(result)
              || !['available', 'attention', 'unknown'].includes(result.status)
              || typeof result.checked_at !== 'string' || !result.coverage) throw new Error();
          resolve(redact(result, readToken(config.apiTokenFile)));
        } catch { resolve({ ok: false, error: 'Health evidence collector returned invalid JSON or local API token is unavailable' }); }
      });
    } catch { resolve({ ok: false, error: 'Health evidence collector could not be started' }); }
  });
}


//# sourceURL=client.ts