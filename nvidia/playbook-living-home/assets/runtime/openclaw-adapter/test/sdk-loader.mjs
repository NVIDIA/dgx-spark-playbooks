import { registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
const host = process.env.OPENCLAW_PACKAGE_ROOT;
if (!host || !path.isAbsolute(host)) throw new Error('Set an absolute OPENCLAW_PACKAGE_ROOT for the optional SDK import smoke check');
registerHooks({ resolve(specifier, context, nextResolve) {
  if (specifier === 'openclaw/plugin-sdk/tool-plugin') return {
    shortCircuit: true, url: pathToFileURL(path.join(host, 'dist', 'plugin-sdk', 'tool-plugin.js')).href,
  };
  return nextResolve(specifier, context);
} });
