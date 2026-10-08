import assert from 'node:assert/strict';
import plugin from '../dist/index.js';
assert.equal(plugin.id, 'living-home');
assert.equal(typeof plugin.register, 'function');
console.log('Portable plugin imports through the separately installed OpenClaw SDK; no service or tool execution performed.');
