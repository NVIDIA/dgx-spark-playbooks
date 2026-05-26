// OpenClaw os.networkInterfaces() shim for OpenShell sandboxes.
//
// Why: OpenClaw 2026.3.x calls os.networkInterfaces() at startup
// (pickPrimaryLanIPv4 -> initSelfPresence). The OpenShell sandbox kernel
// blocks the netlink syscall libuv uses, so the call throws
// `uv_interface_addresses returned Unknown system error 1` and the
// gateway crashes before binding port 18789.
//
// Fix: catch the error and return a deterministic localhost-only
// interface list. Self-presence advertising still works; the gateway
// just reports 127.0.0.1 as its LAN IP, which is fine for a sandbox
// that only ever serves loopback or the proxy-forwarded port.
//
// Wire-up: setup_sandbox.sh and restart_sandbox.sh export
//   NODE_OPTIONS="--require /sandbox/clinical-intelligence/scripts/openclaw-os-shim.js"
// before invoking `openclaw gateway run`.

const os = require('os');
const original = os.networkInterfaces;

os.networkInterfaces = function safeNetworkInterfaces() {
  try {
    return original.call(os);
  } catch (err) {
    const msg = (err && err.message) || '';
    const code = err && err.code;
    if (code === 'ERR_SYSTEM_ERROR' || /uv_interface_addresses/.test(msg)) {
      return {
        lo: [{
          address: '127.0.0.1',
          netmask: '255.0.0.0',
          family: 'IPv4',
          mac: '00:00:00:00:00:00',
          internal: true,
          cidr: '127.0.0.1/8',
        }],
        eth0: [{
          address: '127.0.0.1',
          netmask: '255.0.0.0',
          family: 'IPv4',
          mac: '00:00:00:00:00:00',
          internal: false,
          cidr: '127.0.0.1/8',
        }],
      };
    }
    throw err;
  }
};
