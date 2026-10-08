# Source provenance

This clean adapter was authored for the separate Living Home installer on 2026-10-04. It preserves a bounded subset of the reviewed API/tool conventions in `C:\LivingHome\dev\living-home-openclaw`: plugin ID, two tool names, SDK tool factory pattern, explicit plan authorization, native structured tool results, and report route conventions. The small Node TypeScript stripping build script is adapted from that local package. The new client, health runner, schemas, validation, and operation routing were authored for this clean deployment.

The owner confirmed permission to publish the adapted original code and selected the MIT license on 2026-10-05. The root LICENSE applies to this adapter. The package remains `private: true` to prevent accidental npm publication; it can be distributed with the application to the requested internal NVIDIA repository. No configuration files, credentials, household state, private working artifacts, reference skills, source-specific presentation instructions, or household fixtures were copied.

OpenClaw itself is provided by the separately installed host. Its package and license are not included here. Production loading uses the host's `openclaw/plugin-sdk/tool-plugin`. Local unit tests avoid service startup; an optional SDK smoke test imports only the installed SDK module.
