# Development and review

Use a separate checkout and fresh household state. The supplied virtual-device lab is optional; never run its fixture actions against a real household. Do not copy a live Home Assistant, OpenClaw, Google or Discord configuration into this repository.

Source checks require Python 3.12+ with IANA timezone data, Node 24.16+, and Windows PowerShell for the launcher/bootstrapper build. Run `scripts/Test-Project.ps1 -Python <python> -Node <node> -WindowsBinaries` on Windows. The tests create temporary fixtures, not real device effects.

Before staging, run `python scripts/audit_repository.py` and review the exact file list. It checks Git-visible source, including untracked files and accidentally tracked ignored files. It does not read ignored credentials and never prints matching secret values. A passing focused scan does not replace reviewing source changes or the repository's secret scanning.

The packaged browser test uses an explicitly supplied private lab account and real Home Assistant with virtual entities. See `validation/test_workspace_browser.mjs`; it requires Playwright with Edge and the isolated HA fixtures. It exercises only the actions in that script and records whether models, cron or physical devices were tested. Do not turn a passing fixture test into a broader release claim.

Use the workspace-preview builder only for its documented scope. The strict full-release builder intentionally refuses incomplete gates. Do not flip readiness booleans to bypass missing implementation or verification. Keep user-facing wording, the setup guide and the playbook in sync when changing screens or setup steps.

## Release status

The workspace package requires existing Home Assistant. Model downloads, complete OpenClaw onboarding, Google account integration and the full Spark installer remain development work. Model-generated plans, actual cron runs, clean Windows installation and physical GPU/device operation still need end-to-end validation. See `validation/README.md`, `workflows/FRESH_HOST_CHECKLIST.md`, and `packaging/README.md` for developer verification and release requirements.

The standalone GitHub Actions workflow is not included in this nested source import. To add Windows source checks to GitLab CI, configure a Windows runner to execute `scripts/Test-Project.ps1` from this assets directory with Python and Node installed. The parent repository checks playbook metadata separately. Source checks must use test fixtures and must not deploy automatically.
