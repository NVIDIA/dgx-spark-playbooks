import copy
import json
from pathlib import Path
import unittest

import configure_home_report as cron


class FakeGateway:
    """Validate actual helper argv without invoking any installed CLI/job."""
    def __init__(self, job=None):
        self.jobs = [copy.deepcopy(job)] if job else []
        self.calls = []

    def __call__(self, args):
        self.calls.append(list(args))
        action = args[1]
        if action == "list":
            return {"jobs": copy.deepcopy(self.jobs), "hasMore": False}
        if action == "show":
            return copy.deepcopy(next(j for j in self.jobs if j["id"] == args[2]))
        if action not in {"add", "edit"}:
            raise AssertionError("Unexpected action or attempt to run/enable a real job")
        value = lambda key: args[args.index(key) + 1]
        job = copy.deepcopy(self.jobs[0]) if action == "edit" else {"id": "test-observed-id", "declarationKey": value("--declaration-key")}
        self.jobs = [job]
        job.update(name=value("--name"), description=value("--description"), agentId=value("--agent"),
                   sessionTarget=value("--session"), wakeMode=value("--wake"), enabled=False)
        job["schedule"] = {"kind": "cron", "expr": value("--cron"), "tz": value("--tz"), "staggerMs": 0}
        job["payload"] = {"kind": "agentTurn", "message": value("--message"), "timeoutSeconds": int(value("--timeout-seconds"))}
        job["delivery"] = {"mode": "none"} if "--no-deliver" in args else {
            "mode": "announce", "channel": value("--channel"), "to": value("--to")}
        if "--account" in args:
            job["delivery"]["accountId"] = value("--account")
        for key in ("sessionKey", "trigger", "pacing", "failureAlert"):
            job.pop(key, None)
        self.assert_flags(args, action)
        return copy.deepcopy(job)

    @staticmethod
    def assert_flags(args, action):
        if "--exact" not in args or ("--disabled" if action == "add" else "--disable") not in args:
            raise AssertionError("Job must be saved exact and disabled")
        if action == "edit" and not {"--clear-session-key", "--clear-model", "--clear-tools", "--no-failure-alert"} <= set(args):
            raise AssertionError("Updates must clear inherited execution/delivery overrides")


class CronTests(unittest.TestCase):
    def setUp(self):
        self.cfg = cron.config(json.loads(Path(__file__).with_name("home-report.example.json").read_text()))

    def test_disabled_local_default_then_idempotent_no_mutation(self):
        fake = FakeGateway()
        created = cron.ensure(self.cfg, fake)
        self.assertTrue(created["changed"])
        self.assertFalse(created["enabled"])
        self.assertEqual(fake.jobs[0]["delivery"], {"mode": "none"})
        self.assertEqual(fake.jobs[0]["schedule"]["tz"], "America/Los_Angeles")
        self.assertIn("--no-deliver", fake.calls[1])
        self.assertEqual(fake.calls[1][fake.calls[1].index("--channel") + 1], "")
        fake.calls.clear()
        second = cron.ensure(self.cfg, fake)
        self.assertFalse(second["changed"])
        self.assertEqual([a[1] for a in fake.calls], ["list", "show"])

    def test_repeat_preserves_manually_enabled_job(self):
        fake = FakeGateway()
        cron.ensure(self.cfg, fake)
        fake.jobs[0]["enabled"] = True
        fake.calls.clear()
        result = cron.ensure(self.cfg, fake)
        self.assertTrue(result["enabled"])
        self.assertFalse(result["changed"])
        self.assertNotIn("edit", [a[1] for a in fake.calls])

    def test_schedule_change_updates_observed_id_and_disables(self):
        fake = FakeGateway()
        cron.ensure(self.cfg, fake)
        fake.jobs[0]["enabled"] = True
        self.cfg["cron"] = "15 7 * * *"
        fake.calls.clear()
        result = cron.ensure(self.cfg, fake)
        self.assertFalse(result["enabled"])
        self.assertEqual(fake.calls[1][0:3], ["cron", "edit", "test-observed-id"])
        self.assertEqual(result["schedule"]["expr"], "15 7 * * *")

    def test_optional_delivery_uses_only_explicit_channel_destination(self):
        self.cfg["delivery"] = {"mode": "announce", "channel": "discord", "to": "user-configured-test-target", "accountId": "own-account"}
        self.cfg = cron.config(self.cfg)
        fake = FakeGateway()
        cron.ensure(self.cfg, fake)
        self.assertEqual(fake.jobs[0]["delivery"], self.cfg["delivery"])
        args = fake.calls[1]
        self.assertIn("--announce", args)
        self.assertEqual(args[args.index("--to") + 1], self.cfg["delivery"]["to"])

    def test_switching_delivery_removes_previous_targets(self):
        fake = FakeGateway()
        cron.ensure(self.cfg, fake)
        fake.jobs[0]["delivery"] = {"mode": "announce", "channel": "discord", "to": "old-target", "accountId": "old-account"}
        cron.ensure(self.cfg, fake)
        self.assertEqual(fake.jobs[0]["delivery"], {"mode": "none"})
        self.assertIn("--clear-to", fake.calls[-2])

    def test_foreign_name_collision_refused_without_writes(self):
        fake = FakeGateway({"id": "foreign", "name": cron.NAME, "declarationKey": "another-task"})
        with self.assertRaises(ValueError):
            cron.ensure(self.cfg, fake)
        self.assertEqual([a[1] for a in fake.calls], ["list"])

    def test_readback_schedule_mismatch_refused(self):
        fake = FakeGateway()
        def bad_readback(args):
            result = fake(args)
            if args[1] == "show":
                result["schedule"]["tz"] = "UTC"
            return result
        with self.assertRaises(ValueError):
            cron.ensure(self.cfg, bad_readback)

    def test_workflow_collects_saves_reads_without_device_or_external_writes(self):
        text = cron.report_message("none")
        for fragment in ("operation=health_snapshot", "report_type=health", "publish_to_drive=false", "report_id=latest", "Do not control devices", "do not post to Discord"):
            self.assertIn(fragment, text)

    def test_implicit_delivery_or_bad_profile_rejected(self):
        for update in ({"delivery": {"mode": "announce", "channel": "last", "to": "abc"}},
                       {"openclawProfile": "../private"}, {"timeZone": "Pacific Standard Time"}, {"cron": "bad cron"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                cron.config(dict(self.cfg, **update))


if __name__ == "__main__":
    unittest.main()
