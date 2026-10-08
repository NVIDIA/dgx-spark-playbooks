"""Reviewed plan journal and validation primitives from the local Living Home source.

No service globals, reset routes, household fixtures or local HA filesystem writes.
See PROVENANCE.json for original source hashes. HouseholdPlanStore supplies scope,
transport and native automation persistence.
"""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid
import plan_evidence

ATTRIBUTES = {"friendly_name", "brightness", "color_temp_kelvin", "rgb_color", "color_mode", "xy_color",
              "supported_color_modes", "min_color_temp_kelvin", "max_color_temp_kelvin",
              "temperature", "current_temperature", "min_temp", "max_temp", "target_temp_step",
              "temperature_unit", "hvac_modes", "percentage", "percentage_step", "current_position",
              "source", "source_list", "options", "min", "max", "step", "unit_of_measurement",
              "supported_features", "entity_id", "device_class"}

def _utc():
    return dt.datetime.now(dt.timezone.utc)

def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def _atomic(path, value, *, as_yaml=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    filename = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            filename = handle.name
            if as_yaml:
                raise ValueError("Only JSON journals are supported")
            else:
                json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(filename, path)
    finally:
        if filename and os.path.exists(filename):
            os.unlink(filename)

def _keys(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError("Unexpected or missing fields: " + ", ".join(required))

def _text(value, name, limit=120):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Invalid " + name)
    if any(marker in value for marker in ("{{", "{%", "{#")):
        raise ValueError("Templates are not supported")
    return value

def _number(value, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError("Number outside advertised bounds")
    return value

def _timestamp(value):
    value = _text(value, "timestamp", 80)
    stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Time triggers require an explicit timezone offset")
    return stamp

def _service(fields=None, required=()):
    return {"fields": fields or {}, "required": list(required)}

def _num(low, high):
    return {"type": "number", "minimum": low, "maximum": high}

def _enum(values):
    return {"type": "string", "enum": values}

def _rgb_xy(value):
    """Same wide-RGB D65 conversion and 3-place rounding as HA util.color.

    XY lamps report a lossy RGB reconstruction. Compare their native chromaticity,
    not that reconstructed RGB triplet; brightness remains independently checked.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 255 for v in value):
        return None
    channels = [v / 255 for v in value]
    r, g, b = [((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in channels]
    x = r * 0.664511 + g * 0.154324 + b * 0.162028
    y = r * 0.283881 + g * 0.668433 + b * 0.047685
    z = r * 0.000088 + g * 0.072310 + b * 0.986039
    return (round(x / (x + y + z), 3), round(y / (x + y + z), 3)) if x + y + z else (0.0, 0.0)

def _xy_color_matches(requested, attrs):
    target = _rgb_xy(requested)
    observed = attrs.get("xy_color")
    if target is None or not isinstance(observed, (list, tuple)) or len(observed) != 2:
        return False
    # HA rounds its conversion to .001; Zigbee then quantizes coordinates.
    return all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
               and 0 <= v <= 1 and abs(v - wanted) <= 0.001
               for v, wanted in zip(observed, target))

def _services(eid, attrs):
    domain = eid.split(".", 1)[0]
    services = {}
    if domain in {"light", "switch", "fan", "input_boolean", "media_player"}:
        services = {domain + ".turn_on": _service(), domain + ".turn_off": _service()}
    if domain == "light":
        fields = {}
        modes = set(attrs.get("supported_color_modes") or [])
        if modes - {"onoff", "unknown"} or "brightness" in attrs:
            fields["brightness"] = _num(1, 255)
        if "color_temp" in modes and attrs.get("min_color_temp_kelvin") and attrs.get("max_color_temp_kelvin"):
            fields["color_temp_kelvin"] = _num(attrs["min_color_temp_kelvin"], attrs["max_color_temp_kelvin"])
        if modes.intersection({"rgb", "rgbw", "rgbww", "hs", "xy"}):
            fields["rgb_color"] = {"type": "array", "items": _num(0, 255), "minItems": 3, "maxItems": 3}
        services["light.turn_on"] = _service(fields)
    if domain == "fan" and ("percentage" in attrs or int(attrs.get("supported_features", 0)) & 1):
        services["fan.set_percentage"] = _service({"percentage": _num(0, 100)}, ["percentage"])
    if domain == "cover":
        features = int(attrs.get("supported_features", 0))
        services = {name: _service() for name, mask in (("cover.open_cover", 1), ("cover.close_cover", 2), ("cover.stop_cover", 8)) if features & mask}
        if features & 4:
            services["cover.set_cover_position"] = _service({"position": _num(0, 100)}, ["position"])
    if domain == "climate":
        if attrs.get("min_temp") is not None and attrs.get("max_temp") is not None:
            services["climate.set_temperature"] = _service({"temperature": _num(attrs["min_temp"], attrs["max_temp"])}, ["temperature"])
        if attrs.get("hvac_modes"):
            services["climate.set_hvac_mode"] = _service({"hvac_mode": _enum(attrs["hvac_modes"])}, ["hvac_mode"])
    if domain == "media_player" and attrs.get("source_list"):
        services["media_player.select_source"] = _service({"source": _enum(attrs["source_list"])}, ["source"])
    if domain == "input_select" and attrs.get("options"):
        services["input_select.select_option"] = _service({"option": _enum(attrs["options"])}, ["option"])
    if domain == "input_number" and attrs.get("min") is not None and attrs.get("max") is not None:
        services["input_number.set_value"] = _service({"value": _num(attrs["min"], attrs["max"])}, ["value"])
    return services

class PlanJournal:
    def _load(self):
        if not self.ledger.exists():
            return {"drafts": {}, "requests": {}}
        with self.ledger.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data.get("drafts"), dict) or not isinstance(data.get("requests"), dict):
            raise ValueError("Invalid device plan ledger")
        if not isinstance(data.get("executions", {}), dict):
            raise ValueError("Invalid device execution ledger")
        return data

    def _rows(self):
        rows = self.ha("GET", "/api/states", timeout=8)
        if not isinstance(rows, list):
            raise ValueError("Home Assistant state read failed")
        return {row["entity_id"]: row for row in rows if isinstance(row, dict) and isinstance(row.get("entity_id"), str)}

    def _validate_plan(self, plan, capabilities):
        _keys(plan, {"name", "actions", "automations", "calendar_event", "temporary_window", "placement_requirements"}, {"name"})
        name = _text(plan["name"], "plan name")
        catalog = {entry["entity_id"]: entry for entry in capabilities["entities"]}
        actions, automations = plan.get("actions", []), plan.get("automations", [])
        if not isinstance(actions, list) or len(actions) > 20 or not isinstance(automations, list) or len(automations) > 10 or not (actions or automations):
            raise ValueError("Plan needs actions or automations within advertised limits")
        result = {"name": name, "actions": [self._validate_action(action, catalog) for action in actions], "automations": []}
        calendar_event = None
        if "calendar_event" in plan:
            binding = plan["calendar_event"]
            _keys(binding, {"event_id", "event_start", "event_revision"}, {"event_id", "event_start", "event_revision"})
            for key, value in binding.items():
                _text(value, key, 500)
            data = self.calendar_provider()
            if not isinstance(data, dict) or data.get("available") is False or data.get("ok") is False or data.get("google_error"):
                raise ValueError("Calendar evidence unavailable; event-bound plan cannot be validated")
            matches = [event for event in data.get("events", []) if event.get("id") == binding["event_id"] and event.get("start") == binding["event_start"]]
            if len(matches) != 1 or matches[0].get("event_revision") != binding["event_revision"]:
                raise ValueError("Calendar event changed or is absent; read it and preview a new plan")
            calendar_event = matches[0]
            result["calendar_event"] = copy.deepcopy(binding)
        if "placement_requirements" in plan:
            requirements = plan["placement_requirements"]
            if not isinstance(requirements, list) or not 1 <= len(requirements) <= 40:
                raise ValueError("Placement requirements need one to forty exact entity/area pairs")
            seen = set()
            for requirement in requirements:
                _keys(requirement, {"entity_id", "area_id"}, {"entity_id", "area_id"})
                eid = _text(requirement["entity_id"], "entity_id", 200)
                area = _text(requirement["area_id"], "area_id", 200)
                observed = catalog.get(eid, {}).get("placement", {})
                if eid in seen or observed.get("available") is not True or observed.get("area_id") != area:
                    raise ValueError("Required entity placement is missing, changed or ambiguous: " + eid)
                seen.add(eid)
            result["placement_requirements"] = copy.deepcopy(requirements)
        for auto in automations:
            _keys(auto, {"name", "triggers", "actions"}, {"name", "triggers", "actions"})
            if not isinstance(auto["triggers"], list) or not 1 <= len(auto["triggers"]) <= 10 or not isinstance(auto["actions"], list) or not 1 <= len(auto["actions"]) <= 20:
                raise ValueError("Invalid automation trigger/action count")
            triggers = []
            for trigger in auto["triggers"]:
                if not isinstance(trigger, dict):
                    raise ValueError("Invalid trigger")
                if trigger.get("kind") == "time":
                    _keys(trigger, {"kind", "at", "calendar_reference"}, {"kind", "at"})
                    at = _timestamp(trigger["at"])
                    if not self.now() + dt.timedelta(seconds=5) < at <= self.now() + dt.timedelta(days=366):
                        raise ValueError("Time trigger must be future and within 366 days")
                    canonical = {"kind": "time", "at": at.isoformat()}
                    if "calendar_reference" in trigger:
                        if calendar_event is None:
                            raise ValueError("Calendar-derived trigger requires an exact calendar_event ID/start/revision binding")
                        expected = plan_evidence.calendar_time(trigger["calendar_reference"], calendar_event, _timestamp, _number)
                        if at != expected:
                            raise ValueError("Trigger time differs from its observed calendar boundary and authored offset")
                        canonical["calendar_reference"] = copy.deepcopy(trigger["calendar_reference"])
                    triggers.append(canonical)
                elif trigger.get("kind") == "state":
                    _keys(trigger, {"kind", "entity_id", "to", "attribute"}, {"kind", "entity_id", "to"})
                    entity = catalog.get(trigger["entity_id"])
                    if not entity or not entity["available"] or entity.get("trigger_available") is False:
                        raise ValueError("Trigger entity unavailable in current profile")
                    attribute = trigger.get("attribute")
                    if attribute is not None and attribute not in entity["trigger_attributes"]:
                        raise ValueError("Trigger attribute not observed/advertised")
                    target = trigger["to"]
                    if isinstance(target, str):
                        _text(target, "trigger to", 200)
                    elif attribute is None or isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
                        raise ValueError("Trigger value must be a state string or numeric observed attribute")
                    options = entity["attributes"].get("options") if attribute is None else entity["attributes"].get("source_list") if attribute == "source" else None
                    if options and target not in options:
                        raise ValueError("Trigger value not present in observed options")
                    triggers.append(copy.deepcopy(trigger))
                else:
                    raise ValueError("Unsupported trigger kind")
            result["automations"].append({"name": _text(auto["name"], "automation name"), "triggers": triggers,
                                          "actions": [self._validate_action(action, catalog) for action in auto["actions"]]})
        if "temporary_window" in plan:
            result["temporary_window"] = copy.deepcopy(plan["temporary_window"])
            coverage = plan_evidence.temporary_coverage(result, _timestamp, self.now(),
                {eid: row["group_members"] for eid, row in catalog.items() if row.get("group_members")})
            result["temporary_window"] = {key: coverage[key] for key in ("starts_at", "ends_at")}
        return result

    def _planning_evidence(self, plan, capabilities):
        catalog = {entry["entity_id"]: entry for entry in capabilities["entities"]}
        groups = {eid: row["group_members"] for eid, row in catalog.items() if row.get("group_members")}
        coverage = plan_evidence.temporary_coverage(plan, _timestamp, self.now(), groups)
        if coverage:
            coverage["ending_matches_observed_baseline"] = [
                {"action": action, "observed_entity_id": eid, "matches": self._matches(action, catalog.get(eid))}
                for action in coverage.pop("ending_actions")
                for eid in sorted(plan_evidence.affected_entities(action["entity_id"], groups))]
            coverage["scope"] = "Ending coverage and current baseline comparison only; future execution is not verified"
        return {"calendar_binding": "revision_verified" if "calendar_event" in plan else "not_supplied",
                "timing_provenance": "Calendar references checked when supplied; omitted derivation and user timing are not independently verified",
                "calendar_policy": {"validation_scope": "preview_and_pre_apply_only",
                                    "post_apply_change_monitoring": False, "automatic_rescheduling": False,
                                    "saved_time_triggers": "fixed_absolute_instants"},
                "temporary_window_validation": "complete_ending_coverage" if coverage else "not_requested",
                "observation_semantics": {"verification_scope": "Home Assistant reported entity/control state",
                                          "physical_operation_evidence": "not_collected", "future_execution_evidence": "not_collected"},
                "temporary_window": coverage,
                "placement_requirements": copy.deepcopy(plan.get("placement_requirements", [])),
                "suitability": "not_verified"}

    def _guard(self, body, context):
        if body.get("expected_run_id") != context["run_id"]:
            raise ValueError("stale_run")
        if body.get("device_profile") != context["device_profile"]:
            raise ValueError("profile_mismatch")

    def preview(self, body):
        _keys(body, {"expected_run_id", "device_profile", "plan"}, {"expected_run_id", "device_profile", "plan"})
        with self.run_store.lock, self.lock:
            capabilities = self.capabilities()
            self._guard(body, capabilities)
            plan = self._validate_plan(body["plan"], capabilities)
            draft_id = uuid.uuid4().hex
            entry = {"draft_id": draft_id, "plan_hash": _hash(plan), "plan": plan, "run_id": capabilities["run_id"],
                     "planning_evidence": self._planning_evidence(plan, capabilities),
                     "device_profile": capabilities["device_profile"], "created_at": self.now().isoformat(),
                     "expires_at": (self.now() + dt.timedelta(minutes=15)).isoformat(), "status": "previewed"}
            ledger = self._load()
            ledger["drafts"][draft_id] = entry
            _atomic(self.ledger, ledger)
            return {"ok": True, **copy.deepcopy(entry), "physical_action_attempted": False,
                    "authorization_required": True, "evidence": [{"source": capabilities["source"], "observed_at": capabilities["observed_at"]}]}

    def _matches(self, action, row):
        if not row or row.get("state") in {"unknown", "unavailable"}:
            return False
        service, state, attrs, data = action["service"], row.get("state"), row.get("attributes", {}), action["data"]
        if service.endswith(".turn_on") and (state not in {"on", "playing", "paused", "idle"}):
            return False
        if service.endswith(".turn_off") and state not in {"off", "standby"}:
            return False
        if service == "cover.open_cover" and state != "open" or service == "cover.close_cover" and state != "closed":
            return False
        if service == "cover.stop_cover":
            return state not in {"opening", "closing"}
        for key, value in data.items():
            observed = state if key in {"option", "value", "hvac_mode"} else attrs.get("current_position" if key == "position" else key)
            if key == "rgb_color" and attrs.get("color_mode") == "xy" and "xy" in (attrs.get("supported_color_modes") or []):
                if not _xy_color_matches(value, attrs):
                    return False
            elif key == "color_temp_kelvin" and attrs.get("color_mode") == "color_temp" and "color_temp" in (attrs.get("supported_color_modes") or []):
                # ZHA sends integer mireds; HA reports floor(1e6 / mireds).
                # Accept that exact documented roundtrip, not an arbitrary band.
                if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed):
                    return False
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < 1_000_000:
                    return False
                quantized = math.floor(1_000_000 / math.floor(1_000_000 / value))
                if observed not in {value, quantized}:
                    return False
            elif isinstance(value, (int, float)):
                try:
                    if abs(float(observed) - value) > (3 if key == "brightness" else 0.1):
                        return False
                except (TypeError, ValueError):
                    return False
            elif observed != value:
                return False
        return True

    def _execute(self, action, profile, *, simulated=False):
        native = self._native_action(action)
        domain, service = native["action"].split(".", 1)
        payload = {**native.get("data", {}), **native.get("target", {})}
        evidence = {"action": action, "home_assistant_call": native, "device_profile": profile,
                    "physical_action_attempted": profile == "physical" and not simulated,
                    "is_virtual": profile == "virtual" or simulated, "attempted_at": self.now().isoformat(), "verified": False,
                    "verification_scope": "Home Assistant entity-state readback", "physical_effect_verified": False}
        try:
            response = self.ha("POST", f"/api/services/{domain}/{service}", payload, timeout=15)
            evidence["service_response_received"] = True
            evidence["service_response_kind"] = type(response).__name__
            if isinstance(response, dict) and (response.get("ok") is False or response.get("error")):
                raise RuntimeError("Home Assistant service returned an error")
            timeout = self.media_verify_timeout if action["service"] in {
                "media_player.select_source", "media_player.turn_on", "media_player.turn_off"
            } else self.verify_timeout
            evidence["verification_timeout_seconds"] = timeout
            deadline = time.monotonic() + timeout
            while True:
                rows = self._rows()
                row = rows.get(action["entity_id"], {})
                evidence["observed_state"] = {"entity_id": action["entity_id"], "state": row.get("state"),
                                               "attributes": {k: v for k, v in row.get("attributes", {}).items() if k in ATTRIBUTES}}
                members = row.get("attributes", {}).get("entity_id", []) if action["entity_id"].startswith("light.") else []
                verified = self._matches(action, row) and all(self._matches(action, rows.get(eid, {})) for eid in members)
                if verified or time.monotonic() >= deadline:
                    evidence["verified"] = verified
                    if not verified:
                        evidence["error"] = "Requested state was not observed; effect is unverified"
                    return evidence
                time.sleep(0.2)
        except Exception as exc:
            evidence["error"] = str(exc)[:300]
            return evidence

    def apply(self, body):
        _keys(body, {"draft_id", "plan_hash", "expected_run_id", "device_profile", "authorization", "idempotency_key"},
              {"draft_id", "plan_hash", "expected_run_id", "device_profile", "authorization", "idempotency_key"})
        _keys(body["authorization"], {"confirmed", "user_request"}, {"confirmed", "user_request"})
        if body["authorization"]["confirmed"] is not True:
            raise ValueError("Explicit user authorization is required")
        _text(body["authorization"]["user_request"], "authorization user request", 2000)
        key = _text(body["idempotency_key"], "idempotency key", 200)
        fingerprint = _hash(body)
        with self.run_store.lock, self.lock:
            context = self._context()
            self._guard(body, context)
            ledger = self._load()
            prior = ledger["requests"].get(key)
            if prior:
                if prior["fingerprint"] != fingerprint:
                    raise ValueError("idempotency_conflict")
                return {**copy.deepcopy(prior["receipt"]), "replayed": True}
            draft = ledger["drafts"].get(body["draft_id"])
            if not draft or draft["run_id"] != context["run_id"] or draft["device_profile"] != context["device_profile"] or draft["plan_hash"] != body["plan_hash"] or _hash(draft["plan"]) != body["plan_hash"]:
                raise ValueError("Draft/hash/run/profile mismatch")
            if draft["status"] != "previewed":
                raise ValueError("Draft already attempted; inspect its receipt instead of repeating effects")
            if _timestamp(draft["expires_at"]) <= self.now():
                raise ValueError("Draft expired")
            # Recheck availability, values and triggers immediately before any effect.
            capabilities = self.capabilities()
            self._validate_plan(draft["plan"], capabilities)
            receipt = {"ok": False, "draft_id": draft["draft_id"], "plan_hash": draft["plan_hash"], **context,
                       "status": "applying", "authorization": {**body["authorization"], "recorded_at": self.now().isoformat(), "source": "caller_attested_user_request"},
                       "actions": [], "automations": [], "errors": [], "physical_action_attempted": False, "replayed": False}
            receipt["planning_evidence"] = self._planning_evidence(draft["plan"], capabilities)
            draft["status"] = "applying"
            draft["receipt"] = receipt
            ledger["requests"][key] = {"fingerprint": fingerprint, "receipt": receipt}
            # Persist before effects: a crash is an unknown result, never an automatic replay.
            _atomic(self.ledger, ledger)
            try:
                for action in draft["plan"]["actions"]:
                    selected = next(row for row in capabilities["entities"] if row["entity_id"] == action["entity_id"])
                    result = self._execute(action, context["device_profile"], simulated=selected["is_virtual"])
                    receipt["actions"].append(result)
                    receipt["physical_action_attempted"] |= result["physical_action_attempted"]
                    _atomic(self.ledger, ledger)
                    if not result["verified"]:
                        raise RuntimeError("Action failed verification; remaining actions were not attempted")
                if draft["plan"]["automations"]:
                    receipt["automations"] = self._save_automations(draft)
                    if not all(row["loaded"] and row["enabled"] for row in receipt["automations"]):
                        raise RuntimeError("Automation file saved but active HA load was not verified")
                receipt.update(ok=True, status="applied")
            except Exception as exc:
                receipt["errors"].append(str(exc)[:300])
                receipt["status"] = "partial_or_unverified"
            receipt["finished_at"] = self.now().isoformat()
            draft["status"] = receipt["status"]
            _atomic(self.ledger, ledger)
            return copy.deepcopy(receipt)

    def _execution_receipt(self, binding, apply_body, context, *, error=None):
        """Recover evidence only. A durable reservation is never permission to retry."""
        unknown = {"ok": False, "draft_id": binding["draft_id"], "plan_hash": binding["plan_hash"],
                   **context, "status": "unknown", "actions": [], "automations": [], "errors": [],
                   "physical_action_attempted": None, "replayed": True,
                   "effect_status": "unknown; inspect the bound draft with plan_status; actions were not retried"}
        if error:
            unknown["errors"].append(str(error)[:300] if isinstance(error, ValueError) else type(error).__name__)
        try:
            ledger = self._load()
            draft = ledger["drafts"].get(binding["draft_id"])
            if not draft or any(draft.get(name) != binding.get(name) for name in
                                ("draft_id", "plan_hash", "run_id", "device_profile")) \
                    or _hash(draft["plan"]) != binding["plan_hash"]:
                raise ValueError("execution_binding_mismatch")
            prior = ledger["requests"].get(binding["apply_key"])
            if not prior:
                # Includes the crash window after reservation but before apply's
                # durable attempt. Fail closed, even though no effect may have run.
                return unknown
            if prior.get("fingerprint") != _hash(apply_body):
                raise ValueError("execution_apply_binding_mismatch")
            receipt = copy.deepcopy(prior["receipt"])
            if any(receipt.get(name) != binding.get(name) for name in
                   ("draft_id", "plan_hash", "run_id", "device_profile")):
                raise ValueError("execution_receipt_binding_mismatch")
            if receipt.get("status") not in {"applied", "partial_or_unverified"}:
                # An applying journal can predate a device write. Its last false
                # physical_action_attempted flag cannot prove nothing happened.
                unknown["recorded_receipt"] = receipt
                return unknown
            return {**receipt, "replayed": True}
        except Exception as exc:
            unknown["errors"].append(str(exc)[:300] if isinstance(exc, ValueError) else type(exc).__name__)
            return unknown

    def execute(self, body):
        """Validate and apply caller-authored immediate actions authorized up front.

        A durable operation-to-draft mapping precedes apply's own durable attempt
        journal. Replays recover those receipts without ever resuming effects.
        """
        _keys(body, {"expected_run_id", "device_profile", "plan", "authorization", "idempotency_key"},
              {"expected_run_id", "device_profile", "plan", "authorization", "idempotency_key"})
        _keys(body["authorization"], {"confirmed", "user_request"}, {"confirmed", "user_request"})
        if body["authorization"]["confirmed"] is not True:
            raise ValueError("Explicit user authorization is required")
        _text(body["authorization"]["user_request"], "authorization user request", 2000)
        key = _text(body["idempotency_key"], "idempotency key", 200)
        _keys(body["plan"], {"name", "actions", "automations", "placement_requirements"}, {"name", "actions"})
        if not isinstance(body["plan"].get("automations", []), list) or body["plan"].get("automations"):
            raise ValueError("Execute supports immediate actions only; preview future rules separately")
        fingerprint = _hash(body)
        with self.run_store.lock, self.lock:
            context = self._context()
            self._guard(body, context)
            ledger = self._load()
            binding = ledger.get("executions", {}).get(key)
            replayed = binding is not None
            if replayed:
                if binding.get("fingerprint") != fingerprint:
                    raise ValueError("idempotency_conflict")
            else:
                draft = self.preview({"expected_run_id": body["expected_run_id"],
                                      "device_profile": body["device_profile"], "plan": body["plan"]})
                binding = {"fingerprint": fingerprint, "draft_id": draft["draft_id"],
                           "plan_hash": draft["plan_hash"], **context,
                           "apply_key": "execute:" + draft["draft_id"], "created_at": self.now().isoformat()}
                ledger = self._load()
                ledger.setdefault("executions", {})[key] = binding
                # No effect is possible before both the draft and this exact
                # operation mapping are durably stored in the existing ledger.
                try:
                    _atomic(self.ledger, ledger)
                except Exception as exc:
                    return {"ok": False, "draft_id": binding["draft_id"], "plan_hash": binding["plan_hash"],
                            **context, "status": "not_started", "actions": [], "automations": [],
                            "errors": [type(exc).__name__], "physical_action_attempted": False, "replayed": False}
            apply_body = {"draft_id": binding["draft_id"], "plan_hash": binding["plan_hash"],
                          "expected_run_id": body["expected_run_id"], "device_profile": body["device_profile"],
                          "authorization": copy.deepcopy(body["authorization"]), "idempotency_key": binding["apply_key"]}
            if replayed:
                return self._execution_receipt(binding, apply_body, context)
            try:
                return self.apply(apply_body)
            except Exception as exc:
                # Preserve a completed/partial journal if available, otherwise
                # expose the bound draft as unknown. Never call apply a second time.
                receipt = self._execution_receipt(binding, apply_body, context, error=exc)
                receipt["replayed"] = False
                return receipt

    def status(self, draft_id):
        _text(draft_id, "draft_id", 100)
        with self.lock:
            draft = self._load()["drafts"].get(draft_id)
            if not draft:
                raise ValueError("Unknown draft")
            result = {"ok": True, **copy.deepcopy(draft), "receipt_is_historical": True}
            if draft.get("receipt", {}).get("automations"):
                try:
                    rows = self._rows()
                    ids = {row["automation_id"] for row in draft["receipt"]["automations"]}
                    result["current_automations"] = [{"entity_id": eid, "automation_id": row.get("attributes", {}).get("id"),
                                                       "state": row.get("state"), "last_triggered": row.get("attributes", {}).get("last_triggered"),
                                                       "source": "Home Assistant /api/states"}
                                                      for eid, row in rows.items() if row.get("attributes", {}).get("id") in ids]
                    result["observed_at"] = self.now().isoformat()
                except Exception as exc:
                    result["current_observation_error"] = type(exc).__name__
            return result
