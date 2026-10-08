"""Pure validation helpers; no household registry filesystem access."""
import datetime as dt

def placement(entity_id, registries):
    """Registry assignment is evidence of configured area, not physical suitability."""
    result = {"available": False, "area_id": None, "area_name": None,
              "source": "Home Assistant entity/device/area registries", "missing_evidence": []}
    if not isinstance(registries, dict) or any(not isinstance(registries.get(key, []), list) for key in ("entities", "devices", "areas")):
        result["missing_evidence"] = ["Placement registry evidence unavailable"]
        return result
    rows = registries.get("entities", [])
    matches = [row for row in rows if isinstance(row, dict) and row.get("entity_id") == entity_id]
    if len(matches) != 1:
        result["missing_evidence"] = ["No unique entity registry record establishes a configured area"]
        return result
    row = matches[0]
    area = row.get("area_id")
    inherited = False
    if not area and row.get("device_id"):
        devices = [item for item in registries.get("devices", []) if isinstance(item, dict) and item.get("id") == row["device_id"]]
        if len(devices) == 1:
            area = devices[0].get("area_id")
            inherited = bool(area)
    areas = [item for item in registries.get("areas", []) if isinstance(item, dict) and item.get("id") == area]
    if not isinstance(area, str) or not area or len(areas) != 1 or not isinstance(areas[0].get("name"), str):
        result["missing_evidence"] = ["Entity has no resolvable area assignment in the observed registries"]
        return result
    result.update(available=True, area_id=area, area_name=areas[0]["name"],
                  assignment="device_inherited" if inherited else "entity_explicit")
    return result

def calendar_time(reference, event, timestamp, number):
    if not isinstance(reference, dict) or set(reference) != {"boundary", "offset_seconds"}:
        raise ValueError("Calendar reference requires boundary and offset_seconds")
    if reference["boundary"] not in {"start", "end"}:
        raise ValueError("Calendar boundary must be start or end")
    offset = number(reference["offset_seconds"], -366 * 86400, 366 * 86400)
    if not isinstance(event, dict) or not event.get(reference["boundary"]):
        raise ValueError("Referenced calendar boundary is unavailable; read evidence or ask for the missing time")
    return timestamp(event[reference["boundary"]]) + dt.timedelta(seconds=offset)

def effect_fields(action):
    service, data = action["service"], action.get("data", {})
    fields = set(data)
    if service.endswith((".turn_on", ".turn_off")):
        fields.add("power")
    if service in {"cover.open_cover", "cover.close_cover", "cover.stop_cover", "cover.set_cover_position"}:
        fields.add("position")
    if service in {"input_select.select_option", "input_number.set_value", "climate.set_hvac_mode"}:
        fields.add("state")
    # Setting color through another representation still addresses the same output.
    if fields.intersection({"rgb_color", "color_temp_kelvin"}):
        fields.difference_update({"rgb_color", "color_temp_kelvin"})
        fields.add("color")
    return fields

def affected_entities(entity_id, groups, path=()):
    if entity_id in path:
        raise ValueError("Cyclic observed group membership")
    members = groups.get(entity_id, [])
    if not members:
        return {entity_id}
    result = set()
    for member in members:
        result.update(affected_entities(member, groups, path + (entity_id,)))
    return result

def temporary_coverage(plan, timestamp, now, groups=None):
    """Check declared finite lifetime without choosing any action or interpreting prose."""
    groups = groups or {}
    window = plan.get("temporary_window")
    if window is None:
        return None
    if not isinstance(window, dict) or set(window) != {"starts_at", "ends_at"}:
        raise ValueError("Temporary window requires starts_at and ends_at")
    start, end = timestamp(window["starts_at"]), timestamp(window["ends_at"])
    if not start < end or not now + dt.timedelta(seconds=5) < end <= now + dt.timedelta(days=366):
        raise ValueError("Temporary window must have a later future end within 366 days")
    if plan["actions"] and not now - dt.timedelta(minutes=15) <= start <= now + dt.timedelta(seconds=5):
        raise ValueError("Immediate actions require the temporary window to start now; otherwise schedule them explicitly")
    before = list(plan["actions"])
    ending = []
    for auto in plan["automations"]:
        for trigger in auto["triggers"]:
            if trigger["kind"] != "time":
                raise ValueError("Temporary windows require explicit time triggers; state rules can fire again after the end")
            at = timestamp(trigger["at"])
            if not start <= at <= end:
                raise ValueError("Temporary automation trigger lies outside its declared window")
            (ending if at == end else before).extend(auto["actions"])
    if not before:
        raise ValueError("Temporary window contains no action before its end")
    changed, covered, terminal, final_values = {}, {}, set(), {}
    for action in before:
        for entity_id in affected_entities(action["entity_id"], groups):
            changed.setdefault(entity_id, set()).update(effect_fields(action))
    for action in ending:
        values = dict(action["data"])
        service = action["service"]
        if service.endswith((".turn_on", ".turn_off")):
            values["power"] = service.rsplit(".", 1)[1]
        if service in {"cover.open_cover", "cover.close_cover", "cover.stop_cover"}:
            values["position"] = service
        for entity_id in affected_entities(action["entity_id"], groups):
            for field, value in values.items():
                field = "color" if field in {"rgb_color", "color_temp_kelvin"} else field
                key = (entity_id, field)
                if key in final_values and final_values[key] != value:
                    raise ValueError("Temporary ending actions conflict at the same instant: " + entity_id)
                final_values[key] = value
            covered.setdefault(entity_id, set()).update(effect_fields(action))
            if action["service"].endswith(".turn_off") or action["service"] == "climate.set_hvac_mode" and action["data"].get("hvac_mode") == "off":
                terminal.add(entity_id)
    for entity_id, fields in changed.items():
        if entity_id not in terminal and not fields <= covered.get(entity_id, set()):
            raise ValueError("Temporary window lacks explicit ending actions for all changed settings: " + entity_id)
    return {"starts_at": start.isoformat(), "ends_at": end.isoformat(),
            "ending_actions": ending, "changed_entities": sorted(changed), "coverage_complete": True}
