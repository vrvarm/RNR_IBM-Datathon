"""Decision engine: model score + three explainable layer checks -> Approve / Flag for 2FA / Decline."""
import math

from .config import (AMOUNT_Z_ALERT, AMOUNT_Z_WARN, LOCATION_ALERT_KM, LOCATION_WARN_KM,
                     MIN_ALERT_KM, MIN_WARN_KM, SPEED_ALERT_KMH, SPEED_WARN_KMH)

APPROVE, FLAG, DECLINE = "APPROVE", "FLAG_2FA", "DECLINE"


def layer_location(f):
    d = f["dev_store_km"]
    if d >= LOCATION_ALERT_KM:
        return "alert", f"Device is {d:.0f} km from the store"
    if d >= LOCATION_WARN_KM:
        return "warn", f"Device is {d:.0f} km from the store"
    return "ok", f"Device is at the store ({d:.2f} km)"


def layer_velocity(f):
    if math.isnan(f["speed_kmh"]):
        return "ok", "No prior transaction on file for this card"
    km, speed, mins = f["km_from_last"], f["speed_kmh"], f["hours_since_last"] * 60
    msg = f"{km:.0f} km from the last transaction {mins:.0f} min ago ({speed:.0f} km/h)"
    if speed >= SPEED_ALERT_KMH and km >= MIN_ALERT_KM:
        return "alert", "Impossible travel: " + msg
    if speed >= SPEED_WARN_KMH and km >= MIN_WARN_KM:
        return "warn", "Very fast travel: " + msg
    return "ok", msg


def layer_merchant(f, base_rate):
    z = f["amount_z_cat"]
    risky = f["cat_risk"] >= 2 * base_rate
    new = f["new_cat"] == 1.0
    if z >= AMOUNT_Z_ALERT and (new or risky):
        return "alert", f"Unusually large amount for this category (z={z:.1f}) in a new or high-risk category"
    if z >= AMOUNT_Z_WARN or (new and risky):
        why = "unusually large amount" if z >= AMOUNT_Z_WARN else "first purchase in a high-risk category"
        return "warn", f"{why.capitalize()} (z={z:.1f})"
    return "ok", "Consistent with this card's merchant profile"


def decide(p, f, thresholds, base_rate):
    layers = {
        "location": layer_location(f),
        "velocity": layer_velocity(f),
        "merchant": layer_merchant(f, base_rate),
    }
    alerts = [name for name, (status, _) in layers.items() if status == "alert"]
    reasons = []

    if p >= thresholds["decline"]:
        action = DECLINE
        reasons.append(f"Model fraud score {p:.2f} is above the decline threshold {thresholds['decline']:.2f}")
    elif p >= thresholds["flag"]:
        action = FLAG
        reasons.append(f"Model fraud score {p:.2f} is above the 2FA threshold {thresholds['flag']:.2f}")
    else:
        action = APPROVE

    if len(alerts) >= 2:
        action = DECLINE
        reasons.append("Multiple layers raised alerts: " + ", ".join(alerts))
    elif alerts and action == APPROVE:
        action = FLAG
        reasons.append(f"{alerts[0].capitalize()} layer raised an alert")

    reasons += [msg for status, msg in layers.values() if status != "ok"]
    return {
        "decision": action,
        "layers": {k: {"status": s, "detail": m} for k, (s, m) in layers.items()},
        "reasons": reasons,
    }
