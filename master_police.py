"""
🔱 MASTER POLICE — Health & Diagnostic Watchdog (Report-Only)
================================================================
Yeh sirf DEKHTA hai aur REPORT karta hai — kisi bhi cheez ko
block/pause/stop NAHI karta (woh kaam master_controller_backend.py
ka hai). Do cheezein check karta hai:

  A) INTERNAL — backend ke andar jo files/trackers/indicators chal
     rahe hain, woh zinda hain ya stuck/fake ho gaye hain
  B) EXTERNAL — frontend pages (chart/watchlist/login) load ho rahe
     hain, aur unke peeche ka data fresh hai ya stale

KAISE KAAM KARTA HAI (Internal ke liye):
------------------------------------------
Har internal file/component jise track karna hai, apna "main zinda
hoon" signal bhejta hai ek chhoti si call se — bas itna:

    from master_police import heartbeat
    heartbeat("indicator_engine")          # bas itna, har cycle me ek baar

Police background me check karta rehta hai — agar kisi component ka
heartbeat X minute se nahi aaya, ya value change hi nahi ho rahi
(stuck/fake data ka lakshan), to Telegram par turant alert.

KAISE KAAM KARTA HAI (External ke liye):
------------------------------------------
Neeche PAGE_ENDPOINTS list me apne pages/APIs ke URL daal do — Police
unhe periodically HTTP GET karega, response check karega.

INTEGRATION:
    from master_police import POLICE
    POLICE.start()   # app start hote hi ek baar
"""

import time
import json
import threading
import requests
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from master_controller_backend import send_telegram, audit_log   # reuse same telegram+log


# ============================================================
# CONFIG — apne URLs/thresholds yahan daalo
# ============================================================

# Har internal component ka heartbeat itni der me na aaye to STUCK maano (seconds)
HEARTBEAT_STALE_THRESHOLD_SEC = 180   # 3 minute — apne component ke normal cycle se thoda zyada rakho

# External pages/APIs jo check karne hain
# 'freshness_field' optional hai — agar response JSON me koi timestamp field hai
# jisse data ki freshness pata chal sake, uska naam do (jaise "last_updated")
PAGE_ENDPOINTS = [
    {"name": "Chart Page", "url": "http://localhost:5000/chart", "freshness_field": None},
    {"name": "Monitor Control Page", "url": "http://localhost:5000/monitor", "freshness_field": None},
    {"name": "Login Page", "url": "http://localhost:5000/login", "freshness_field": None},
    {"name": "Chart Data API", "url": "http://localhost:5000/api/chart-data", "freshness_field": "last_updated"},
    {"name": "Watchlist API", "url": "http://localhost:5000/api/watchlist", "freshness_field": "last_updated"},
    # ... apne actual URLs/ports se replace/add karo
]

DATA_STALE_THRESHOLD_SEC = 120   # agar freshness_field itni purani ho to stale maano
CHECK_INTERVAL_SEC = 60          # har 1 minute check


# ============================================================
# INTERNAL HEARTBEAT STORE
# ============================================================

@dataclass
class ComponentHeartbeat:
    name: str
    last_seen: float = field(default_factory=time.time)
    last_value: Any = None
    stuck_alert_sent: bool = False
    dead_alert_sent: bool = False


_heartbeats: Dict[str, ComponentHeartbeat] = {}
_hb_lock = threading.Lock()


def heartbeat(component_name: str, value: Any = None):
    """Har internal file/component yahan se apna 'main zinda hoon' signal bheje.
    'value' optional hai — agar do (jaise current price/indicator value), Police
    yeh bhi check karega ki value waqai badal rahi hai ya hamesha same hai (fake/stuck)."""
    with _hb_lock:
        if component_name not in _heartbeats:
            _heartbeats[component_name] = ComponentHeartbeat(name=component_name)
        hb = _heartbeats[component_name]
        hb.last_seen = time.time()
        hb.last_value = value
        hb.stuck_alert_sent = False
        hb.dead_alert_sent = False


# ============================================================
# MASTER POLICE
# ============================================================

class MasterPolice:
    def __init__(self):
        self._running = False
        self._last_value_snapshot: Dict[str, Any] = {}
        self._unchanged_since: Dict[str, float] = {}

    def start(self):
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        send_telegram("🔱 MASTER REPORT (Police)\n\n👮 Master Police started — internal aur external health monitor ho raha hai.")

    def _loop(self):
        while self._running:
            self._check_internal()
            self._check_external()
            time.sleep(CHECK_INTERVAL_SEC)

    # ---------- internal check ----------

    def _check_internal(self):
        now = time.time()
        with _hb_lock:
            for name, hb in list(_heartbeats.items()):
                age = now - hb.last_seen

                # 1. Dead check — heartbeat hi nahi aa raha
                if age > HEARTBEAT_STALE_THRESHOLD_SEC and not hb.dead_alert_sent:
                    audit_log("POLICE_COMPONENT_DEAD", {"component": name, "seconds_since_last_heartbeat": round(age)})
                    send_telegram(f"🔱 MASTER REPORT (Police)\n\n"
                                   f"🚨 '{name}' se {round(age)} second se koi heartbeat nahi aaya — "
                                   f"yeh component STUCK ya CRASH ho sakta hai. Check karo.")
                    hb.dead_alert_sent = True

                # 2. Fake/stuck value check — heartbeat aa raha hai par value badal hi nahi rahi
                if hb.last_value is not None:
                    prev = self._last_value_snapshot.get(name)
                    if prev is not None and prev == hb.last_value:
                        unchanged_since = self._unchanged_since.setdefault(name, now)
                        stuck_duration = now - unchanged_since
                        if stuck_duration > HEARTBEAT_STALE_THRESHOLD_SEC and not hb.stuck_alert_sent:
                            audit_log("POLICE_VALUE_STUCK", {"component": name, "value": str(hb.last_value),
                                                               "stuck_seconds": round(stuck_duration)})
                            send_telegram(f"🔱 MASTER REPORT (Police)\n\n"
                                          f"⚠️ '{name}' ka value {round(stuck_duration)} second se BADLA NAHI hai "
                                          f"(current: {hb.last_value}) — yeh FAKE/STALE data ho sakta hai. Check karo.")
                            hb.stuck_alert_sent = True
                    else:
                        self._unchanged_since[name] = now
                    self._last_value_snapshot[name] = hb.last_value

    # ---------- external check ----------

    def _check_external(self):
        for page in PAGE_ENDPOINTS:
            self._check_one_page(page)

    def _check_one_page(self, page: dict):
        name, url = page["name"], page["url"]
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                audit_log("POLICE_PAGE_ERROR", {"page": name, "status_code": resp.status_code})
                send_telegram(f"🔱 MASTER REPORT (Police)\n\n"
                               f"🚨 '{name}' response status {resp.status_code} — yeh page down/error de raha hai.")
                return

            # Freshness check (agar field specify ki hai)
            freshness_field = page.get("freshness_field")
            if freshness_field:
                try:
                    data = resp.json()
                    ts_str = data.get(freshness_field)
                    if ts_str:
                        last_updated = datetime.fromisoformat(ts_str)
                        age = (datetime.now() - last_updated).total_seconds()
                        if age > DATA_STALE_THRESHOLD_SEC:
                            audit_log("POLICE_DATA_STALE", {"page": name, "age_seconds": round(age)})
                            send_telegram(f"🔱 MASTER REPORT (Police)\n\n"
                                          f"⚠️ '{name}' ka data {round(age)} second purana hai — "
                                          f"stale/not-updating lag raha hai.")
                except Exception:
                    pass   # freshness field parse na ho paye to silently skip, page to up hai

        except requests.exceptions.RequestException as e:
            audit_log("POLICE_PAGE_UNREACHABLE", {"page": name, "error": str(e)})
            send_telegram(f"🔱 MASTER REPORT (Police)\n\n"
                           f"🚨 '{name}' se connect hi nahi ho paya — page/server down ho sakta hai.\nError: {e}")

    def stop(self):
        self._running = False


POLICE = MasterPolice()
