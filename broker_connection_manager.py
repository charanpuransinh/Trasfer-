"""
🔱 BROKER CONNECTION MANAGER
================================================================
Subah 8:00 AM auto-connect sab brokers (Fyers/Dhan/Kotak), shaam ko
auto-disconnect — taaki raat bhar baar-baar reconnect attempt na ho
(jo broker ko block karne ka risk banata hai).

4:00 PM ke baad agar kaam karna ho (testing/manual), to sirf MANUAL
connect() call karo — yeh auto-connect kabhi khud nahi karega us
window ke bahar.

INTEGRATION:
------------
    from broker_connection_manager import BROKER_MGR

    # Apna connect/disconnect function har broker ke liye register karo:
    BROKER_MGR.register_broker(
        name="fyers",
        connect_fn=my_fyers_connect_function,
        disconnect_fn=my_fyers_disconnect_function,
    )
    BROKER_MGR.register_broker("dhan", my_dhan_connect, my_dhan_disconnect)
    BROKER_MGR.register_broker("kotak", my_kotak_connect, my_kotak_disconnect)

    BROKER_MGR.start()   # background scheduler shuru — app start hote hi ek baar call karo

    # Manual override kabhi bhi (4 baje ke baad testing ke liye):
    BROKER_MGR.manual_connect("fyers")
    BROKER_MGR.manual_disconnect("fyers")
"""

import time
import threading
from datetime import datetime, time as dtime
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from master_controller_backend import send_telegram, audit_log   # reuse same telegram+log


AUTO_CONNECT_TIME = dtime(8, 0)
AUTO_DISCONNECT_TIME = dtime(17, 30)     # 5:30 PM — apne hisaab se 17:00-19:00 me adjust kar lena

MAX_RETRY_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = [30, 60, 120, 240, 480]   # badhta hua gap har retry par


@dataclass
class BrokerEntry:
    name: str
    connect_fn: Callable[[], bool]       # True return kare agar connect successful
    disconnect_fn: Callable[[], None]
    connected: bool = False
    last_action_date: str = ""           # taaki ek din me sirf ek baar auto-connect/disconnect ho


class BrokerConnectionManager:
    def __init__(self):
        self.brokers: Dict[str, BrokerEntry] = {}
        self._lock = threading.Lock()
        self._running = False

    def register_broker(self, name: str, connect_fn: Callable, disconnect_fn: Callable):
        self.brokers[name] = BrokerEntry(name=name, connect_fn=connect_fn, disconnect_fn=disconnect_fn)

    # ---------- connect with backoff ----------

    def _connect_with_retry(self, entry: BrokerEntry) -> bool:
        for attempt, delay in enumerate(RETRY_BACKOFF_SECONDS, start=1):
            try:
                ok = entry.connect_fn()
                if ok:
                    entry.connected = True
                    audit_log("BROKER_CONNECTED", {"broker": entry.name, "attempt": attempt})
                    return True
            except Exception as e:
                audit_log("BROKER_CONNECT_ERROR", {"broker": entry.name, "attempt": attempt, "error": str(e)})

            if attempt < MAX_RETRY_ATTEMPTS:
                time.sleep(delay)

        # Sab retries fail — ab ruk jao, telegram bata do, raat bhar mat koshish karte raho
        audit_log("BROKER_CONNECT_FAILED_FINAL", {"broker": entry.name, "attempts": MAX_RETRY_ATTEMPTS})
        send_telegram(f"🔱 MASTER REPORT (Backend)\n\n"
                       f"⚠️ {entry.name.upper()} connect FAIL ho gaya {MAX_RETRY_ATTEMPTS} attempts ke baad.\n"
                       f"Auto-retry ab ROOK diya (broker block hone se bachne ke liye).\n"
                       f"Manual check karo.")
        return False

    def manual_connect(self, name: str) -> bool:
        entry = self.brokers.get(name)
        if not entry:
            return False
        return self._connect_with_retry(entry)

    def manual_disconnect(self, name: str):
        entry = self.brokers.get(name)
        if not entry:
            return
        try:
            entry.disconnect_fn()
            entry.connected = False
            audit_log("BROKER_DISCONNECTED_MANUAL", {"broker": entry.name})
        except Exception as e:
            audit_log("BROKER_DISCONNECT_ERROR", {"broker": entry.name, "error": str(e)})

    # ---------- scheduler ----------

    def start(self):
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            now_t = now.time()

            with self._lock:
                for entry in self.brokers.values():
                    # Auto-connect window — sirf ek baar per din, sirf iss time ke aas-paas
                    if (now_t >= AUTO_CONNECT_TIME and now_t < AUTO_DISCONNECT_TIME
                            and entry.last_action_date != today and not entry.connected):
                        ok = self._connect_with_retry(entry)
                        if ok:
                            entry.last_action_date = today
                            send_telegram(f"🔱 MASTER REPORT (Backend)\n\n✅ {entry.name.upper()} auto-connected ({now.strftime('%H:%M')})")

                    # Auto-disconnect — ek baar, evening
                    if now_t >= AUTO_DISCONNECT_TIME and entry.connected:
                        self.manual_disconnect(entry.name)
                        send_telegram(f"🔱 MASTER REPORT (Backend)\n\n🌙 {entry.name.upper()} auto-disconnected ({now.strftime('%H:%M')}) — raat bhar reconnect attempt nahi hoga.")

            time.sleep(30)   # har 30 sec check — halka hai

    def stop(self):
        self._running = False


BROKER_MGR = BrokerConnectionManager()
