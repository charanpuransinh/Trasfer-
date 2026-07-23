"""
🔱 MASTER CONTROLLER — Backend (KillSwitch / Risk Gatekeeper)
================================================================
Trishul Pro ke liye independent risk-control layer.

DESIGN PRINCIPLE:
Koi bhi strategy directly broker ko order NAHI bhejegi. Har order pehle
is Master ke `validate_order()` se guzregi. Master allow kare tabhi
order aage jayega — nahi to block ho jayega, aur reason ke saath log +
Telegram alert turant chala jayega.

Yeh module kisi strategy code ko import/modify nahi karta — strategies
ko iske through guzarna PADEGA (gatekeeper pattern). Isse:
  - Kisi ek strategy ke bug se doosri strategies affect nahi hongi
  - Rules ek hi jagah se control hote hain, har jagah alag-alag nahi
  - Audit log append-only hai — koi bhi purana record edit/delete
    nahi kar sakta (sirf naya add ho sakta hai)

INTEGRATION (Trishul Pro me kaise use karo):
--------------------------------------------
    from master_controller_backend import MASTER

    # Naya order lagane se PEHLE:
    allowed, reason = MASTER.validate_order(
        strategy_name="तड़ित_Index_EMA_Trend",
        side="BUY",
        symbol="SENSEX25JUL77000CE",
    )
    if not allowed:
        print(f"Order BLOCKED: {reason}")
        return
    # ... yahan asli broker order call karo (fyers/dhan/kotak) ...

    # Trade CLOSE hone ke baad (SL hit ho ya target hit ho ya manual close):
    MASTER.record_trade_close(
        strategy_name="तड़ित_Index_EMA_Trend",
        pnl=-45.2,
        hit_sl=True,   # True agar SL se close hua
    )

    # Force square-off ke liye apna close-all function register karo:
    MASTER.register_force_close_handler(my_close_all_positions_function)

    # Background thread start karo (ek hi baar, app start hote hi):
    MASTER.start()
"""

import os
import json
import time
import threading
from datetime import datetime, time as dtime
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import requests   # pip install requests


# ============================================================
# CONFIG — apne hisaab se yahan set karo
# ============================================================

# Telegram — Trishul Pro me pehle se jo bot chal raha hai, usi ka token/chat_id
TELEGRAM_BOT_TOKEN = os.environ.get("TRISHUL_TELEGRAM_BOT_TOKEN", "PASTE_YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TRISHUL_TELEGRAM_CHAT_ID", "PASTE_YOUR_CHAT_ID_HERE")

# Trading window
TRADE_START_TIME = dtime(9, 20)          # isse pehle koi entry nahi
NO_NEW_ENTRY_AFTER = dtime(15, 15)       # isse baad koi NAYI entry nahi
FORCE_SQUARE_OFF_TIME = dtime(15, 25)    # is time par sab kuch force-close

# Risk limits (TESTING ke daura in numbers ko conservative rakho,
# live jaane se pehle apni capital ke hisaab se tune karo)
MAX_SL_HITS_PER_STRATEGY_PER_DAY = 3     # itni baar SL hit -> strategy pause
MAX_LOSS_PER_STRATEGY_PER_DAY = -2000.0  # is se zyada loss -> strategy pause
MAX_OVERALL_DAILY_LOSS = -8000.0         # is se zyada -> SAB strategies band (full killswitch)

AUDIT_LOG_PATH = os.environ.get("MASTER_AUDIT_LOG", "master_audit_log.jsonl")


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message: str):
    if "PASTE_YOUR" in TELEGRAM_BOT_TOKEN:
        print(f"[Master][Telegram not configured] {message}")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        print(f"[Master] Telegram send failed: {e}")


# ============================================================
# AUDIT LOG — append-only, kabhi overwrite/delete nahi hota
# ============================================================

_audit_lock = threading.Lock()


def audit_log(event_type: str, details: dict):
    entry = {
        "ts": datetime.now().isoformat(),
        "event": event_type,
        **details,
    }
    with _audit_lock:
        # 'a' mode = append only. Yeh function kabhi file ko truncate/rewrite nahi karta.
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ============================================================
# STATE — per-strategy aur overall din ka hisaab
# ============================================================

@dataclass
class StrategyState:
    name: str
    sl_hits_today: int = 0
    pnl_today: float = 0.0
    paused: bool = False
    pause_reason: str = ""


@dataclass
class MasterState:
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    strategies: Dict[str, StrategyState] = field(default_factory=dict)
    overall_pnl_today: float = 0.0
    global_kill_switch: bool = False
    global_kill_reason: str = ""
    force_close_done_today: bool = False

    def get_strategy(self, name: str) -> StrategyState:
        if name not in self.strategies:
            self.strategies[name] = StrategyState(name=name)
        return self.strategies[name]


# ============================================================
# MASTER CONTROLLER
# ============================================================

class MasterController:
    def __init__(self):
        self.state = MasterState()
        self._force_close_handlers: List[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._running = False

    # ---------- registration ----------

    def register_force_close_handler(self, fn: Callable[[], None]):
        """Trishul Pro apna 'close all open positions' function yahan register kare."""
        self._force_close_handlers.append(fn)

    # ---------- core gate ----------

    def validate_order(self, strategy_name: str, side: str, symbol: str,
                        is_exit: bool = False) -> (bool, str):
        """
        Har order (entry ya exit) is function se guzarna chahiye.
        Return: (allowed: bool, reason: str)
        Exit orders (is_exit=True) ko square-off tak allow rakha jata hai,
        entry orders trade-window aur limits dono se check hote hain.
        """
        with self._lock:
            self._check_day_rollover()
            now_t = datetime.now().time()

            # 1. Global kill switch — sabse pehle check
            if self.state.global_kill_switch:
                reason = f"GLOBAL KILL SWITCH ACTIVE: {self.state.global_kill_reason}"
                audit_log("ORDER_BLOCKED", {"strategy": strategy_name, "symbol": symbol,
                                             "side": side, "reason": reason})
                return False, reason

            # 2. Exit orders — sirf square-off time ke baad bhi allowed rehte hain
            #    (loss cut karna hamesha allow honा chahiye, chahe time kuch bhi ho)
            if is_exit:
                audit_log("ORDER_ALLOWED_EXIT", {"strategy": strategy_name, "symbol": symbol})
                return True, "exit allowed"

            # 3. Strategy-level pause
            st = self.state.get_strategy(strategy_name)
            if st.paused:
                reason = f"Strategy '{strategy_name}' PAUSED: {st.pause_reason}"
                audit_log("ORDER_BLOCKED", {"strategy": strategy_name, "symbol": symbol,
                                             "side": side, "reason": reason})
                return False, reason

            # 4. Trade window check (sirf NEW entries ke liye)
            if now_t < TRADE_START_TIME:
                reason = f"Before trade start time ({TRADE_START_TIME}) — no entry allowed"
                audit_log("ORDER_BLOCKED", {"strategy": strategy_name, "symbol": symbol,
                                             "side": side, "reason": reason})
                return False, reason

            if now_t >= NO_NEW_ENTRY_AFTER:
                reason = f"After no-new-entry cutoff ({NO_NEW_ENTRY_AFTER}) — no entry allowed"
                audit_log("ORDER_BLOCKED", {"strategy": strategy_name, "symbol": symbol,
                                             "side": side, "reason": reason})
                return False, reason

            # sab check pass -> allow
            audit_log("ORDER_ALLOWED_ENTRY", {"strategy": strategy_name, "symbol": symbol,
                                                "side": side})
            return True, "allowed"

    # ---------- trade result feedback ----------

    def record_trade_close(self, strategy_name: str, pnl: float, hit_sl: bool = False):
        """Trade close hone par (SL/target/manual) yeh call karo — isi se limits track hoti hain."""
        with self._lock:
            self._check_day_rollover()
            st = self.state.get_strategy(strategy_name)
            st.pnl_today += pnl
            self.state.overall_pnl_today += pnl
            if hit_sl:
                st.sl_hits_today += 1

            audit_log("TRADE_CLOSED", {"strategy": strategy_name, "pnl": pnl,
                                        "hit_sl": hit_sl, "strategy_pnl_today": st.pnl_today,
                                        "overall_pnl_today": self.state.overall_pnl_today})

            # per-strategy SL-hit limit
            if not st.paused and st.sl_hits_today >= MAX_SL_HITS_PER_STRATEGY_PER_DAY:
                self._pause_strategy(st, f"{st.sl_hits_today} SL hits today "
                                          f"(limit {MAX_SL_HITS_PER_STRATEGY_PER_DAY})")

            # per-strategy loss limit
            if not st.paused and st.pnl_today <= MAX_LOSS_PER_STRATEGY_PER_DAY:
                self._pause_strategy(st, f"Loss {st.pnl_today:.2f} crossed limit "
                                          f"{MAX_LOSS_PER_STRATEGY_PER_DAY}")

            # overall daily loss -> full kill switch
            if not self.state.global_kill_switch and \
               self.state.overall_pnl_today <= MAX_OVERALL_DAILY_LOSS:
                self._trigger_global_kill(f"Overall daily loss {self.state.overall_pnl_today:.2f} "
                                           f"crossed limit {MAX_OVERALL_DAILY_LOSS}")

    def _pause_strategy(self, st: StrategyState, reason: str):
        st.paused = True
        st.pause_reason = reason
        audit_log("STRATEGY_PAUSED", {"strategy": st.name, "reason": reason})
        send_telegram(f"🔱 MASTER REPORT (Backend)\n\n"
                       f"⏸️ STRATEGY PAUSED: {st.name}\n"
                       f"Reason: {reason}\n"
                       f"Today's P&L for this strategy: {st.pnl_today:.2f}")

    def _trigger_global_kill(self, reason: str):
        self.state.global_kill_switch = True
        self.state.global_kill_reason = reason
        audit_log("GLOBAL_KILL_SWITCH_TRIGGERED", {"reason": reason})
        send_telegram(f"🔱 MASTER REPORT (Backend)\n\n"
                       f"🛑🛑🛑 GLOBAL KILL SWITCH ACTIVATED 🛑🛑🛑\n"
                       f"Reason: {reason}\n"
                       f"ALL strategies stopped. Force-closing open positions now.")
        self._force_close_all()

    def manual_kill_switch(self, reason: str = "Manual trigger"):
        """Aap khud kabhi bhi emergency me is function ko call karke sab band kar sakte ho."""
        with self._lock:
            self._trigger_global_kill(reason)

    def manual_resume(self):
        """Agli trading day shuru karne se pehle, ya galti se trigger hua ho to, manually resume."""
        with self._lock:
            self.state.global_kill_switch = False
            self.state.global_kill_reason = ""
            for st in self.state.strategies.values():
                st.paused = False
                st.pause_reason = ""
            audit_log("MANUAL_RESUME", {})
            send_telegram("🔱 MASTER REPORT (Backend)\n\n✅ System manually resumed. All strategies active.")

    # ---------- force close ----------

    def _force_close_all(self):
        for fn in self._force_close_handlers:
            try:
                fn()
            except Exception as e:
                audit_log("FORCE_CLOSE_HANDLER_ERROR", {"error": str(e)})
                send_telegram(f"🔱 MASTER REPORT (Backend)\n\n⚠️ Force-close handler error: {e}")
        self.state.force_close_done_today = True
        audit_log("FORCE_CLOSE_ALL_EXECUTED", {})

    # ---------- day rollover ----------

    def _check_day_rollover(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.state.date:
            audit_log("DAY_ROLLOVER", {"old_date": self.state.date, "new_date": today,
                                        "final_pnl": self.state.overall_pnl_today})
            send_telegram(f"🔱 MASTER REPORT (Backend)\n\n"
                           f"📅 Naya trading day: {today}\n"
                           f"Pichle din ka final P&L: {self.state.overall_pnl_today:.2f}\n"
                           f"Sab strategies aur counters reset ho gaye.")
            self.state = MasterState(date=today)

    # ---------- background watchdog thread ----------

    def start(self):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._watchdog_loop, daemon=True)
        t.start()
        send_telegram("🔱 MASTER REPORT (Backend)\n\n✅ Master Controller started aur active hai.")

    def _watchdog_loop(self):
        # Routine (non-emergency) reports SIRF in fixed times par jayenge.
        # Emergency alerts (pause/kill/force-close) hamesha turant jaate hain — yeh alag hai.
        SCHEDULED_REPORT_TIMES = [dtime(16, 0), dtime(18, 0), dtime(20, 0)]  # 4PM, 6PM, 8PM (last)
        sent_today_for = set()   # kis time-slot ka report is date me bhej diya, dobara na bheje
        last_date = datetime.now().strftime("%Y-%m-%d")

        while self._running:
            with self._lock:
                self._check_day_rollover()
                now = datetime.now()
                now_t = now.time()
                today = now.strftime("%Y-%m-%d")

                if today != last_date:
                    sent_today_for = set()
                    last_date = today

                # Force square-off ek hi baar per day
                if now_t >= FORCE_SQUARE_OFF_TIME and not self.state.force_close_done_today:
                    audit_log("FORCE_SQUARE_OFF_TIME_HIT", {})
                    send_telegram("🔱 MASTER REPORT (Backend)\n\n"
                                  "⏰ 3:25 PM — Force square-off time. Closing all positions.")
                    self._force_close_all()

                # Scheduled routine reports — sirf 4PM/6PM/8PM par, ek-ek baar
                for slot in SCHEDULED_REPORT_TIMES:
                    slot_key = f"{today}_{slot}"
                    if now_t >= slot and slot_key not in sent_today_for:
                        self._send_hourly_report()
                        sent_today_for.add(slot_key)

            time.sleep(15)   # har 15 second check — halka hai, koi heavy load nahi

    def _send_hourly_report(self):
        with self._lock:
            lines = [f"🔱 MASTER REPORT (Backend) — {datetime.now().strftime('%H:%M:%S')}",
                     "",
                     f"Overall P&L today: {self.state.overall_pnl_today:.2f}",
                     f"Kill switch: {'🛑 ACTIVE — ' + self.state.global_kill_reason if self.state.global_kill_switch else '✅ inactive'}",
                     ""]
            for st in self.state.strategies.values():
                status = f"⏸️ PAUSED ({st.pause_reason})" if st.paused else "✅ active"
                lines.append(f"  {st.name}: P&L {st.pnl_today:.2f} | SL hits {st.sl_hits_today} | {status}")
            send_telegram("\n".join(lines))

    def stop(self):
        self._running = False


# Singleton — Trishul Pro isi ko import karke use kare
MASTER = MasterController()


if __name__ == "__main__":
    # Quick standalone test (real broker se koi connection nahi, sirf logic test)
    MASTER.start()

    print(MASTER.validate_order("test_strategy", "BUY", "SENSEX-TEST"))
    MASTER.record_trade_close("test_strategy", pnl=-500, hit_sl=True)
    MASTER.record_trade_close("test_strategy", pnl=-600, hit_sl=True)
    MASTER.record_trade_close("test_strategy", pnl=-700, hit_sl=True)  # 3rd SL hit -> pause hona chahiye
    print(MASTER.validate_order("test_strategy", "BUY", "SENSEX-TEST"))  # ab blocked hona chahiye

    time.sleep(2)
