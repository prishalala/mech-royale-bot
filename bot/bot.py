"""Survival-first Mech Royale bot.

Decision loop (one call per observation, ~10 per match-second):
  1. update memory (enemies, damage, sounds), resolve upgrade offers
  2. threat assessment of every known mech
  3. pick a mode: ZONE (must move), RETREAT (hide + regen), FIGHT, LOOT, HOLD (camp with cover)
  4. movement: cached resumable A* on a 2 m grid + string pulling, stuck recovery, dodging
  5. combat: lead-aimed gun, ballistic mortar, laser charge/release, lightning cone checks;
     only fires when it's worth revealing ourselves
"""
from __future__ import annotations

import math
import random
import time

import mechfight as mf
from worldmap import WorldMap, PathJob, log

TAU = 2 * math.pi


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(dx, dy):
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return 0.0, 0.0
    return dx / d, dy / d


WEAPON_VALUE = {"lightning": 3.2, "laser": 3.0, "mortar": 1.2, "gun": 0.0}

# Upgrade preference for a survival bot. Unknown ids fall back to effect-based scoring.
UPGRADE_SCORE = {
    "gun_rate": 10.0, "gun_damage": 9.8, "armor": 9.5, "plating": 9.0, "nanites": 7.0,
    "laser_damage": 8.0, "laser_charge": 8.2,
    "lightning_efficiency": 8.0, "lightning_range": 6.0, "servos": 5.0,
    "gun_mag": 6.5, "capacitor": 4.0, "reactor": 4.0, "sensors": 4.2, "gyros": 3.5,
    "jets": 3.0, "gun_velocity": 3.5, "lightning_chain": 3.0, "laser_pierce": 2.0,
    "mortar_reload": 2.5, "mortar_splash": 2.0, "mortar_velocity": 2.0,
}


class Bot:
    name = "ironwall-survivor-v2"

    def __init__(self, hello):
        t0 = time.perf_counter()
        self.cfg = hello["config"]
        self.me_id = int(hello.get("slot", -1))
        self.rng = random.Random(int(hello.get("seed", 1)))
        mech = self.cfg.get("mech", {})
        self.wm = WorldMap(hello.get("map", {}).get("dir", "/arena"), mech)
        self.W = self.cfg.get("weapons", {})
        self.zone_cfg = self.cfg.get("zone", {})
        self.pickup_r = float(self.cfg.get("loot", {}).get("pickup_radius", 4))
        self.cube_r = float(self.cfg.get("cubes", {}).get("pickup_radius", 3.5))
        self.memory_s = float(self.cfg.get("vision", {}).get("memory_s", 5))
        self.upgrade_defs = {u["id"]: u for u in self.cfg.get("upgrades", [])}
        # memory
        self.enemies = {}        # id -> dict (last seen info)
        self.dmg_log = []        # (t, amount, by, bearing)
        self.sound_log = []      # (t, kind, bearing, dist)
        self.last_hurt = -99.0
        self.last_hit = -99.0
        self.dmg_by = {}         # attacker id -> last t
        self.t = 0.0
        # navigation
        self.job = None
        self.path = None
        self.path_goal = None
        self.path_t = -99.0
        self.path_i = 0
        self.pos_hist = []
        self.stuck_until = -1.0
        self.stuck_dir = (0.0, 0.0)
        self.stuck_count = 0
        self.stuck_ticks = 0
        self.stuck_mode = None
        # tactics
        self.camp = None
        self.camp_key = None
        self.camp_t = -99.0
        self.hide = None
        self.hide_t = -99.0
        self.strafe = 1
        self.strafe_t = 0.0
        self.scan_yaw = None
        self.mode = "start"
        self.bad_pickups = {}
        self.bad_crates = set()
        self.focus_id = None
        self.focus_t = -99.0
        log(f"bot init {1000 * (time.perf_counter() - t0):.0f} ms, map ok={self.wm.ok}")

    # ================================================================== main
    def act(self, obs):
        t_start = time.perf_counter()
        # Leave time for combat, serialization and the wrapper after navigation.
        self.deadline = t_start + 0.005
        self.obs = obs
        me = obs["self"]
        self.t = t = float(obs["t"])
        self.me = me
        self.pos = me["pos"]
        self.stats = me.get("stats", {})
        self.hpf = me["hp"] / max(1.0, me["max_hp"])
        self.scale = 1.0 + float(self.cfg.get("cubes", {}).get("size_per_cube", 0.05)) * me.get("cubes", 0)
        self.slots = {}
        for i, w in enumerate(me.get("weapons") or []):
            if w:
                self.slots[w["kind"]] = (i, w)
        act = {}
        self._events(obs)
        self._sounds(obs)
        self._track(obs)
        if obs.get("offer"):
            act["pick"] = self._pick_upgrade(obs["offer"])

        threats = self._assess(obs)
        self.threats = threats
        engaged = [e for e in threats if e["engaged"]]

        # --- choose mode and movement goal
        self.nav_jump = self.nav_boost = False
        mv, boost, jump, label = self._decide_move(obs, threats, engaged)
        if self.nav_boost:
            boost = True
            jump = jump or self.nav_jump
            label += " climb"

        # --- combat: aim + fire
        aim, fire, reload = self._combat(obs, threats, engaged)

        # --- dodging overlays
        mv, boost = self._dodge(obs, mv, boost)

        # --- stuck recovery
        mv, boost, jump = self._unstick(mv, boost, jump)

        # energy discipline: keep a reserve for emergencies
        if boost and me["energy"] < 12 and me.get("in_zone", True):
            boost = False

        act["move"] = [round(mv[0], 3), round(mv[1], 3)]
        act["boost"] = bool(boost)
        if jump:
            act["jump"] = True
        act["aim"] = [round(aim[0], 4), round(max(self.stats.get("min_pitch", -0.7), min(self.stats.get("max_pitch", 1.3), aim[1])), 4)]
        act["fire"] = fire
        if reload:
            act["reload"] = True
        tk = self._loot_take(obs)
        if tk:
            act.update(tk)
        mf.draw.label(f"{label} hp{me['hp']:.0f} e{me['energy']:.0f} {1000 * (time.perf_counter() - t_start):.0f}ms")
        return act

    # ================================================================== memory
    def _events(self, obs):
        t = self.t
        for e in obs.get("events") or []:
            ty = e.get("type")
            if ty == "damaged":
                self.last_hurt = t
                if e.get("by") is not None:
                    self.dmg_log.append((t, float(e.get("amount", 0)), e.get("by"), e.get("bearing")))
                    self.dmg_by[e["by"]] = t
            elif ty == "kill":
                # A dead focus target must not absorb decisions from tracking memory.
                self.enemies.pop(e["target"], None)
            elif ty == "hit":
                self.last_hit = t
            elif ty == "take_failed":
                self.bad_pickups[e.get("id")] = t
        self.dmg_log = [d for d in self.dmg_log if t - d[0] < 6.0]

    def _sounds(self, obs):
        t = self.t
        for s in obs.get("sounds") or []:
            self.sound_log.append((t - float(s.get("age", 0)), s.get("kind"), float(s.get("bearing", 0)), s.get("dist")))
        self.sound_log = [s for s in self.sound_log if t - s[0] < 4.0][-40:]

    def _track(self, obs):
        t = self.t
        for m in obs.get("mechs") or []:
            if m.get("kind") != "mech" or m.get("id") == self.me_id:
                continue
            e = self.enemies.get(m["id"]) or {"first": t}
            e.update(m)
            e["seen"] = t
            self.enemies[m["id"]] = e
        for k in [k for k, e in self.enemies.items() if t - e["seen"] > 20]:
            del self.enemies[k]

    # ================================================================== threat model
    def _dps(self, weapons, d, own=False):
        dps = 0.0
        for w in weapons:
            # Use upgraded slot stats for us and the live base weapon for opponents.
            st = self.slots[w][1]["stats"] if own else self.W.get(w, {})
            if w == "gun" and d <= st["range"]:
                cycle = st["mag"] * st["cooldown"] + st["reload"]
                dps += st["damage"] * st["mag"] / cycle
            elif w == "laser" and d <= st["range"]:
                dps += st["damage"] / (st["charge"] + st["cooldown"])
            elif w == "lightning" and d <= st["range"]:
                dps += st["damage"] / st["cooldown"]
            elif w == "mortar" and st["splash_radius"] * 2 < d < st["speed"] ** 2 / st["gravity"]:
                dps += st["splash"] / st["cooldown"]
        return dps

    def _assess(self, obs):
        t = self.t
        px, py, pz = self.pos
        out = []
        for eid, e in self.enemies.items():
            age = t - e["seen"]
            if age > 8:
                continue
            ex, ey, ez = e["pos"]
            d = math.hypot(ex - px, ey - py)
            to_me = math.atan2(py - ey, px - ex)
            aim_err = abs(wrap(e.get("aim", [0, 0])[0] - to_me))
            hit_me = t - self.dmg_by.get(eid, -99) < 4.0
            targeting = hit_me or (aim_err < 0.3 and d < 260 and age < 1.0)
            hpf = e.get("hp", 1400) / max(1.0, e.get("max_hp", 1400))
            weps = e.get("weapons") or ["gun"]
            dps = self._dps(weps, d)
            # The server tracks enemies through walls for vision.memory_s.
            # Count open firing lanes once and reuse them for target selection.
            size = 1 + self.cfg["cubes"]["size_per_cube"] * e.get("cubes", 0)
            vis = self.wm.los(px, py, pz + self.stats["muzzle_height"], ex, ey,
                              ez + self.cfg["mech"]["height"] * size / 2)
            threat = dps * (1.6 if targeting else 0.5) * (0.5 + 0.5 * hpf) * (1.0 if age < 1 else 0.6)
            engaged = vis and (age < 3.0) and (hit_me or (targeting and d < 220) or d < 70)
            out.append({"id": eid, "e": e, "d": d, "age": age, "targeting": targeting, "hit_me": hit_me,
                        "hpf": hpf, "hp": e.get("hp", 1400), "threat": threat, "engaged": engaged, "weps": weps,
                        "size": size, "vis": vis, "charging": bool(e.get("charging")) and vis and age < 0.5, "aim_err": aim_err})
        out.sort(key=lambda x: -x["threat"])
        return out

    def _my_dps(self, d):
        cubes = self.cfg["cubes"]
        return self._dps(self.slots, d, own=True) * (1 + cubes["damage_per_cube"] * min(self.me.get("cubes", 0), cubes["stat_cap"]))

    # ================================================================== zone
    def _circle_at(self, z, t):
        ss, se = z["shrink_start"], z["shrink_end"]
        if t <= ss or se <= ss:
            return z["center"], z["radius"]
        if t >= se:
            return z["next_center"], z["next_radius"]
        f = (t - ss) / (se - ss)
        c, n = z["center"], z["next_center"]
        return [c[0] + (n[0] - c[0]) * f, c[1] + (n[1] - c[1]) * f], z["radius"] + (z["next_radius"] - z["radius"]) * f

    def _zone_eta(self, z, p, margin=6.0):
        """Seconds until point p is outside the (shrinking) zone, capped at 60."""
        t = self.t
        for k in range(0, 61, 2):
            c, r = self._circle_at(z, t + k)
            if math.hypot(p[0] - c[0], p[1] - c[1]) > r - margin:
                return float(k)
        return 60.0

    def _safe_point(self, z):
        """Where to stand: inside the next circle, on our side of it, preferring cover."""
        nc, nr = z["next_center"], z["next_radius"]
        cc, cr = z["center"], z["radius"]
        px, py = self.pos[0], self.pos[1]
        key = (round(nc[0]), round(nc[1]), round(nr), round(cr))
        if self.camp is not None and self.camp_key == key and self.t - self.camp_t < 20:
            return self.camp
        dx, dy = px - nc[0], py - nc[1]
        d = math.hypot(dx, dy)
        frac = 0.55 if nr > 100 else 0.4
        want = min(d, nr * frac)
        ux, uy = unit(dx, dy) if d > 1 else (0.0, 0.0)
        bx, by = nc[0] + ux * want, nc[1] + uy * want
        if math.hypot(bx - cc[0], by - cc[1]) > cr - 5:
            # keep the spot inside the current circle too (matters while the last circles hold)
            f = 0.5
            bx, by = nc[0] + (cc[0] - nc[0]) * f * 0.3, nc[1] + (cc[1] - nc[1]) * f * 0.3
        wm = self.wm
        # The last circle contracts to one point. Sampling around it wastes HP.
        if nr == 0:
            return nc
        best, bs = None, -1e9
        rad = max(6.0, min(36.0, nr * 0.35))
        nf = wm.nearest_free(px, py, 4)
        my_comp = wm.comp[nf] if nf is not None else 0
        for k in range(64):
            if time.perf_counter() > self.deadline:
                break
            a = self.rng.random() * TAU
            r = rad * math.sqrt(self.rng.random())
            x, y = bx + r * math.cos(a), by + r * math.sin(a)
            idx = wm.idx(x, y)
            if not wm.free[idx]:
                continue
            if nr > 0 and math.hypot(x - nc[0], y - nc[1]) > nr * 0.8:
                continue
            if math.hypot(x - cc[0], y - cc[1]) > cr - 4:
                continue
            s = wm.cover(idx)
            if my_comp and wm.comp[idx] != my_comp:
                s -= 30.0          # reachable only by jetting over walls * 2.5 - r * 0.15 - (wm.pen[idx] * 0.5)
            if s > bs:
                bs, best = s, (x, y)
        if (best is None or bs < -50) and my_comp and nr > 0:
            # nothing reachable near the ideal spot: closest same-region cell inside the circle
            for k in range(80):
                a = self.rng.random() * TAU
                r = (nr - 4) * math.sqrt(self.rng.random())
                x, y = nc[0] + r * math.cos(a), nc[1] + r * math.sin(a)
                idx = wm.idx(x, y)
                if wm.free[idx] and wm.comp[idx] == my_comp and math.hypot(x - cc[0], y - cc[1]) < cr - 4:
                    sc = -math.hypot(x - bx, y - by) * 0.3 + wm.cover(idx)
                    if sc > bs:
                        bs, best = sc, (x, y)
        if best is None:
            k = wm.nearest_free(bx, by, max(3, int(nr / 2)))
            if k is not None:
                cx, cy = wm.center(k)
                if nr <= 0 or math.hypot(cx - nc[0], cy - nc[1]) < nr - 2:
                    best = (cx, cy)
        if best is None:
            best = (bx, by)
        self.camp, self.camp_key, self.camp_t = best, key, self.t
        return best

    # ================================================================== mode / movement
    def _decide_move(self, obs, threats, engaged):
        z = obs["zone"]
        me = self.me
        t = self.t
        p = self.pos
        walk = self.stats.get("walk_speed", 11.7)
        safe = self._safe_point(z)
        d_safe = dist2(p, safe)
        eta = self._zone_eta(z, p)
        need = d_safe / walk * 1.25 + 3.0
        in_zone = me.get("in_zone", True)
        # current circle check also, in case next circle will move away
        nc, nr = z["next_center"], z["next_radius"]
        in_next = math.hypot(p[0] - nc[0], p[1] - nc[1]) < nr - 5
        zone_urgent = (not in_zone) or (eta < need + 6 and not in_next)
        zone_boost = (not in_zone) or eta < need

        n_eng = len(engaged)
        recently_hurt = t - self.last_hurt < 2.5
        retreat = False
        if engaged:
            # seconds for them to kill us vs. for us to kill the main target (accuracy cancels roughly)
            their_dps = sum(max(40.0, self._dps(e["weps"], e["d"])) for e in engaged)
            ttk_me = me["hp"] / their_dps
            # Compare our survival window with the quickest available finish.
            ttk_them = min(e["hp"] / max(40.0, self._my_dps(e["d"])) for e in engaged)
            if (self.hpf < 0.35 or ttk_me < ttk_them * 0.45 or (n_eng >= 3 and self.hpf < 0.6)
                    or (n_eng >= 2 and self.hpf < 0.45)):
                retreat = True
            # cornered at very close range: fighting beats turning our back
            if engaged[0]["d"] < 18 and self.hpf > 0.25 and n_eng == 1:
                retreat = False

        if zone_urgent and (not engaged or not in_zone or eta < need):
            self.mode = "ZONE"
            # An uncontested rotation can spend more of the tick on long routes.
            if not threats:
                self.deadline += 0.003
            # Brake on approach instead of boosting back and forth across a tiny ring.
            if nr == 0 and d_safe < 4 and self.wm.walk_line(*p[:2], *safe):
                dt = self.cfg["match"]["decide_every"] / self.cfg["match"]["tick_hz"]
                step = max(walk * dt, d_safe)
                return ((safe[0] - p[0]) / step, (safe[1] - p[1]) / step), False, False, "zone-center"
            mv = self._navigate(safe)
            if not in_zone and mv == (0.0, 0.0):
                c, r = self._circle_at(z, t + 1)
                mv = self._steer_free(unit(c[0] - p[0], c[1] - p[1]))
            return mv, zone_boost or (retreat and me["energy"] > 40), False, "zone"

        if retreat:
            self.mode = "RETREAT"
            spot = self._hide_spot(engaged, z)
            mv = self._navigate(spot) if spot else self._away(engaged)
            return mv, me["energy"] > 35 and recently_hurt, False, "retreat"

        if engaged:
            self.mode = "FIGHT"
            # Arm the empty slot while shooting if the detour is short and walkable.
            if len(self.slots) == 1 and n_eng == 1:
                nearby = [pk for pk in obs.get("pickups", []) if pk["kind"] not in self.slots
                          and dist2(p, pk["pos"]) < walk * 1.5 and abs(pk["pos"][2] - p[2]) <= 4
                          and t - self.bad_pickups.get(pk["id"], -99) > 5
                          and self.wm.walk_line(*p[:2], *pk["pos"][:2])]
                if nearby:
                    pk = max(nearby, key=lambda item: WEAPON_VALUE.get(item["kind"], 0) / (1 + dist2(p, item["pos"]) / walk))
                    return self._navigate(pk["pos"], arrive=self.pickup_r * 0.75), False, False, "fight-equip"
            gun = self.slots.get("gun")
            other_ready = any(k in self.slots and self.slots[k][1].get("cooldown", 0) <= 0.3 for k in ("laser", "lightning", "mortar"))
            if gun and gun[1].get("reload", 0) > 0.5 and not other_ready:
                spot = self._hide_spot(engaged, z)
                if spot and dist2(p, spot) < 22:
                    return self._navigate(spot, arrive=1.5), False, False, "fight-reload"
            return self._fight_move(engaged[0], z), False, False, "fight"

        # after a fight: if hurt, stay hidden to regenerate before doing anything else
        recent = [e for e in threats if e["age"] < 6 and e["d"] < 200]
        if self.hpf < 0.7 and recent:
            self.mode = "RECOVER"
            # Once the wall breaks contact, stay put while regen runs.
            if me.get("in_zone", True) and all(not e["vis"] for e in recent) and t - self.last_hurt > 0.5:
                return (0.0, 0.0), False, False, "recover-covered"
            spot = self._hide_spot(recent, z)
            if spot:
                return self._navigate(spot), False, False, "recover"

        goal = self._loot_goal(obs, threats, eta, need)
        if goal is not None:
            self.mode = "LOOT"
            return self._navigate(goal[:2], arrive=0.6), False, False, "loot " + goal[2]

        self.mode = "HOLD"
        if d_safe < 3.0:
            return (0.0, 0.0), False, False, "hold"
        mv = self._navigate(safe)
        # walk slower (silently) when close and nothing is pressing
        if d_safe < 25 and not recently_hurt:
            mv = (mv[0] * 0.4, mv[1] * 0.4)
        return mv, False, False, "pos"

    def _away(self, engaged):
        px, py = self.pos[0], self.pos[1]
        vx = vy = 0.0
        for e in engaged:
            ex, ey = e["e"]["pos"][0], e["e"]["pos"][1]
            ux, uy = unit(px - ex, py - ey)
            vx += ux
            vy += uy
        return self._steer_free(unit(vx, vy))

    def _fight_move(self, tgt, z):
        px, py = self.pos[0], self.pos[1]
        e = tgt["e"]
        ex, ey = e["pos"][0], e["pos"][1]
        d = tgt["d"]
        if "lightning" in self.slots:
            pref = 28.0
        elif "laser" in self.slots:
            pref = 110.0
        elif "mortar" in self.slots:
            pref = 100.0
        else:
            pref = 75.0
        if "lightning" in tgt["weps"] and "lightning" not in self.slots:
            pref = max(pref, 70.0)
        if tgt["hpf"] < 0.3 and self.hpf > 0.5:
            pref = min(pref, 45.0)
        ux, uy = unit(ex - px, ey - py)
        if self.t > self.strafe_t:
            self.strafe = -self.strafe if self.rng.random() < 0.7 else self.strafe
            self.strafe_t = self.t + 0.6 + self.rng.random() * 1.0
        sx, sy = -uy * self.strafe, ux * self.strafe
        rad = max(-1.0, min(1.0, (d - pref) / max(15.0, pref * 0.4)))
        mx, my = sx * 0.85 + ux * rad, sy * 0.85 + uy * rad
        # keep inside the zone
        c, r = self._circle_at(z, self.t + 5)
        dc = math.hypot(px - c[0], py - c[1])
        if dc > r * 0.85:
            cx, cy = unit(c[0] - px, c[1] - py)
            mx, my = mx + cx * 1.2, my + cy * 1.2
        mv = unit(mx, my)
        if not self._clear_ahead(mv):
            self.strafe = -self.strafe
            self.strafe_t = self.t + 0.8
            mv = unit(-uy * self.strafe + ux * rad, ux * self.strafe + uy * rad)
            mv = self._steer_free(mv)
        return mv

    def _clear_ahead(self, mv, dist=5.0):
        x, y = self.pos[0] + mv[0] * dist, self.pos[1] + mv[1] * dist
        return self.wm.is_free(x, y)

    def _steer_free(self, mv):
        if mv == (0.0, 0.0) or self._clear_ahead(mv):
            return mv
        base = math.atan2(mv[1], mv[0])
        for da in (0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.2, -2.2):
            v = (math.cos(base + da), math.sin(base + da))
            if self._clear_ahead(v):
                return v
        return mv

    def _hide_spot(self, threats, z):
        """A nearby free cell with no line of sight from the threats, inside the zone."""
        t = self.t
        if self.hide is not None and t - self.hide_t < 0.6:
            return self.hide
        wm = self.wm
        px, py, pz = self.pos
        eyes = []
        for e in threats[:3]:
            ep = e["e"]["pos"]
            eyes.append((ep[0], ep[1], ep[2] + 8.0))
        for (dt, amt, by, br) in self.dmg_log[-3:]:
            if br is not None and by not in [e["id"] for e in threats]:
                eyes.append((px + 80 * math.cos(br), py + 80 * math.sin(br), pz + 8))
        if not eyes:
            return None
        c, r = self._circle_at(z, t + 8)
        best, bs = None, -1e9
        for rr in (7.0, 13.0, 20.0, 30.0):
            for k in range(14):
                if time.perf_counter() > self.deadline:
                    break
                a = k * TAU / 14 + rr * 0.1
                x, y = px + rr * math.cos(a), py + rr * math.sin(a)
                idx = wm.idx(x, y)
                if not wm.free[idx]:
                    continue
                if math.hypot(x - c[0], y - c[1]) > r - 4:
                    continue
                gz = wm.surf[idx] + 6.0
                hidden = 0
                closest = 1e9
                for (ex, ey, ez) in eyes:
                    if not wm.los(ex, ey, ez, x, y, gz, 3.0):
                        hidden += 1
                    closest = min(closest, math.hypot(x - ex, y - ey))
                s = 60.0 * hidden / len(eyes) - rr * 0.9 + min(closest, 120) * 0.15 + wm.cover(idx)
                if hidden == len(eyes):
                    s += 25
                if s > bs:
                    bs, best = s, (x, y)
        if best is not None:
            self.hide, self.hide_t = best, t
        return best

    def _loot_goal(self, obs, threats, eta, need):
        """(x, y, what) worth walking to, or None."""
        t = self.t
        px, py, pz = self.pos
        spare = eta - need  # seconds we can spend before the zone matters
        if spare < 8:
            return None
        late = obs["zone"]["next_radius"] < 100
        max_d = 25.0 if late else 110.0
        danger_near = any(e["age"] < 4 and e["d"] < 120 for e in threats)
        best, bs = None, 0.0
        # weapons
        cur2 = None
        for k in self.slots:
            if k != "gun":
                cur2 = k
        cur_v = WEAPON_VALUE.get(cur2, 0.0) if cur2 else 0.0
        for pk in obs.get("pickups") or []:
            if pk["id"] in self.bad_pickups and t - self.bad_pickups[pk["id"]] < 5:
                continue
            kind = pk.get("kind")
            if kind in self.slots:
                continue
            gain = WEAPON_VALUE.get(kind, 0) - cur_v
            if gain < 0.5:
                continue
            d = dist2(self.pos, pk["pos"])
            if d > max_d or abs(pk["pos"][2] - pz) > 4:
                continue
            s = gain * 30 / (1 + d / 25)
            if danger_near:
                s *= 0.4
            if s > bs:
                bs, best = s, (pk["pos"][0], pk["pos"][1], kind)
        # cubes
        if self.me.get("cubes", 0) < 12:
            for cb in obs.get("cubes") or []:
                d = dist2(self.pos, cb["pos"])
                if d > min(90.0, max_d) or abs(cb["pos"][2] - pz) > 4:
                    continue
                s = 20 / (1 + d / 20)
                if danger_near:
                    s *= 0.5
                if s > bs:
                    bs, best = s, (cb["pos"][0], cb["pos"][1], "cube")
        # supply drop: full heal + 3 cubes, but only if nobody is around
        for dr in obs.get("drops") or []:
            d = dist2(self.pos, dr["pos"])
            if d > 160:
                continue
            holder = dr.get("holder")
            if holder is not None and holder != self.me_id:
                continue
            crowd = any(e["age"] < 5 and dist2(e["e"]["pos"], dr["pos"]) < 90 for e in threats)
            if crowd:
                continue
            if dr.get("lands_in", 0) > d / 11.0 + 8:
                continue
            s = (25 + 30 * (1 - self.hpf)) / (1 + d / 60)
            if s > bs:
                bs, best = s, (dr["pos"][0], dr["pos"][1], "drop")
        # crates and drones: walk into shooting range (shooting happens in _combat)
        if not danger_near and not late:
            tgt = self._farm_target(obs)
            if tgt is not None:
                d = dist2(self.pos, tgt["pos"])
                if d > 40:
                    s = 8 / (1 + d / 40)
                    if s > bs:
                        bs, best = s, (tgt["pos"][0], tgt["pos"][1], "farm")
        return best

    def _farm_target(self, obs):
        """Crate or NPC drone to shoot for cubes when the coast is clear."""
        t = self.t
        if self.hpf < 0.6 or any(e["age"] < 5 and e["d"] < 160 for e in self.threats):
            return None
        if t - self.last_hurt < 3 and not any(d[2] is not None and d[2] >= 100 for d in self.dmg_log):
            return None
        px, py, pz = self.pos
        best, bd = None, 1e9
        for c in obs.get("crates") or []:
            if c["id"] in self.bad_crates:
                continue
            d = dist2(self.pos, c["pos"])
            if d < 90 and d < bd:
                best, bd = {"pos": c["pos"], "id": c["id"], "kind": "crate", "vel": [0, 0, 0], "h": 1.2}, d
        if self.me.get("cubes", 0) < 10 and self.hpf > 0.75:
            for m in obs.get("mechs") or []:
                if m.get("kind") == "npc":
                    d = dist2(self.pos, m["pos"])
                    if d < 120 and d * 0.8 < bd:
                        best, bd = {"pos": m["pos"], "id": m["id"], "kind": "npc", "vel": m.get("vel", [0, 0, 0]), "h": 1.75}, d * 0.8
        return best

    # ================================================================== navigation
    def _navigate(self, goal, arrive=2.5):
        wm = self.wm
        px, py = self.pos[0], self.pos[1]
        gx, gy = goal[0], goal[1]
        dgoal = math.hypot(gx - px, gy - py)
        if dgoal < arrive:
            return (0.0, 0.0)
        if not wm.ok:
            return unit(gx - px, gy - py)
        # straight shot if clear and short
        if dgoal < 40 and wm.walk_line(px, py, gx, gy):
            self.path = None
            self.job = None
            return unit(gx - px, gy - py)
        goal_changed = self.path_goal is None or dist2(self.path_goal, goal) > 6
        # Finish a time-sliced search for the same goal; a stale path's
        # refresh timer must not repeatedly discard unfinished work.
        if self.job is None or goal_changed or (self.job.done and self.t - self.path_t > 4.0):
            if goal_changed or self.path_t < 0:
                self.path = None
            s = wm.idx(px, py)
            g = wm.idx(gx, gy)
            if not wm.free[g] and not wm.solid[g]:
                g = wm.nearest_free(gx, gy, 3) or g     # wall margin: step to the open cell next to it
            self.job = PathJob(wm, s, g, climb=14.0)
            self.path_goal = (gx, gy)
            self.path_t = self.t
        if self.job is not None and not self.job.done:
            if self.job.step(4000, deadline=self.deadline):
                end = self.job.path[-1]
                ex, ey = wm.center(end)
                if math.hypot(ex - gx, ey - gy) > 8 and self.job.climb < 40:
                    # goal not reachable with small hops: allow big jet climbs (100 energy ~ 60 m)
                    self.job = PathJob(wm, self.job.start, self.job.goal, climb=45.0)
                else:
                    self.path = self.job.path
                    self.path_i = 0
        if self.path:
            return self._follow()
        return self._steer_free(unit(gx - px, gy - py))

    def _follow(self):
        wm = self.wm
        path = self.path
        px, py = self.pos[0], self.pos[1]
        # advance index to the closest node near the front
        n = len(path)
        best_i, bd = self.path_i, 1e9
        for i in range(self.path_i, min(n, self.path_i + 30)):
            cx, cy = wm.center(path[i])
            d = (cx - px) ** 2 + (cy - py) ** 2
            if d < bd:
                bd, best_i = d, i
        if bd > 12 ** 2:
            self.path_t = -99  # off path: replan soon
        self.path_i = best_i
        # string pulling: farthest visible node within a window
        target = path[min(n - 1, best_i + 1)]
        for k in (24, 16, 10, 6, 3):
            i = best_i + k
            if i >= n:
                i = n - 1
            cx, cy = wm.center(path[i])
            if wm.walk_line(px, py, cx, cy):
                target = path[i]
                break
        # climbing: a path cell just ahead is above our feet -> jump and jet toward it
        pz = self.pos[2]
        for i in range(best_i, min(n, best_i + 5)):
            k = path[i]
            h = wm.hgt[k]
            cx, cy = wm.center(k)
            if math.hypot(cx - px, cy - py) > 9:
                break
            if h > pz + wm.step_up + 0.3:
                need = (h - pz) * 1.8 + 8
                if self.me.get("grounded", True) and self.me["energy"] < min(need, 95) and self.me.get("in_zone", True):
                    return (0.0, 0.0)          # wait for energy at the foot of the wall
                self.nav_boost = True
                self.nav_jump = bool(self.me.get("grounded", True))
                nxt = path[min(n - 1, i + 1)]
                tx, ty = wm.center(nxt)
                if pz < h + 0.5:
                    # still below the edge: push toward the wall lightly while rising
                    tx, ty = cx, cy
                return unit(tx - px, ty - py)
        cx, cy = wm.center(target)
        return unit(cx - px, cy - py)

    def _unstick(self, mv, boost, jump):
        """Commanded to walk but not moving for ~0.8 s: hop + jet briefly, then try a side direction."""
        t = self.t
        combat = self.mode in ("FIGHT", "RETREAT")
        if t < self.stuck_until:
            if combat:
                self.stuck_until = -1.0
                return mv, boost, jump
            grounded = self.me.get("grounded", False)
            return self.stuck_dir, self.me["energy"] > 15, grounded
        spd = math.hypot(mv[0], mv[1])
        v = self.me.get("vel") or [0, 0, 0]
        actual = math.hypot(v[0], v[1])
        walk = self.stats.get("walk_speed", 11.7)
        if spd > 0.3 and actual < 0.15 * walk * spd and self.me.get("grounded", True) and self.mode == self.stuck_mode:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        self.stuck_mode = self.mode
        if self.stuck_ticks >= 8:
            self.stuck_ticks = 0
            self.stuck_count += 1
            self.path_t = -99          # replan
            a = math.atan2(mv[1], mv[0])
            if combat:
                self.strafe = -self.strafe
                a += self.rng.choice((1.2, -1.2))
                return (math.cos(a), math.sin(a)), boost, False
            if self.stuck_count % 3 == 0:
                a += self.rng.choice((1.4, -1.4))
            self.stuck_dir = (math.cos(a), math.sin(a))
            self.stuck_until = t + 0.7
            return self.stuck_dir, self.me["energy"] > 15, True
        return mv, boost, jump

    def _dodge(self, obs, mv, boost):
        px, py, pz = self.pos
        g = float(self.W.get("mortar", {}).get("gravity", 22))
        sr = float(self.W.get("mortar", {}).get("splash_radius", 7)) + 4.0
        # mortar shells: predict where they land
        for pr in obs.get("projectiles") or []:
            if pr.get("kind") != "mortar":
                continue
            x, y, zz = pr["pos"]
            vx, vy, vz = pr["vel"]
            # solve zz + vz t - g t^2/2 = pz + 1
            a, b, c = -0.5 * g, vz, zz - (pz + 1.0)
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            tl = (-b - math.sqrt(disc)) / (2 * a)
            if tl < 0 or tl > 3.0:
                continue
            lx, ly = x + vx * tl, y + vy * tl
            dl = math.hypot(lx - px, ly - py)
            if dl < sr:
                ux, uy = unit(px - lx, py - ly)
                if ux == 0 and uy == 0:
                    ux, uy = unit(-vy, vx)
                return self._steer_free((ux, uy)), self.me["energy"] > 10
        # charging laser pointed at us: sidestep
        for e in self.threats:
            if e["charging"] and e["aim_err"] < 0.2 and e["d"] < 280:
                ex, ey = e["e"]["pos"][0], e["e"]["pos"][1]
                ux, uy = unit(ex - px, ey - py)
                sx, sy = -uy * self.strafe, ux * self.strafe
                return self._steer_free((sx, sy)), self.me["energy"] > 20
        return mv, boost

    # ================================================================== combat
    def _combat(self, obs, threats, engaged):
        me = self.me
        px, py, pz = self.pos
        mz = pz + self.stats.get("muzzle_height", 7.0)
        fire = [False, False]
        reload = False
        # --- target selection: mechs first
        tgt = None
        best = -1e9
        for e in threats:
            if e["age"] > 1.5:
                continue
            d = e["d"]
            if d > 285:
                continue
            vis = e["vis"]
            # seconds to kill it (with a hit-probability falloff by range): finish the weakest first
            hitp = 1.0 / (1.0 + (d / 130.0) ** 2)
            ttk = e["hp"] / (max(40.0, self._my_dps(d)) * hitp)
            s = -ttk
            if e["id"] == self.focus_id and self.t - self.focus_t < 3.0:
                s += 2.5 + 0.15 * ttk          # sticky: don't spread damage (they regen after 5 s)
            if e["hit_me"]:
                s += 1.0
            elif e["targeting"]:
                s += 0.5
            if not vis:
                s -= 60.0
            if s > best:
                best, tgt = s, e
        should = False
        if tgt is not None:
            should = (tgt["engaged"] or tgt["d"] < 80 or
                      (tgt["hpf"] < 0.4 and self.hpf > 0.5 and tgt["d"] < 160) or
                      (self.hpf > 0.85 and tgt["hpf"] < 0.6 and tgt["d"] < 120) or
                      (obs.get("alive", 16) <= 3 and self.hpf > 0.5))
            should = should or (tgt["d"] < 180 and self.hpf > 0.55)
            if self.mode == "RECOVER" and not tgt["engaged"]:
                should = False  # outgoing damage would restart the regen delay
            if not tgt["vis"] and "mortar" not in self.slots:
                should = should and tgt["d"] < 60
            if should and tgt["vis"]:
                self.focus_id, self.focus_t = tgt["id"], self.t
        farm = None
        if tgt is None or not should:
            farm = self._npc_attacker(obs)
            if farm is None and self.mode in ("LOOT", "HOLD"):
                farm = self._farm_target(obs)
        # A tracked opponent behind a wall cannot justify exposing rear armor
        # to a fresh damage bearing. The legs keep following their current goal.
        if self.dmg_log and (tgt is None or not tgt["vis"]):
            when, _, attacker, bearing = self.dmg_log[-1]
            if bearing is not None and self.t - when < 1 and (tgt is None or attacker != tgt["id"]):
                return [bearing, 0.0], fire, reload
        if tgt is not None and (should or farm is None):
            size = tgt["size"]
            info = {"pos": tgt["e"]["pos"], "vel": tgt["e"].get("vel", [0, 0, 0]), "h": self.cfg["mech"]["height"] * size / 2,
                    "r": self.cfg["mech"]["radius"] * size, "vis": tgt["vis"]}
            aim, fire, reload = self._shoot(info, should)
            return aim, fire, reload
        if farm is not None:
            fp = farm["pos"]
            d = dist2(self.pos, fp)
            vis = self.wm.los(px, py, mz, fp[0], fp[1], fp[2] + farm["h"])
            if d < 85 and vis:
                info = {"pos": fp, "vel": farm["vel"], "h": farm["h"], "r": 1.4 if farm["kind"] == "crate" else 1.8, "vis": True}
                aim, fire, reload = self._shoot(info, True, farm=True)
                return aim, fire, reload
        # --- no target: look where danger is likely, else scan
        yaw = self._look_yaw()
        gun = self.slots.get("gun")
        if gun and gun[1].get("ammo", 99) < gun[1]["stats"].get("mag", 12) * 0.6 and gun[1].get("reload", 0) == 0:
            reload = True
        # don't leave a laser charge hanging
        return [yaw, 0.0], fire, reload

    def _npc_attacker(self, obs):
        t = self.t
        best, bd = None, 1e9
        for m in obs.get("mechs") or []:
            if m.get("kind") == "npc" and t - self.dmg_by.get(m["id"], -99) < 4.0:
                d = dist2(self.pos, m["pos"])
                if d < 120 and d < bd:
                    best, bd = {"pos": m["pos"], "id": m["id"], "kind": "npc", "vel": m.get("vel", [0, 0, 0]), "h": 1.75}, d
        return best

    def _look_yaw(self):
        t = self.t
        # recent damage from an unseen attacker
        for (dt, amt, by, br) in reversed(self.dmg_log):
            if t - dt < 2.0 and br is not None:
                return br
        # engaged-ish enemies we remember
        px, py = self.pos[0], self.pos[1]
        for e in self.threats:
            if e["age"] < 6 and e["d"] < 200:
                ep = e["e"]["pos"]
                return math.atan2(ep[1] - py, ep[0] - px)
        # combat sounds nearby
        for (st, kind, br, dist) in reversed(self.sound_log):
            if t - st < 1.5 and dist in ("near", "mid") and kind in ("gun", "laser", "laser_charge", "lightning", "step", "boost", "jet", "mortar"):
                return br
        # sweep
        if self.scan_yaw is None:
            self.scan_yaw = self.me["aim"][0]
        self.scan_yaw = wrap(self.scan_yaw + 0.24)
        return self.scan_yaw

    def _lead(self, origin, tp, tv, speed):
        """Intercept point for a straight projectile."""
        rx, ry, rz = tp[0] - origin[0], tp[1] - origin[1], tp[2] - origin[2]
        vx, vy, vz = tv
        a = vx * vx + vy * vy + vz * vz - speed * speed
        # Flight starts ahead of the muzzle along the aim ray:
        # |relative_position + velocity * t| = barrel_offset + speed * t.
        offset = self.stats.get("radius", self.cfg["mech"]["radius"] * self.scale) + 0.5
        b = 2 * (rx * vx + ry * vy + rz * vz - offset * speed)
        c = rx * rx + ry * ry + rz * rz - offset * offset
        tt = None
        if abs(a) < 1e-6:
            tt = -c / b if b != 0 else None
        else:
            disc = b * b - 4 * a * c
            if disc >= 0:
                sq = math.sqrt(disc)
                cands = [x for x in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if x > 0]
                tt = min(cands) if cands else None
        if tt is None:
            tt = max(0.0, math.sqrt(max(0, c + offset * offset)) - offset) / speed
        tt = max(0, min(tt, 2.0))
        return (tp[0] + vx * tt, tp[1] + vy * tt, tp[2] + vz * tt), tt

    def _shoot(self, info, should, farm=False):
        me = self.me
        px, py, pz = self.pos
        mz = pz + self.stats.get("muzzle_height", 7.0)
        tp = (info["pos"][0], info["pos"][1], info["pos"][2] + info["h"])
        tv = info["vel"]
        d = math.hypot(tp[0] - px, tp[1] - py)
        cur = me["aim"]
        fire = [False, False]
        reload = False
        gun = self.slots.get("gun")
        laser = self.slots.get("laser")
        light = self.slots.get("lightning")
        mortar = self.slots.get("mortar")
        # default aim: gun lead (or direct)
        speed = gun[1]["stats"].get("speed", 252) if gun else 252
        lp, tt = self._lead((px, py, mz), tp, tv, speed)
        aim_yaw = math.atan2(lp[1] - py, lp[0] - px)
        aim_pitch = math.atan2(lp[2] - mz, max(1.0, math.hypot(lp[0] - px, lp[1] - py)))
        tol = math.atan2(info["r"] * 0.8, max(d, 1.0))
        mode = "gun"
        # laser: once charged, aim for the beam (hitscan) and release when on target
        if laser and should and not farm:
            li, lw = laser
            st = lw["stats"]
            full = st.get("charge", 0.75)
            charge = lw.get("charge", 0) or 0
            if charge > 0 and charge >= full * 0.8:
                mode = "laser"
                p2 = (tp[0] + tv[0] * 0.1, tp[1] + tv[1] * 0.1, tp[2])
                aim_yaw = math.atan2(p2[1] - py, p2[0] - px)
                aim_pitch = math.atan2(p2[2] - mz, max(1.0, d))
        # mortar: lob when the gun has no shot
        use_mortar = False
        if mortar and should and not farm and mode == "gun":
            mi, mw = mortar
            if mw.get("cooldown", 0) <= 0 and 25 < d < 230:
                gun_busy = (not gun) or gun[1].get("reload", 0) > 0 or gun[1].get("ammo", 1) == 0 or not info["vis"] or d > 200
                if gun_busy:
                    sol = self._mortar_solve(mw["stats"], tp, tv, info["vis"])
                    if sol:
                        use_mortar = True
                        mode = "mortar"
                        aim_yaw, aim_pitch = sol
        # error between our actual torso aim and the solution
        yaw_err = abs(wrap(cur[0] - aim_yaw))
        pitch_err = abs(cur[1] - aim_pitch)
        # Both angles must fit the observed target, including at point-blank range.
        pitch_tol = math.atan2(info["h"] * 0.8, max(d, 1.0))
        on = yaw_err < tol and pitch_err < pitch_tol
        # Gun gate at fire time. self.aim is one decision behind a moving lead point,
        # but the torso slews turn_rate/tick_hz per tick toward the aim we send now.
        # So fire when one tick of slew puts the torso within tolerance of the
        # solution. Far-off aims still hold fire until the torso arrives.
        mech = self.cfg["mech"]
        tick = 1.0 / self.cfg["match"]["tick_hz"]
        turn = self.stats.get("turn_rate", mech["turn_rate"]) * tick
        prate = self.stats.get("pitch_rate", mech["pitch_rate"]) * tick
        gun_on = yaw_err < tol + turn and pitch_err < pitch_tol + prate
        if should:
            if gun and info["vis"] and d < gun[1]["stats"].get("range", 230) and mode == "gun":
                gw = gun[1]
                if gw.get("ammo", 1) > 0 and gw.get("reload", 0) == 0:
                    if gun_on:
                        fire[gun[0]] = True
            if use_mortar and yaw_err < 0.03 and pitch_err < 0.02:
                fire[mortar[0]] = True
            if light and not farm:
                li, lw = light
                st = lw["stats"]
                rng = st.get("range", 45)
                # The configured cone is the allowed yaw error on either side.
                cone = math.radians(st["cone_deg"])
                dy = abs(wrap(cur[0] - math.atan2(tp[1] - py, tp[0] - px)))
                if d < rng and dy < cone and info["vis"] and lw.get("cooldown", 0) < self.cfg["match"]["decide_every"] / self.cfg["match"]["tick_hz"] \
                        and me["energy"] >= st.get("energy", 3) + 6:
                    fire[li] = True
            if laser and not farm:
                li, lw = laser
                st = lw["stats"]
                full = st.get("charge", 0.75)
                charge = lw.get("charge", 0) or 0
                rng = st.get("range", 280)
                if charge > 0:
                    held_too_long = charge >= full + st.get("hold_s", 2.0) * 0.7
                    if charge >= full and (self._beam_on(cur, aim_yaw, aim_pitch, tp, tv, tol, pitch_tol, turn, prate)
                                           or held_too_long):
                        fire[li] = False   # release: fires now along our actual aim
                    elif not info["vis"] and charge >= full and yaw_err > 0.3:
                        fire[li] = False
                    else:
                        fire[li] = True
                elif lw.get("cooldown", 0) <= 0 and d < rng * 0.95 and info["vis"] and yaw_err < 0.5:
                    fire[li] = True        # start charging
        elif farm:
            if gun and d < 90 and gun_on:
                gw = gun[1]
                if gw.get("ammo", 1) > 0 and gw.get("reload", 0) == 0:
                    fire[gun[0]] = True
        else:
            # watching, not shooting: reload if needed
            if gun and gun[1].get("ammo", 99) < gun[1]["stats"].get("mag", 12) * 0.5 and gun[1].get("reload", 0) == 0:
                reload = True
        # a laser charge in progress without a shot: keep holding (it auto-fires at hold_s)
        if laser and not fire[laser[0]] and (laser[1].get("charge", 0) or 0) > 0 and not should:
            fire[laser[0]] = (laser[1].get("charge", 0) or 0) < laser[1]["stats"].get("charge", 0.75)
        return [aim_yaw, aim_pitch], fire, reload

    def _beam_on(self, cur, aim_yaw, aim_pitch, tp, tv, tol, pitch_tol, turn, prate):
        """Will a laser released now hit? Letting go applies on the next tick, and the
        protocol doesn't say whether the torso slews before or after it. So require a
        hit both along the current aim and along the aim after one tick of slew toward
        the command, against the target where it will be at release (relative motion).
        The old check compared the aim to the 0.1 s lead point, so it held a ready shot
        while the target moved fast."""
        tick = 1.0 / self.cfg["match"]["tick_hz"]
        px, py, pz = self.pos
        mz = pz + self.stats.get("muzzle_height", 7.0)
        sv = self.me.get("vel") or (0.0, 0.0, 0.0)
        rx = tp[0] + (tv[0] - sv[0]) * tick - px
        ry = tp[1] + (tv[1] - sv[1]) * tick - py
        rz = tp[2] + (tv[2] - sv[2]) * tick - mz
        byaw = math.atan2(ry, rx)
        bpitch = math.atan2(rz, max(1.0, math.hypot(rx, ry)))
        syaw = cur[0] + max(-turn, min(turn, wrap(aim_yaw - cur[0])))
        spitch = cur[1] + max(-prate, min(prate, aim_pitch - cur[1]))
        return all(abs(wrap(y - byaw)) < tol and abs(p - bpitch) < pitch_tol
                   for y, p in ((cur[0], cur[1]), (syaw, spitch)))

    def _mortar_solve(self, st, tp, tv, vis):
        px, py, pz = self.pos
        v = st.get("speed", 72)
        g = st.get("gravity", 22)
        # Reported height already includes the mech's cubes.
        lz = pz + self.stats.get("height", self.cfg["mech"]["height"] * self.scale) + 0.5
        aim_p = tp
        sol = None
        for _ in range(3):
            dx, dy = aim_p[0] - px, aim_p[1] - py
            d = math.hypot(dx, dy)
            h = (aim_p[2] - 3.0) - lz
            disc = v ** 4 - g * (g * d * d + 2 * h * v * v)
            if disc < 0 or d < 1:
                return None
            sq = math.sqrt(disc)
            lo = math.atan((v * v - sq) / (g * d))
            hi = math.atan((v * v + sq) / (g * d))
            th = lo if vis else hi
            if th > self.stats.get("max_pitch", 1.3):
                th = lo
            tfl = d / (v * math.cos(th))
            sol = (math.atan2(dy, dx), th)
            aim_p = (tp[0] + tv[0] * tfl, tp[1] + tv[1] * tfl, tp[2])
        return sol

    # ================================================================== loot & upgrades
    def _loot_take(self, obs):
        px, py, pz = self.pos
        cur2 = None
        cur2_slot = None
        for k, (i, w) in self.slots.items():
            if k != "gun":
                cur2, cur2_slot = k, i
        empty = None
        for i, w in enumerate(self.me.get("weapons") or []):
            if w is None:
                empty = i
        for pk in obs.get("pickups") or []:
            kind = pk.get("kind")
            if kind in self.slots:
                continue
            if math.hypot(pk["pos"][0] - px, pk["pos"][1] - py) > self.pickup_r - 0.3 or abs(pk["pos"][2] - pz) > 3.8:
                continue
            if empty is not None and kind != "gun":
                return {"take": pk["id"]}
            if cur2 is not None and WEAPON_VALUE.get(kind, 0) > WEAPON_VALUE.get(cur2, 0) + 0.5:
                w = self.slots[cur2][1]
                if cur2 == "laser" and (w.get("charge", 0) or 0) > 0:
                    continue
                return {"take": pk["id"], "drop": cur2_slot}
        return None

    def _pick_upgrade(self, offer):
        best, bi = -1e9, 0
        for i, ch in enumerate(offer.get("choices") or []):
            s = self._score_upgrade(ch.get("id"))
            if s > best:
                best, bi = s, i
        return bi

    def _score_upgrade(self, uid):
        u = self.upgrade_defs.get(uid)
        req = u.get("requires") if u else None
        if req and req not in self.slots:
            return 0.5
        if uid in UPGRADE_SCORE:
            s = UPGRADE_SCORE[uid]
        elif u:
            s = 1.0
            for ef in u.get("effects", []):
                p = ef.get("path", "")
                if "armor" in p or p.endswith(".hp") or "regen" in p:
                    s += 5
                elif "damage" in p or "rate" in p or "cooldown" in p:
                    s += 3
                else:
                    s += 1
        else:
            s = 1.0
        if req and req != "gun":
            s += 0.5
        return s


if __name__ == "__main__":
    mf.run(Bot)
