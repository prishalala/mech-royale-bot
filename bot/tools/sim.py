"""Crude single-bot kinematic simulator for smoke-testing navigation, zone play and crash-freedom.

NOT the real game: simplified walls/slopes, no jets, scripted enemies that walk at you and shoot
with a fixed hit chance. Use it to catch exceptions, stuck spots and zone deaths, not to tune skill.

    python tools/sim.py --matches 20 --enemies 3 --arena ../arena
"""
import argparse
import gzip
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import mechfight as mf  # noqa: E402
import bot as botmod  # noqa: E402


def load_hello(kit):
    with gzip.open(os.path.join(kit, "check", "stream.jsonl.gz"), "rt") as f:
        return json.loads(f.readline())


class Sim:
    def __init__(self, hello, arena, seed, n_enemies):
        self.rng = random.Random(seed)
        self.hello = json.loads(json.dumps(hello))
        self.hello["map"]["dir"] = arena
        self.hello["seed"] = seed
        self.cfg = self.hello["config"]
        self.bot = botmod.Bot(self.hello)
        self.wm = self.bot.wm
        self.n_enemies = n_enemies
        z = self.cfg["zone"]
        self.zc, self.zr = list(z["start_center"]), z["start_radius"]
        self.phase = 0
        self.t_phase = 0.0
        self.next = self._new_next(self.zc, self.zr, z["phases"][0]["radius"])
        self.pos = self._spawn()
        self.vz = 0.0
        self.grounded = True
        self.vel = [0.0, 0.0, 0.0]
        self.aim = [0.0, 0.0]
        self.hp = self.maxhp = 1400.0
        self.energy = 100.0
        self.cubes = 0
        self.weapons = [{"kind": "gun", "cooldown": 0, "ammo": 12, "reload": 0, "stats": dict(self.cfg["weapons"]["gun"])}, None]
        self.last_dmg = -99
        self.pickups = []
        for k in range(60):
            x, y = self._rand_free()
            self.pickups.append({"id": 8000 + k, "kind": self.rng.choice(["laser", "lightning", "mortar"]), "pos": [x, y, self.wm.surface(x, y)]})
        self.cubes_list = []
        for k in range(60):
            x, y = self._rand_free()
            self.cubes_list.append({"id": 5000 + k, "pos": [x, y, self.wm.surface(x, y)]})
        self.enemies = []
        self.offer = None
        self.thr = list(self.cfg["cubes"]["upgrade_thresholds"])
        self.stats = {"stuck_events": 0, "zone_dmg": 0.0, "max_ms": 0.0, "ms": [], "taken": 0}

    def _rand_free(self):
        while True:
            a = self.rng.random() * math.tau
            r = self.zr * 0.8 * math.sqrt(self.rng.random())
            x, y = self.zc[0] + r * math.cos(a), self.zc[1] + r * math.sin(a)
            if self.wm.is_free(x, y):
                return x, y

    def _spawn(self):
        x, y = self._rand_free()
        if os.environ.get("SPAWN"):
            x, y = map(float, os.environ["SPAWN"].split(","))
        return [x, y, self.wm.surface(x, y)]

    def _new_next(self, c, r, nr):
        a = self.rng.random() * math.tau
        off = (r - nr) * math.sqrt(self.rng.random()) * 0.95
        return [c[0] + off * math.cos(a), c[1] + off * math.sin(a)], nr

    def zone_obs(self, t):
        z = self.cfg["zone"]
        ph = z["phases"][min(self.phase, len(z["phases"]) - 1)]
        ss = self.t_phase + ph["hold_s"]
        se = ss + ph["shrink_s"]
        return {"center": self.zc, "radius": self.zr, "next_center": self.next[0], "next_radius": self.next[1],
                "shrink_start": ss, "shrink_end": se, "dps": ph["dps"]}

    def zone_now(self, t):
        z = self.zone_obs(t)
        if t <= z["shrink_start"]:
            return z["center"], z["radius"], z
        f = min(1.0, (t - z["shrink_start"]) / (z["shrink_end"] - z["shrink_start"]))
        c = [z["center"][0] + (z["next_center"][0] - z["center"][0]) * f, z["center"][1] + (z["next_center"][1] - z["center"][1]) * f]
        return c, z["radius"] + (z["next_radius"] - z["radius"]) * f, z

    def advance_zone(self, t):
        z = self.zone_obs(t)
        if t >= z["shrink_end"] and self.phase < len(self.cfg["zone"]["phases"]) - 1:
            self.zc, self.zr = list(self.next[0]), self.next[1]
            self.phase += 1
            self.t_phase = t
            nr = self.cfg["zone"]["phases"][self.phase]["radius"]
            self.next = self._new_next(self.zc, self.zr, nr)

    def collide(self, x, y, z):
        return self.clearance(x, y, z) < 2.5

    def clearance(self, x, y, z):
        wm = self.wm
        best = 9.0
        i, j = wm.cell(x, y)
        for dj in (-2, -1, 0, 1, 2):
            for di in (-2, -1, 0, 1, 2):
                ii, jj = i + di, j + dj
                if not (0 <= ii < wm.NX and 0 <= jj < wm.NY):
                    return 0.0
                k = jj * wm.NX + ii
                if wm.solid[k] and wm.btop[k] > z + 1.2:
                    cx0, cy0 = wm.X0 + ii * 2, wm.Y0 + jj * 2
                    nx_ = min(max(x, cx0), cx0 + 2)
                    ny_ = min(max(y, cy0), cy0 + 2)
                    best = min(best, ((nx_ - x) ** 2 + (ny_ - y) ** 2) ** 0.5)
        return best

    def ground(self, x, y):
        h = self.wm.hgt
        i = self.wm.idx
        return max(h[i(x, y)], h[i(x + 1.2, y)], h[i(x - 1.2, y)], h[i(x, y + 1.2)], h[i(x, y - 1.2)])

    def try_move(self, dx, dy):
        x, y, z = self.pos
        c0 = self.clearance(x, y, z)
        for (mx, my) in ((dx, dy), (dx, 0), (0, dy)):
            nx_, ny_ = x + mx, y + my
            c1 = self.clearance(nx_, ny_, z)
            if c1 < 2.5 and c1 < c0 + 1e-6:
                continue
            g = self.ground(nx_, ny_)
            if self.grounded and g - z > 1.25 + 1.19 * math.hypot(mx, my):
                continue
            if not self.grounded and g > z + 0.3:
                continue
            self.pos[0], self.pos[1] = nx_, ny_
            return True
        return False

    def vertical(self, act, dt):
        g = self.ground(self.pos[0], self.pos[1])
        if self.grounded:
            if act.get("jump"):
                self.vz = 11.0
                self.grounded = False
            elif g < self.pos[2] - 0.6:
                self.grounded = False
            else:
                self.pos[2] = g
        if not self.grounded:
            self.vz -= 22 * dt
            if act.get("boost") and self.energy > 0:
                self.vz = min(10.0, self.vz + 34 * dt)
                self.energy -= 15 * dt
            self.pos[2] += self.vz * dt
            if self.pos[2] <= g:
                self.pos[2] = g
                self.vz = 0.0
                self.grounded = True

    def spawn_enemies(self, t):
        while len(self.enemies) < self.n_enemies:
            a = self.rng.random() * math.tau
            r = 120 + self.rng.random() * 80
            x, y = self.pos[0] + r * math.cos(a), self.pos[1] + r * math.sin(a)
            if self.wm.is_free(x, y):
                self.enemies.append({"id": len(self.enemies) + 1 + self.rng.randint(0, 9) * 0, "pos": [x, y, self.wm.surface(x, y)],
                                     "hp": 1400.0, "max_hp": 1400.0, "weapons": [self.rng.choice(["gun", "gun", "laser", "lightning"])],
                                     "cd": 0.0, "born": t})
                for k, e in enumerate(self.enemies):
                    e["id"] = k + 1

    def run(self):
        t = 0.0
        tick = 0
        idle_pos = None
        idle_t = 0.0
        while t < self.cfg["match"]["duration_s"] and self.hp > 0:
            self.advance_zone(t)
            if t > 20:
                self.spawn_enemies(t)
            c, r, zobs = self.zone_now(t)
            in_zone = math.hypot(self.pos[0] - c[0], self.pos[1] - c[1]) <= r
            events = []
            mechs = []
            eye = (self.pos[0], self.pos[1], self.pos[2] + 8)
            for e in self.enemies:
                d = math.hypot(e["pos"][0] - self.pos[0], e["pos"][1] - self.pos[1])
                vis = d < 200 and self.wm.los(eye[0], eye[1], eye[2], e["pos"][0], e["pos"][1], e["pos"][2] + 6)
                to_me = math.atan2(self.pos[1] - e["pos"][1], self.pos[0] - e["pos"][0])
                if vis:
                    mechs.append({"id": e["id"], "kind": "mech", "pos": list(e["pos"]), "vel": e.get("vel", [0, 0, 0]), "aim": [to_me, 0],
                                  "hp": e["hp"], "max_hp": 1400, "cubes": 0, "weapons": e["weapons"], "charging": False})
            vis_items = lambda L: [it for it in L if math.hypot(it["pos"][0] - self.pos[0], it["pos"][1] - self.pos[1]) < 60]
            obs = {"type": "obs", "tick": tick, "t": round(t, 2), "alive": 16,
                   "self": {"id": 0, "pos": list(self.pos), "vel": list(self.vel), "aim": list(self.aim), "hp": self.hp, "max_hp": self.maxhp,
                            "energy": self.energy, "max_energy": 100, "grounded": self.grounded, "cubes": self.cubes, "upgrades": [],
                            "weapons": self.weapons, "in_zone": in_zone, "stats": dict(self.cfg["mech"])},
                   "mechs": mechs, "projectiles": [], "crates": [], "cubes": vis_items(self.cubes_list),
                   "pickups": vis_items(self.pickups), "sounds": [], "beams": [], "events": self.pending_events if hasattr(self, "pending_events") else [],
                   "drops": [], "zone": zobs, "offer": self.offer}
            obs = json.loads(json.dumps(obs))
            t0 = time.perf_counter()
            mf.draw.clear()
            act = self.bot.act(obs)
            json.dumps(act)
            ms = (time.perf_counter() - t0) * 1000
            self.stats["ms"].append(ms)
            self.pending_events = []
            # apply action
            if act.get("pick") is not None and self.offer:
                self.offer = None
            mv = act.get("move", [0, 0])
            ml = math.hypot(*mv)
            if ml > 1:
                mv = [mv[0] / ml, mv[1] / ml]
            sp = self.cfg["mech"]["walk_speed"]
            if act.get("boost") and self.energy > 1 and ml > 0.1 and self.grounded:
                sp = self.cfg["mech"]["boost_speed"]
                self.energy -= self.cfg["mech"]["boost_drain"] * 0.1
            elif not act.get("boost"):
                self.energy = min(100, self.energy + 18 * 0.1)
            if not self.grounded:
                sp = min(sp, 9.0)
            dx, dy = mv[0] * sp * 0.1, mv[1] * sp * 0.1
            moved = True
            for _ in range(2):
                if ml > 0.05:
                    moved = self.try_move(dx / 2, dy / 2) and moved
                self.vertical(act, 0.05)
            self.vel = [dx * 10, dy * 10, 0] if moved else [0, 0, 0]
            # aim slew
            a = act.get("aim", self.aim)
            dyaw = (a[0] - self.aim[0] + math.pi) % math.tau - math.pi
            self.aim[0] = (self.aim[0] + max(-0.4, min(0.4, dyaw)) + math.pi) % math.tau - math.pi
            self.aim[1] += max(-0.3, min(0.3, a[1] - self.aim[1]))
            # take
            if act.get("take") is not None:
                for pk in list(self.pickups):
                    if pk["id"] == act["take"] and math.hypot(pk["pos"][0] - self.pos[0], pk["pos"][1] - self.pos[1]) < 4:
                        slot = act.get("drop", 1)
                        self.weapons[int(slot)] = {"kind": pk["kind"], "cooldown": 1.0, "charge": 0, "stats": dict(self.cfg["weapons"][pk["kind"]])}
                        self.pickups.remove(pk)
                        self.stats["taken"] += 1
            for cb in list(self.cubes_list):
                if math.hypot(cb["pos"][0] - self.pos[0], cb["pos"][1] - self.pos[1]) < 3.5:
                    self.cubes_list.remove(cb)
                    self.cubes += 1
                    self.maxhp *= 1.04
                    if self.cubes in self.thr and self.offer is None:
                        self.offer = {"id": self.cubes, "expires": t + 5, "choices": [{"id": u["id"], "name": u["name"], "desc": u["desc"]} for u in self.rng.sample(self.cfg["upgrades"][:9], 3)]}
            # our shots hit enemies (crudely)
            fire = act.get("fire", [False, False])
            if fire and fire[0]:
                for e in self.enemies:
                    to = math.atan2(e["pos"][1] - self.pos[1], e["pos"][0] - self.pos[0])
                    if abs((to - self.aim[0] + math.pi) % math.tau - math.pi) < 0.05:
                        e["hp"] -= 45 * 0.4 * 0.6
                        self.pending_events.append({"type": "hit", "target": e["id"], "amount": 45, "weapon": "gun"})
            self.enemies = [e for e in self.enemies if e["hp"] > 0]
            # enemies approach and shoot
            for e in self.enemies:
                d = math.hypot(e["pos"][0] - self.pos[0], e["pos"][1] - self.pos[1])
                ux, uy = (self.pos[0] - e["pos"][0]) / max(d, 1), (self.pos[1] - e["pos"][1]) / max(d, 1)
                if d > 50:
                    nx_, ny_ = e["pos"][0] + ux * 1.0, e["pos"][1] + uy * 1.0
                    if self.wm.is_free(nx_, ny_):
                        e["pos"] = [nx_, ny_, self.wm.surface(nx_, ny_)]
                        e["vel"] = [ux * 10, uy * 10, 0]
                    else:
                        side = (-uy, ux)
                        nx_, ny_ = e["pos"][0] + side[0], e["pos"][1] + side[1]
                        if self.wm.is_free(nx_, ny_):
                            e["pos"] = [nx_, ny_, self.wm.surface(nx_, ny_)]
                e["cd"] -= 0.1
                if e["cd"] <= 0 and d < 200 and self.wm.los(e["pos"][0], e["pos"][1], e["pos"][2] + 7, self.pos[0], self.pos[1], self.pos[2] + 4.5):
                    e["cd"] = 0.24
                    if self.rng.random() < 0.35:
                        to_e = math.atan2(e["pos"][1] - self.pos[1], e["pos"][0] - self.pos[0])
                        rear = abs((to_e - self.aim[0] + math.pi) % math.tau - math.pi) > math.pi / 2
                        amt = 45 * (1.5 if rear else 1)
                        self.hp -= amt
                        self.last_dmg = t
                        self.pending_events.append({"type": "damaged", "amount": amt, "weapon": "gun", "bearing": to_e, "by": e["id"], "rear": rear})
            # zone, regen
            if not in_zone:
                self.hp -= zobs["dps"] * 0.1
                self.stats["zone_dmg"] += zobs["dps"] * 0.1
                self.stats.setdefault("zph", {}).setdefault(self.phase, 0)
                self.stats["zph"][self.phase] += round(zobs["dps"] * 0.1)
                self.last_dmg = t
            elif t - self.last_dmg > 5:
                self.hp = min(self.maxhp, self.hp + 4.5)
            # stuck stat: commanded movement but no progress for 3 s
            if idle_pos is None or math.hypot(self.pos[0] - idle_pos[0], self.pos[1] - idle_pos[1]) > 3:
                idle_pos, idle_t = list(self.pos), t
            elif ml > 0.5 and t - idle_t > 3:
                self.stats["stuck_events"] += 1
                idle_t = t
            t += 0.1
            tick += 2
        ms = sorted(self.stats["ms"])
        return {"t": round(t, 1), "hp": round(self.hp), "zone_dmg": round(self.stats["zone_dmg"]), "stuck": self.stats["stuck_events"],
                "p50": round(ms[len(ms) // 2], 2), "p99": round(ms[int(len(ms) * 0.99)], 2), "max": round(ms[-1], 2),
                "weapons": [w and w["kind"] for w in self.weapons], "cubes": self.cubes, "taken": self.stats["taken"], "zph": self.stats.get("zph")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=10)
    ap.add_argument("--enemies", type=int, default=0)
    ap.add_argument("--kit", default=os.path.join(HERE, "..", ".."))
    ap.add_argument("--arena", default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    hello = load_hello(a.kit)
    arena = a.arena or os.path.join(a.kit, "arena")
    for m in range(a.matches):
        s = Sim(hello, arena, a.seed + m, a.enemies)
        print(m, s.run(), flush=True)


if __name__ == "__main__":
    main()
