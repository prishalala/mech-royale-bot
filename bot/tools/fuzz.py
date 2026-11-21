"""Fuzz act() with the recorded stream plus injected situations; fails loudly on any exception.

    python tools/fuzz.py
"""
import copy, gzip, json, math, os, random, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import bot as B  # noqa: E402

KIT = os.path.join(HERE, "..", "..")
msgs = [json.loads(l) for l in gzip.open(os.path.join(KIT, "check", "stream.jsonl.gz"), "rt") if l.strip()]
hello = msgs[0]
hello["map"]["dir"] = os.path.join(KIT, "arena")
obs_list = [m for m in msgs if m.get("type") == "obs"]
W = hello["config"]["weapons"]
rng = random.Random(7)


def weapon(kind):
    w = {"kind": kind, "cooldown": rng.choice([0, 0, 0.5]), "stats": dict(W[kind])}
    if kind == "gun":
        w.update(ammo=rng.randint(0, 12), reload=rng.choice([0, 0, 1.2]))
    if kind == "laser":
        w["charge"] = rng.choice([0, 0, 0.3, 0.75, 1.5, 2.6])
    return w


def mutate(o):
    o = copy.deepcopy(o)
    me = o["self"]
    p = me["pos"]
    second = rng.choice([None, "laser", "mortar", "lightning"])
    me["weapons"] = [weapon("gun"), weapon(second) if second else None]
    if rng.random() < 0.15:
        me["weapons"] = [None, weapon(rng.choice(["laser", "mortar", "lightning"]))]
    me["hp"] = rng.uniform(1, me["max_hp"])
    me["energy"] = rng.uniform(0, 100)
    me["grounded"] = rng.random() < 0.8
    me["in_zone"] = rng.random() < 0.8
    for _ in range(rng.randint(0, 4)):
        a = rng.random() * math.tau
        d = rng.uniform(5, 260)
        mid = rng.randint(0, 15)
        o["mechs"].append({"id": mid, "kind": "mech", "pos": [p[0] + d * math.cos(a), p[1] + d * math.sin(a), p[2] + rng.uniform(-3, 20)],
                           "vel": [rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(-5, 5)], "aim": [rng.uniform(-3.1, 3.1), 0],
                           "hp": rng.uniform(1, 1400), "max_hp": 1400, "cubes": rng.randint(0, 12),
                           "weapons": rng.sample(["gun", "laser", "mortar", "lightning"], rng.randint(0, 2)), "charging": rng.random() < 0.3})
    if rng.random() < 0.3:
        o["mechs"].append({"id": 100 + rng.randint(0, 11), "kind": "npc", "pos": [p[0] + 30, p[1], p[2]], "vel": [0, 0, 0], "aim": [0, 0], "hp": 350, "max_hp": 350})
        o["events"].append({"type": "damaged", "amount": 25, "weapon": "npc", "bearing": 0.0, "by": o["mechs"][-1]["id"]})
    if rng.random() < 0.3:
        o["projectiles"].append({"id": 9, "kind": "mortar", "pos": [p[0] + 30, p[1], p[2] + 30], "vel": [-20, 0, 5]})
    if rng.random() < 0.2:
        o["drops"].append({"id": 20000, "pos": [p[0] + 50, p[1] + 10, p[2]], "radius": 8, "lands_in": rng.choice([0, 10]), "capture_s": 4, "progress": 0, "holder": rng.choice([None, 3, me["id"]])})
    if rng.random() < 0.2:
        o["offer"] = {"id": 1, "expires": o["t"] + 5, "choices": [{"id": u, "name": u, "desc": ""} for u in rng.sample(["armor", "plating", "gun_rate", "laser_charge", "mystery_new", "sensors"], 3)]}
    if rng.random() < 0.2:
        o["pickups"].append({"id": 8001, "kind": rng.choice(["laser", "mortar", "lightning", "gun"]), "pos": [p[0] + 1, p[1] + 1, p[2]]})
    if rng.random() < 0.2:
        o["crates"].append({"id": 1001, "pos": [p[0] + 20, p[1], p[2]], "hp": 300})
        o["cubes"].append({"id": 5001, "pos": [p[0] - 10, p[1], p[2]]})
    if rng.random() < 0.3:
        o["events"].append({"type": "damaged", "amount": 45, "weapon": "gun", "bearing": rng.uniform(-3, 3), "by": rng.randint(0, 15), "rear": True})
    if rng.random() < 0.1:
        o["zone"].update(next_radius=0, radius=25, shrink_start=o["t"] - 1, shrink_end=o["t"] + 20)
    return o


bot = B.Bot(hello)
times, fails = [], 0
for rep in range(8):
    for o in obs_list:
        m = mutate(o)
        t0 = time.perf_counter()
        try:
            a = bot.act(m)
            json.dumps(a)
            assert all(isinstance(v, bool) for v in a["fire"]) and len(a["fire"]) == 2
            assert all(math.isfinite(v) for v in a["move"] + a["aim"])
        except Exception:
            fails += 1
            if fails <= 3:
                traceback.print_exc()
        times.append((time.perf_counter() - t0) * 1000)
times.sort()
print(f"fuzz: {len(times)} decisions, {fails} failures, p50 {times[len(times)//2]:.2f} ms, p99 {times[int(len(times)*0.99)]:.2f} ms, max {times[-1]:.2f} ms")
sys.exit(1 if fails else 0)
