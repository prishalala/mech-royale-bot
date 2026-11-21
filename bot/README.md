# ironwall-survivor — Mech Royale bot

A survival-first autonomous mech for the Buildathon 2026 Mech Royale track (`mechfight/1` protocol).
Pure standard-library Python 3: no dependencies, no network, writes nothing to disk.

## Layout

| File | Purpose |
| --- | --- |
| `Dockerfile` | Submission image (`python:3.13-slim`, entrypoint `bot.py`). |
| `bot.py` | Decision loop: memory, threat model, modes, combat, loot, upgrades. |
| `worldmap.py` | Map model from `/arena/v2`: 2 m grid of terrain + rasterized prisms, walkability, connected regions, line of sight, cover, resumable A* with jet-climb links. |
| `mechfight.py` | Kit wrapper (stdin/stdout loop, newest-observation rule, crash-safe coasting). |
| `tools/sim.py` | Crude single-bot kinematic simulator (zone, walls, jets, scripted enemies) for smoke tests. Not the real game. |
| `tools/fuzz.py` | Replays the kit's recorded stream with injected situations; fails on any exception. |

## Strategy

Score is placement, so every decision optimizes for staying alive:

- **Zone:** predicts the shrinking circle, estimates travel time, and leaves early. It camps at a spot inside the next circle (on our side of it) that has nearby building cover, is in our own walkable region, and is also inside the current circle.
- **Navigation:** the map is rasterized once at `hello` (~0.7 s). A* runs a few ms per decision (time-capped) with string-pulled path following. Walls up to ~45 m can be crossed with jump + jet (e.g. out of the stadium bowl). Stuck detection compares actual vs. commanded velocity.
- **Threat model:** for each known mech it tracks distance, weapons, HP, whether its torso aims at us, and whether it hit us recently. It also remembers damage bearings and combat sounds.
- **Modes:** `ZONE` > `RETREAT` (out-gunned by time-to-kill, low HP, or 2+ attackers → go to a cell with no line of sight from the threats, still facing and shooting them) > `FIGHT` (strafe at the weapon's preferred range, flip direction at random intervals) > `RECOVER` (hide to regen 45 HP/s) > `LOOT` > `HOLD` (stand still: silent, sweep the torso to scan).
- **Combat:** lead-aimed gun with LOS and torso-error gating; laser charge/release when the actual torso aim is on target; lightning cone/range/energy checks; ballistic mortar (low arc with LOS, lob over cover otherwise). It holds fire at far, unaware enemies to stay hidden. It faces attackers (rear hits deal ×1.5).
- **Dodging:** predicts mortar landing points and steps out of the splash; sidesteps a charging laser aimed at us.
- **Loot:** prefers lightning/laser over mortar as the second weapon (swaps mortar out). It picks up nearby cubes, shoots crates and NPC drones only when no mechs are around, and contests supply drops only when nobody is near them.
- **Upgrades:** armor > plating > nanites > weapon upgrades for held weapons; unknown ids are scored from their `effects`.

## Test

From the kit folder (this repo checked out as `mechfight-kit/bot`):

```sh
python3 check/check.py -- python3 -u bot/bot.py
docker build -t my-bot bot
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" my-bot
python3 bot/tools/fuzz.py
python3 bot/tools/sim.py --matches 10 --enemies 0
```
