"""Static map model: terrain, solid prisms rasterized to a 2 m grid, walkability, line of sight, A*.

Pure standard library. Built once at `hello` (a couple of seconds), then queried every decision.
"""
from __future__ import annotations

import array
import heapq
import json
import math
import os
import sys
import time

CELL = 2.0            # nav grid resolution, meters
SQ2 = math.sqrt(2.0)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class WorldMap:
    def __init__(self, map_dir: str, mech_cfg: dict):
        self.ok = False
        self.mech_h = float(mech_cfg.get("height", 9))
        self.step_up = float(mech_cfg.get("step_up", 1.2))
        self.max_slope = math.tan(math.radians(float(mech_cfg.get("max_slope_deg", 50))))
        try:
            self._load(map_dir)
            self.ok = True
        except Exception as e:  # no map: the bot still plays, just without navigation
            log("worldmap: failed to load map:", repr(e))
            self._fallback()

    # ------------------------------------------------------------------ loading
    def _fallback(self):
        self.X0, self.Y0, self.NX, self.NY = -520.0, -340.0, 540, 370
        n = self.NX * self.NY
        self.ground = array.array("f", [0.0]) * n
        self.surf = array.array("f", [0.0]) * n
        self.btop = array.array("f", [-1e4]) * n
        self.bbase = array.array("f", [1e4]) * n
        self.otop = array.array("f", [-1e4]) * n
        self.obase = array.array("f", [1e4]) * n
        self.solid = bytearray(n)
        self.free = bytearray(b"\x01") * n
        self.pen = bytearray(n)
        self.hgt = array.array("f", [0.0]) * n
        self.comp = array.array("i", [1]) * n
        self.tx0, self.ty0, self.tnx, self.tny, self.tstep = -513.0, -334.0, 1, 1, 1.0
        self.terr = array.array("h", [0])
        self.tscale = 0.01
        self.nb = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                   (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]
        self.comp_size = [0, n]
        self._cover_cache = {}

    def _load(self, map_dir):
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            w = json.load(f)
        tr = w["terrain"]
        self.tx0, self.ty0 = float(tr["minX"]), float(tr["minY"])
        self.tnx, self.tny, self.tstep = int(tr["nx"]), int(tr["ny"]), float(tr.get("step", 1.0))
        scale = float(tr.get("scale", 0.01))
        terr = array.array("h")
        with open(os.path.join(map_dir, "v2", tr.get("file", "terrain.i16")), "rb") as f:
            terr.frombytes(f.read())
        if sys.byteorder != "little":
            terr.byteswap()
        self.terr = terr
        self.tscale = scale
        b = w["bounds"]
        self.bminx, self.bmaxx, self.bminy, self.bmaxy = b["minX"], b["maxX"], b["minY"], b["maxY"]
        self.X0, self.Y0 = self.tx0, self.ty0
        self.NX = int(math.ceil((self.tx0 + (self.tnx - 1) * self.tstep - self.X0) / CELL))
        self.NY = int(math.ceil((self.ty0 + (self.tny - 1) * self.tstep - self.Y0) / CELL))
        NX, NY = self.NX, self.NY
        n = NX * NY
        # ground height at each cell center
        ground = array.array("f", bytes(4 * n))
        th = self.terrain_h
        for j in range(NY):
            y = self.Y0 + (j + 0.5) * CELL
            row = j * NX
            for i in range(NX):
                ground[row + i] = th(self.X0 + (i + 0.5) * CELL, y)
        self.ground = ground
        surf = array.array("f", ground)
        btop = array.array("f", [-1e4]) * n     # ground-based solids: [bbase, btop]
        bbase = array.array("f", [1e4]) * n
        otop = array.array("f", [-1e4]) * n     # overpasses (bridges you walk under)
        obase = array.array("f", [1e4]) * n
        solid = bytearray(n)
        H, step = self.mech_h, self.step_up
        for p in w["prisms"]:
            base, top = float(p["base"]), float(p["top"])
            rings = [p["ring"]] + list(p.get("holes") or [])
            edges = []
            mny, mxy = 1e9, -1e9
            for r in rings:
                m = len(r) // 2
                for k in range(m):
                    x1, y1 = r[2 * k], r[2 * k + 1]
                    x2, y2 = r[2 * ((k + 1) % m)], r[2 * ((k + 1) % m) + 1]
                    if y1 != y2:
                        edges.append((x1, y1, x2, y2))
                    mny = min(mny, y1)
                    mxy = max(mxy, y1)
            j0 = max(0, int((mny - self.Y0) / CELL))
            j1 = min(NY - 1, int((mxy - self.Y0) / CELL))
            for j in range(j0, j1 + 1):
                cells = set()
                for yy in (self.Y0 + j * CELL + 0.3, self.Y0 + (j + 0.5) * CELL, self.Y0 + (j + 1) * CELL - 0.3):
                    xs = []
                    for (x1, y1, x2, y2) in edges:
                        if (y1 <= yy < y2) or (y2 <= yy < y1):
                            xs.append(x1 + (yy - y1) * (x2 - x1) / (y2 - y1))
                    xs.sort()
                    for k in range(0, len(xs) - 1, 2):
                        i0 = max(0, int((xs[k] - self.X0) / CELL))
                        i1 = min(NX - 1, int((xs[k + 1] - self.X0) / CELL))
                        cells.update(range(i0, i1 + 1))
                row = j * NX
                for i in cells:
                    idx = row + i
                    g = ground[idx]
                    if base > g + H + 0.5:          # overpass: walk under it
                        if top > otop[idx]:
                            otop[idx] = top
                        if base < obase[idx]:
                            obase[idx] = base
                        continue
                    if top > btop[idx]:
                        btop[idx] = top
                    if base < bbase[idx]:
                        bbase[idx] = base
                    if top <= g + step:              # low platform: walkable surface
                        if top > surf[idx]:
                            surf[idx] = top
                    else:
                        solid[idx] = 1
        self.surf, self.btop, self.bbase, self.otop, self.obase, self.solid = surf, btop, bbase, otop, obase, solid
        # map edge is a wall
        mi0 = max(0, int((self.bminx - self.X0) / CELL) + 1)
        mi1 = min(NX - 1, int((self.bmaxx - self.X0) / CELL) - 1)
        mj0 = max(0, int((self.bminy - self.Y0) / CELL) + 1)
        mj1 = min(NY - 1, int((self.bmaxy - self.Y0) / CELL) - 1)
        # free = walkable for a mech center: not solid, not adjacent to solid (radius 2.5 m)
        free = bytearray(n)
        pen = bytearray(n)
        for j in range(mj0 + 1, mj1):
            row = j * NX
            for i in range(mi0 + 1, mi1):
                idx = row + i
                if solid[idx] or solid[idx - 1] or solid[idx + 1] or solid[idx - NX] or solid[idx + NX] \
                        or solid[idx - NX - 1] or solid[idx - NX + 1] or solid[idx + NX - 1] or solid[idx + NX + 1]:
                    continue
                free[idx] = 1
        # soft penalty two cells from a wall (prefer clearance)
        for j in range(mj0 + 2, mj1 - 1):
            row = j * NX
            for i in range(mi0 + 2, mi1 - 1):
                idx = row + i
                if free[idx] and (not free[idx - 2] or not free[idx + 2] or not free[idx - 2 * NX] or not free[idx + 2 * NX]
                                  or not free[idx - NX - 1] or not free[idx + NX + 1] or not free[idx - NX + 1] or not free[idx + NX - 1]):
                    pen[idx] = 1
        self.free, self.pen = free, pen
        # hgt: what you stand on in a cell (roof top for solids). Used for jet-over-wall links in A*.
        hgt = array.array("f", surf)
        for k in range(n):
            if solid[k] and btop[k] > hgt[k]:
                hgt[k] = btop[k]
        self.hgt = hgt
        self._components()
        self.nb = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                   (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]
        self._cover_cache = {}

    def _components(self):
        """Label free cells connected by walking (no jets). comp 0 = not free."""
        from collections import deque
        NX, NY = self.NX, self.NY
        n = NX * NY
        free, surf = self.free, self.surf
        comp = array.array("i", bytes(4 * n))
        lim = max(self.step_up, self.max_slope * CELL)
        limd = max(self.step_up, self.max_slope * CELL * SQ2)
        nbs = ((1, lim), (-1, lim), (NX, lim), (-NX, lim), (NX + 1, limd), (NX - 1, limd), (-NX + 1, limd), (-NX - 1, limd))
        label = 0
        sizes = [0]
        for start in range(NX, n - NX):
            if not free[start] or comp[start]:
                continue
            label += 1
            comp[start] = label
            q = deque([start])
            size = 0
            while q:
                c = q.popleft()
                size += 1
                sc = surf[c]
                for d, lm in nbs:
                    m = c + d
                    if free[m] and not comp[m]:
                        # undirected approximation: connected if the step is walkable at least one way
                        if abs(surf[m] - sc) <= lm or surf[m] < sc:
                            comp[m] = label
                            q.append(m)
            sizes.append(size)
        self.comp = comp
        self.comp_size = sizes

    def comp_at(self, x, y):
        return self.comp[self.idx(x, y)]

    # ------------------------------------------------------------------ queries
    def terrain_h(self, x, y):
        fx = (x - self.tx0) / self.tstep
        fy = (y - self.ty0) / self.tstep
        i = int(fx)
        j = int(fy)
        if i < 0 or j < 0 or i >= self.tnx - 1 or j >= self.tny - 1:
            i = min(max(i, 0), self.tnx - 1)
            j = min(max(j, 0), self.tny - 1)
            return self.terr[j * self.tnx + i] * self.tscale
        ax = fx - i
        ay = fy - j
        t = self.terr
        k = j * self.tnx + i
        h00, h10, h01, h11 = t[k], t[k + 1], t[k + self.tnx], t[k + self.tnx + 1]
        return ((h00 * (1 - ax) + h10 * ax) * (1 - ay) + (h01 * (1 - ax) + h11 * ax) * ay) * self.tscale

    def cell(self, x, y):
        i = int((x - self.X0) / CELL)
        j = int((y - self.Y0) / CELL)
        if i < 0:
            i = 0
        elif i >= self.NX:
            i = self.NX - 1
        if j < 0:
            j = 0
        elif j >= self.NY:
            j = self.NY - 1
        return i, j

    def idx(self, x, y):
        i, j = self.cell(x, y)
        return j * self.NX + i

    def center(self, idx):
        j, i = divmod(idx, self.NX)
        return self.X0 + (i + 0.5) * CELL, self.Y0 + (j + 0.5) * CELL

    def is_free(self, x, y):
        return bool(self.free[self.idx(x, y)])

    def surface(self, x, y):
        return self.surf[self.idx(x, y)]

    def nearest_free(self, x, y, max_r=12):
        ci, cj = self.cell(x, y)
        NX, NY, free = self.NX, self.NY, self.free
        if free[cj * NX + ci]:
            return cj * NX + ci
        for r in range(1, max_r + 1):
            best, bd = None, 1e9
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    i, j = ci + di, cj + dj
                    if 0 <= i < NX and 0 <= j < NY and free[j * NX + i]:
                        d = di * di + dj * dj
                        if d < bd:
                            bd, best = d, j * NX + i
            if best is not None:
                return best
        return None

    def los(self, x0, y0, z0, x1, y1, z1, step=2.0):
        """True if the segment clears buildings and terrain (trees ignored: they don't stop shots)."""
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        d = math.hypot(dx, dy)
        n = int(d / step)
        if n < 2:
            return True
        X0, Y0, NX, NY = self.X0, self.Y0, self.NX, self.NY
        bt, bb, ot, ob, gr = self.btop, self.bbase, self.otop, self.obase, self.ground
        inv = 1.0 / n
        for k in range(1, n):
            f = k * inv
            i = int((x0 + dx * f - X0) * 0.5)
            j = int((y0 + dy * f - Y0) * 0.5)
            if i < 0 or j < 0 or i >= NX or j >= NY:
                return False
            idx = j * NX + i
            z = z0 + dz * f
            if z < gr[idx] - 0.3:
                return False
            if bb[idx] <= z <= bt[idx] or ob[idx] <= z <= ot[idx]:
                return False
        return True

    def walk_line(self, x0, y0, x1, y1):
        """True if a straight walk stays on free cells (and no steep climb)."""
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        n = int(d / 1.5) + 1
        free, surf = self.free, self.surf
        prev = self.idx(x0, y0)
        ps = surf[prev]
        for k in range(1, n + 1):
            f = k / n
            idx = self.idx(x0 + dx * f, y0 + dy * f)
            if idx == prev:
                continue
            if not free[idx]:
                return False
            s = surf[idx]
            if s - ps > max(self.step_up, self.max_slope * 1.5):
                return False
            prev, ps = idx, s
        return True

    def cover(self, idx):
        """0..8: how many of 8 directions have a tall solid within ~8 m of this cell."""
        c = self._cover_cache.get(idx)
        if c is not None:
            return c
        NX, NY = self.NX, self.NY
        j, i = divmod(idx, NX)
        g = self.ground[idx]
        c = 0
        for (di, dj, _) in self.nb:
            for s in (2, 3, 4):
                ii, jj = i + di * s, j + dj * s
                if 0 <= ii < NX and 0 <= jj < NY:
                    k = jj * NX + ii
                    if self.solid[k] and self.btop[k] > g + 8:
                        c += 1
                        break
        self._cover_cache[idx] = c
        return c


class PathJob:
    """Resumable weighted A* on the free grid. Call step(budget) each decision until done."""

    def __init__(self, wm: WorldMap, start_idx, goal_idx, weight=1.4, max_total=60000, avoid=None, climb=14.0):
        self.wm = wm
        self.climb = climb
        self.start = start_idx
        self.goal = goal_idx
        self.w = weight
        self.done = False
        self.path = None
        self.total = 0
        self.max_total = max_total
        self.avoid = avoid  # function(idx)->extra cost, optional
        NX = wm.NX
        self.gj, self.gi = divmod(goal_idx, NX)
        self.g = {start_idx: 0.0}
        self.parent = {start_idx: -1}
        self.open = [(self._h(start_idx), 0.0, start_idx)]
        self.closed = set()
        self.best = (self._h(start_idx), start_idx)

    def _h(self, idx):
        j, i = divmod(idx, self.wm.NX)
        dx, dy = abs(i - self.gi), abs(j - self.gj)
        return self.w * ((dx + dy) + (SQ2 - 2) * min(dx, dy))

    def step(self, budget=1200, deadline=None):
        """Expand up to `budget` nodes, stopping early at perf_counter() >= deadline."""
        if self.done:
            return True
        clock = time.perf_counter
        wm = self.wm
        NX, NY = wm.NX, wm.NY
        free, surf, pen, hgt, solid = wm.free, wm.surf, wm.pen, wm.hgt, wm.solid
        climb = self.climb
        lim = max(wm.step_up, wm.max_slope * CELL)
        limd = max(wm.step_up, wm.max_slope * CELL * SQ2)
        g, parent, openl, closed = self.g, self.parent, self.open, self.closed
        goal = self.goal
        avoid = self.avoid
        push, pop = heapq.heappush, heapq.heappop
        nbs = ((1, 1.0), (-1, 1.0), (NX, 1.0), (-NX, 1.0), (NX + 1, SQ2), (NX - 1, SQ2), (-NX + 1, SQ2), (-NX - 1, SQ2))
        n = 0
        while openl and n < budget:
            # Check on entry too: another planner may have spent this tick's budget.
            if deadline is not None and (n & 31) == 0 and clock() >= deadline:
                break
            f, gc, cur = pop(openl)
            if cur in closed:
                continue
            closed.add(cur)
            n += 1
            if cur == goal:
                self._finish(cur)
                return True
            h = f - gc
            if h < self.best[0]:
                self.best = (h, cur)
            ci = cur % NX
            if ci == 0 or ci == NX - 1 or cur < NX or cur >= NX * (NY - 1):
                continue
            h0 = hgt[cur]
            for (d, c) in nbs:
                nx_ = cur + d
                if nx_ in closed:
                    continue
                rise = hgt[nx_] - h0
                if free[nx_]:
                    if rise > (lim if c == 1.0 else limd):
                        if rise > climb:
                            continue
                        cost = c + 14.0 + rise * 2.0     # jump + jet up a steep step
                    else:
                        cost = c + (0.6 if pen[nx_] else 0.0)
                else:
                    if rise > climb:
                        continue
                    cost = c + (6.0 if solid[nx_] else 2.0) + (14.0 + rise * 2.0 if rise > 1.2 else 0.0)   # climb over a wall / roof
                if avoid is not None:
                    cost += avoid(nx_)
                ng = gc + cost
                if ng < g.get(nx_, 1e18):
                    g[nx_] = ng
                    parent[nx_] = cur
                    push(openl, (ng + self._h(nx_), ng, nx_))
        self.total += n
        if not openl or self.total >= self.max_total:
            self._finish(self.best[1])  # partial path toward the closest reached cell
            return True
        return False

    def _finish(self, end):
        p = []
        cur = end
        while cur != -1:
            p.append(cur)
            cur = self.parent[cur]
        p.reverse()
        self.path = p
        self.done = True
