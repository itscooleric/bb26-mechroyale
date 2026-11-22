"""Static map model for mechfight/1: terrain, prisms, line of sight and a walk grid.

Standard library only. Built once from the map files in hello.map.dir (protocol.md 11).
"""
from __future__ import annotations

import array
import heapq
import json
import math
import os
import sys

INF = float("inf")
CELL = 4.0          # walk-grid cell size, meters
CLEAR = 1.5         # extra clearance sampled around each cell (mech radius is 2.5)
BUCKET = 32.0       # spatial bucket for prism lookups
HEAD = 9.0          # mech height: overhangs higher than this can be walked under


class World:
    def __init__(self, map_dir: str, cfg: dict):
        mech = cfg.get("mech", {})
        self.step_up = float(mech.get("step_up", 1.2))
        self.max_slope = math.radians(float(mech.get("max_slope_deg", 50)))
        with open(os.path.join(map_dir, "v2", "world.json"), "rb") as f:
            wj = json.loads(f.read())
        t = wj["terrain"]
        self.tminx, self.tminy = float(t["minX"]), float(t["minY"])
        self.tstep = float(t.get("step", 1.0))
        self.tnx, self.tny = int(t["nx"]), int(t["ny"])
        scale = float(t.get("scale", 0.01))
        a = array.array("h")
        with open(os.path.join(map_dir, "v2", t.get("file", "terrain.i16")), "rb") as f:
            a.frombytes(f.read())
        if sys.byteorder == "big":
            a.byteswap()
        self.terr = [v * scale for v in a]

        b = wj["bounds"]
        self.minx, self.maxx = float(b["minX"]), float(b["maxX"])
        self.miny, self.maxy = float(b["minY"]), float(b["maxY"])

        # Prisms: (base, top, edges, bbox). Edges hold every ring (outer + holes) for even-odd tests.
        self.prisms = []
        self.buckets: dict = {}
        for p in wj.get("prisms", []):
            rings = [p["ring"]] + list(p.get("holes") or [])
            edges = []
            xs, ys = [], []
            for r in rings:
                n = len(r) // 2
                for k in range(n):
                    x1, y1 = r[2 * k], r[2 * k + 1]
                    x2, y2 = r[2 * ((k + 1) % n)], r[2 * ((k + 1) % n) + 1]
                    edges.append((x1, y1, x2, y2))
                    xs.append(x1)
                    ys.append(y1)
            if not edges:
                continue
            bbox = (min(xs), min(ys), max(xs), max(ys))
            idx = len(self.prisms)
            self.prisms.append((float(p["base"]), float(p["top"]), edges, bbox))
            for bi in range(int(math.floor(bbox[0] / BUCKET)), int(math.floor(bbox[2] / BUCKET)) + 1):
                for bj in range(int(math.floor(bbox[1] / BUCKET)), int(math.floor(bbox[3] / BUCKET)) + 1):
                    self.buckets.setdefault((bi, bj), []).append(idx)
        self._build_grid()

    # ---------- terrain and solids ----------

    def ground(self, x: float, y: float) -> float:
        fx = (x - self.tminx) / self.tstep
        fy = (y - self.tminy) / self.tstep
        i = min(max(int(fx), 0), self.tnx - 2)
        j = min(max(int(fy), 0), self.tny - 2)
        u = min(max(fx - i, 0.0), 1.0)
        v = min(max(fy - j, 0.0), 1.0)
        k = j * self.tnx + i
        t = self.terr
        return (t[k] * (1 - u) + t[k + 1] * u) * (1 - v) + (t[k + self.tnx] * (1 - u) + t[k + self.tnx + 1] * u) * v

    @staticmethod
    def _inside(edges, x: float, y: float) -> bool:
        c = False
        for x1, y1, x2, y2 in edges:
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                c = not c
        return c

    def solid(self, x: float, y: float, z: float) -> bool:
        """True if (x, y, z) is inside a building, stadium or bridge prism, or under the ground."""
        if z < self.ground(x, y) - 0.3:
            return True
        for idx in self.buckets.get((int(math.floor(x / BUCKET)), int(math.floor(y / BUCKET))), ()):
            base, top, edges, bb = self.prisms[idx]
            if base <= z <= top and bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3] and self._inside(edges, x, y):
                return True
        return False

    def los(self, a, b, step: float = 1.5) -> bool:
        """Line of sight between two 3D points (prisms and terrain block; trees don't stop shots)."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        n = int(d / step)
        for s in range(1, n):
            f = s / n
            if self.solid(a[0] + dx * f, a[1] + dy * f, a[2] + dz * f):
                return False
        return True

    # ---------- walk grid ----------

    def _build_grid(self) -> None:
        self.gx0, self.gy0 = self.minx, self.miny
        self.nx = int(math.ceil((self.maxx - self.minx) / CELL))
        self.ny = int(math.ceil((self.maxy - self.miny) / CELL))
        nx, ny = self.nx, self.ny
        base_terr = [0.0] * (nx * ny)
        for j in range(ny):
            yc = self.gy0 + (j + 0.5) * CELL
            for i in range(nx):
                base_terr[j * nx + i] = self.ground(self.gx0 + (i + 0.5) * CELL, yc)
        surf = list(base_terr)
        roof = bytearray(nx * ny)
        for base, top, edges, bb in self.prisms:
            j0 = max(0, int(math.floor((bb[1] - CLEAR - self.gy0) / CELL)))
            j1 = min(ny - 1, int(math.floor((bb[3] + CLEAR - self.gy0) / CELL)))
            for j in range(j0, j1 + 1):
                yc = self.gy0 + (j + 0.5) * CELL
                for yy in (yc - CLEAR, yc, yc + CLEAR):
                    xs = []
                    for x1, y1, x2, y2 in edges:
                        if (y1 > yy) != (y2 > yy):
                            xs.append(x1 + (yy - y1) * (x2 - x1) / (y2 - y1))
                    xs.sort()
                    for q in range(0, len(xs) - 1, 2):
                        xa, xb = xs[q] - CLEAR, xs[q + 1] + CLEAR
                        i0 = max(0, int(math.ceil((xa - self.gx0) / CELL - 0.5)))
                        i1 = min(nx - 1, int(math.floor((xb - self.gx0) / CELL - 0.5)))
                        for i in range(i0, i1 + 1):
                            k = j * nx + i
                            if base < base_terr[k] + HEAD and top > surf[k]:
                                surf[k] = top
                                roof[k] = 1
        self.surf = surf
        self.roof = roof
        tan = math.tan(self.max_slope) * 0.9
        nb = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            d = CELL * math.hypot(di, dj)
            nb.append((di, dj, d, d * tan))
        self.nb = nb

    def cell(self, x: float, y: float) -> int:
        i = min(max(int((x - self.gx0) / CELL), 0), self.nx - 1)
        j = min(max(int((y - self.gy0) / CELL), 0), self.ny - 1)
        return j * self.nx + i

    def center(self, k: int):
        j, i = divmod(k, self.nx)
        return (self.gx0 + (i + 0.5) * CELL, self.gy0 + (j + 0.5) * CELL)

    def can_step(self, u: int, v: int, slope_rise: float) -> bool:
        """Can a mech walk from cell u into neighbouring cell v?"""
        dh = self.surf[v] - self.surf[u]
        if self.roof[u] or self.roof[v]:
            return dh <= self.step_up
        return dh <= max(self.step_up, slope_rise)

    def field(self, cx: float, cy: float, radius: float):
        """Generator: reverse Dijkstra from every cell within `radius` of (cx, cy).

        Yields None while working (so the caller can spread it over several decisions),
        then yields the finished distance list (meters of walking to the region)."""
        nx, ny = self.nx, self.ny
        dist = [INF] * (nx * ny)
        heap = []
        r = max(radius, CELL * 0.75)
        cj0 = max(0, int((cy - r - self.gy0) / CELL))
        cj1 = min(ny - 1, int((cy + r - self.gy0) / CELL))
        ci0 = max(0, int((cx - r - self.gx0) / CELL))
        ci1 = min(nx - 1, int((cx + r - self.gx0) / CELL))
        for j in range(cj0, cj1 + 1):
            for i in range(ci0, ci1 + 1):
                k = j * nx + i
                x, y = self.center(k)
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    dist[k] = 0.0
                    heap.append((0.0, k))
        if not heap:
            k = self.cell(cx, cy)
            dist[k] = 0.0
            heap.append((0.0, k))
        heapq.heapify(heap)
        pops = 0
        surf, roof, step_up, nb = self.surf, self.roof, self.step_up, self.nb
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist[v]:
                continue
            j, i = divmod(v, nx)
            sv, rv = surf[v], roof[v]
            for di, dj, cost, rise in nb:
                ii, jj = i + di, j + dj
                if ii < 0 or jj < 0 or ii >= nx or jj >= ny:
                    continue
                u = jj * nx + ii
                dh = sv - surf[u]
                if dh > (step_up if (rv or roof[u]) else max(step_up, rise)):
                    continue
                nd = d + cost
                if nd < dist[u]:
                    dist[u] = nd
                    heapq.heappush(heap, (nd, u))
            pops += 1
            if pops % 400 == 0:
                yield None
        yield dist

    def descend(self, dist, x: float, y: float, look: int = 3):
        """Direction (unit vector) to follow `dist` downhill from (x, y), or None at the goal."""
        k = self.cell(x, y)
        if dist[k] == 0.0:
            return None
        if dist[k] == INF:
            # Off the grid (a roof, or a cell the rasterizer filled): head for the nearest reachable cell.
            best, bd = None, INF
            j0, i0 = divmod(k, self.nx)
            for dj in range(-4, 5):
                for di in range(-4, 5):
                    jj, ii = j0 + dj, i0 + di
                    if 0 <= ii < self.nx and 0 <= jj < self.ny:
                        u = jj * self.nx + ii
                        if dist[u] < INF:
                            score = dist[u] + CELL * math.hypot(di, dj) * 3
                            if score < bd:
                                best, bd = u, score
            if best is None:
                return None
            tx, ty = self.center(best)
        else:
            cur = k
            for _ in range(look):
                j, i = divmod(cur, self.nx)
                nxt, nd = cur, dist[cur]
                for di, dj, _c, rise in self.nb:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < self.nx and 0 <= jj < self.ny:
                        u = jj * self.nx + ii
                        if dist[u] < nd and self.can_step(cur, u, rise):
                            nxt, nd = u, dist[u]
                if nxt == cur:
                    break
                cur = nxt
                if dist[cur] == 0.0:
                    break
            if cur == k:
                return None
            tx, ty = self.center(cur)
        dx, dy = tx - x, ty - y
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return None
        return (dx / n, dy / n)

    def cover_spots(self, dist, x: float, y: float, radius: float = 36.0):
        """Ground cells in the goal region (dist == 0) next to a tall wall, nearest first.

        Returns [(walk_estimate, cell, open_yaw)], open_yaw pointing away from the wall."""
        out = []
        j0, i0 = divmod(self.cell(x, y), self.nx)
        rc = int(radius / CELL)
        for dj in range(-rc, rc + 1):
            for di in range(-rc, rc + 1):
                jj, ii = j0 + dj, i0 + di
                if not (1 <= ii < self.nx - 1 and 1 <= jj < self.ny - 1):
                    continue
                k = jj * self.nx + ii
                if dist[k] != 0.0 or self.roof[k]:
                    continue
                wx = wy = 0.0
                walls = 0
                for ddi, ddj, _c, _r in self.nb:
                    u = (jj + ddj) * self.nx + ii + ddi
                    if self.surf[u] - self.surf[k] > 6.0:
                        walls += 1
                        wx += ddi
                        wy += ddj
                if walls == 0 or walls > 5:
                    continue
                cx, cy = self.center(k)
                out.append((math.hypot(cx - x, cy - y), k, math.atan2(-wy, -wx)))
        out.sort()
        return out
