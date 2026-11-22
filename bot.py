"""ericiscool: a survival-first Mech Royale bot for mechfight/1.

Placement is all that scores (16 - place), so the bot plays to outlast, not to hunt:
- walks the shortest path into the next zone circle (a walk grid built from the map),
- settles beside a wall inside it, facing the open side, and keeps quiet,
- turns to face whatever hits it or makes weapon noise (rear hits do 1.5x),
- fights back with a led gun and, if it finds one, a laser or lightning gun,
- backs off when hurt, grabs supply drops it can reach first, never chases.

Standard library only. Uses the kit's mechfight.py wrapper for the stdin/stdout loop.
"""
from __future__ import annotations

import math
import random
import time

import mechfight as mf
from world import INF, World

TAU = 2 * math.pi
NOISY = {"gun", "mortar", "explosion", "laser", "laser_charge", "lightning", "jet", "boost", "step"}
SLICE_S = 0.012                       # time per decision spent on path planning
UPGRADE_ORDER = ["armor", "plating", "nanites", "laser_damage", "laser_charge", "lightning_efficiency",
                 "gun_damage", "gun_rate", "sensors", "gyros", "lightning_range", "gun_mag",
                 "gun_velocity", "capacitor", "reactor", "servos", "jets", "lightning_chain"]
WANT = ("laser", "lightning")         # secondary weapons worth picking up (mortar makes lasers cook us off)


def wrap(a: float) -> float:
    return (a + math.pi) % TAU - math.pi


class Bot:
    name = "ericiscool"

    def __init__(self, hello: mf.Hello):
        self.cfg = hello["config"]
        self.me_id = hello.get("slot", -1)
        self.rng = random.Random(hello.get("seed", 0))
        self.world = None
        try:
            self.world = World(hello["map"]["dir"], self.cfg)
        except Exception as e:  # play on without a map rather than crash
            mf.log(f"map load failed: {e!r}")
        self.dist = None            # finished distance field to the goal region
        self.field_key = None
        self.field_job = None
        self.job_key = None
        self.hold = None            # (cell, open_yaw) we settle at inside the goal region
        self.hold_key = None
        self.bad_spots = set()
        self.dmg = []               # (t, bearing, by)
        self.heard = []             # (t, bearing, dist)
        self.trail = []             # (t, x, y)
        self.unstick_until = -1.0
        self.move_t = 0.0             # last time we were not trying to move
        self.unstick_dir = (1.0, 0.0)
        self.strafe = 1
        self.strafe_t = 0.0
        self.scan0 = self.rng.uniform(-math.pi, math.pi)
        self.bad_pickups = {}
        self.goal = (0.0, 0.0, 1e9)

    # ------------------------------------------------------------ helpers

    def _plan(self, zone, t0: float) -> None:
        cx, cy = zone.get("next_center") or zone.get("center")
        nr = float(zone.get("next_radius", zone.get("radius", 0)))
        r = max(nr - 40.0, nr * 0.6)
        self.goal = (cx, cy, r)
        w = self.world
        if w is None:
            return
        key = (round(cx, 1), round(cy, 1), round(r, 1))
        if key != self.field_key and key != self.job_key:
            self.field_job = w.field(cx, cy, r)
            self.job_key = key
        if self.field_job is not None:
            while time.perf_counter() - t0 < SLICE_S:
                out = next(self.field_job)
                if out is not None:
                    self.dist, self.field_key = out, self.job_key
                    self.field_job, self.job_key = None, None
                    self.hold, self.hold_key, self.bad_spots = None, None, set()
                    break

    def _lead(self, me, tgt, speed: float):
        x, y, z = me["pos"]
        mz = z + me["stats"].get("muzzle_height", 7)
        tx, ty, tz = tgt["pos"]
        vx, vy, vz = tgt.get("vel", (0, 0, 0))
        px, py, pz = tx, ty, tz
        for _ in range(2):
            d = math.dist((px, py, pz + 4.5), (x, y, mz))
            tf = d / speed if speed > 0 else 0.0
            px, py, pz = tx + vx * tf, ty + vy * tf, tz + vz * tf * 0.5
        dh = math.hypot(px - x, py - y)
        return math.atan2(py - y, px - x), math.atan2(pz + 4.5 - mz, max(dh, 1e-3)), dh

    def _move_toward(self, x, y, tx, ty):
        dx, dy = tx - x, ty - y
        n = math.hypot(dx, dy)
        k = min(1.0, n / 3.0)         # slow down on arrival instead of overshooting
        return (dx / n * k, dy / n * k) if n > 1e-6 else (0.0, 0.0)

    def _path_dir(self, x, y):
        if self.dist is None:
            gx, gy, _ = self.goal
            return self._move_toward(x, y, gx, gy)
        d = self.world.descend(self.dist, x, y)
        if d is None and not self._in_goal(x, y):
            gx, gy, _ = self.goal
            return self._move_toward(x, y, gx, gy)
        return d if d is not None else (0.0, 0.0)

    def _safe(self, zone, px, py, margin: float = 8.0) -> bool:
        """Is (px, py) comfortably inside both the current circle and the one it is closing to?"""
        for c, r in ((zone.get("center"), zone.get("radius")), (zone.get("next_center"), zone.get("next_radius"))):
            if c is not None and r is not None and math.hypot(px - c[0], py - c[1]) > r - margin:
                return False
        return True

    def _in_goal(self, x, y) -> bool:
        gx, gy, r = self.goal
        return math.hypot(x - gx, y - gy) <= r

    # ------------------------------------------------------------ main

    def act(self, obs: mf.Observation) -> mf.Action:
        t0 = time.perf_counter()
        me = obs["self"]
        t = obs["t"]
        x, y, z = me["pos"]
        w = self.world
        zone = obs["zone"]
        self._plan(zone, t0)
        act: mf.Action = {"tick": obs["tick"]}

        # ---- memory: damage, sounds, our own trail
        for e in obs.get("events", []):
            if e.get("type") == "damaged" and e.get("bearing") is not None:
                self.dmg.append((t, e["bearing"], e.get("by")))
            elif e.get("type") == "take_failed":
                self.bad_pickups[e.get("id")] = t
        for s in obs.get("sounds", []):
            if s.get("kind") in NOISY and s.get("dist") in ("near", "mid"):
                self.heard.append((t - s.get("age", 0), s["bearing"], s["dist"]))
        self.dmg = [d for d in self.dmg if t - d[0] < 6]
        self.heard = [h for h in self.heard if t - h[0] < 3]
        self.trail.append((t, x, y))
        self.trail = [p for p in self.trail if t - p[0] < 2.0]

        hp_frac = me["hp"] / max(me["max_hp"], 1)
        energy = me.get("energy", 0)
        mechs = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        npcs = [m for m in obs.get("mechs", []) if m.get("kind") == "npc"]
        attackers = {d[2] for d in self.dmg if t - d[0] < 4 and d[2] is not None}

        # ---- pick a target: whoever is shooting us, else the closest threat in range
        def tscore(m):
            d = math.dist(m["pos"][:2], (x, y))
            return d + 0.08 * m.get("hp", 0) - (80 if m["id"] in attackers else 0)
        cands = [m for m in mechs if m["id"] in attackers or math.dist(m["pos"][:2], (x, y)) < 160]
        if not cands:
            cands = [m for m in npcs if m["id"] in attackers]
        target = min(cands, key=tscore) if cands else None
        nearest = min((math.dist(m["pos"][:2], (x, y)) for m in mechs), default=INF)

        # ---- movement
        move = (0.0, 0.0)
        boost = False
        mode = "hold"
        in_goal = self._in_goal(x, y)
        if not me.get("in_zone", True):
            mode = "zone!"
            move = self._path_dir(x, y)
            boost = energy > 10
        elif hp_frac < 0.4 and mechs and nearest < 120:
            mode = "retreat"
            foe = min(mechs, key=lambda m: math.dist(m["pos"][:2], (x, y)))
            ax, ay = self._move_toward(foe["pos"][0], foe["pos"][1], x, y)
            px, py = self._path_dir(x, y)
            if in_goal:
                gx, gy, r = self.goal
                # stay inside the goal circle: drop the outward part near its edge
                rx, ry = x - gx, y - gy
                rn = math.hypot(rx, ry)
                if rn > 0.7 * r and (ax * rx + ay * ry) > 0:
                    ax, ay = ax - rx / rn * (ax * rx + ay * ry) / rn, ay - ry / rn * (ax * rx + ay * ry) / rn
                move = (ax, ay)
            else:
                move = (ax + px, ay + py)
            if math.hypot(*move) < 0.5:
                # foe stands on our way: sidestep, using the part of the path across the foe's line
                fx, fy = -ax, -ay
                dot = px * fx + py * fy
                sx, sy = px - dot * fx, py - dot * fy
                if math.hypot(sx, sy) < 0.2:
                    sx, sy = -fy, fx
                n = math.hypot(sx, sy)
                if n < 1e-9:   # foe stacked right on top of us: no line to step across
                    move = (px, py) if math.hypot(px, py) > 0.5 else (1.0, 0.0)
                else:
                    move = (sx / n, sy / n)
            boost = energy > 25
        else:
            drop = self._drop_goal(obs, mechs, x, y)
            pick = None if drop else self._pickup_goal(obs, me, mechs, x, y, t)
            if drop is not None:
                mode = "drop"
                if math.dist(drop["pos"][:2], (x, y)) > drop.get("radius", 8) - 3:
                    move = self._move_toward(x, y, *drop["pos"][:2])
            elif pick is not None:
                mode = "loot"
                if math.dist(pick["pos"][:2], (x, y)) < self.cfg["loot"]["pickup_radius"] - 0.5:
                    act["take"] = pick["id"]
                else:
                    move = self._move_toward(x, y, *pick["pos"][:2])
            elif not in_goal:
                mode = "path"
                move = self._path_dir(x, y)
                boost = self._late(zone, t, x, y) and energy > 30
            elif target is not None and math.dist(target["pos"][:2], (x, y)) < 200:
                mode = "fight"
                if t - self.strafe_t > self.rng.uniform(1.2, 2.2):
                    self.strafe, self.strafe_t = -self.strafe, t
                by = math.atan2(target["pos"][1] - y, target["pos"][0] - x) + self.strafe * math.pi / 2
                move = (math.cos(by) * 0.8, math.sin(by) * 0.8)
                if not self._in_goal(x + move[0] * 6, y + move[1] * 6):
                    self.strafe, self.strafe_t = -self.strafe, t
                    move = (-move[0], -move[1])
            else:
                mode = "hold"
                move = self._hold_move(x, y)

        # ---- stuck? jump and jet the way we were going
        if math.hypot(*move) <= 0.5:
            self.move_t = t
        if (mode != "fight" and math.hypot(*move) > 0.5 and len(self.trail) > 10
                and t - self.trail[0][0] > 1.4 and t - self.move_t > 1.4):
            moved = math.dist(self.trail[0][1:], (x, y))
            if moved < 2.5 and t > self.unstick_until + 1.5:
                ang = math.atan2(move[1], move[0]) + self.rng.choice((-1, 1)) * self.rng.uniform(0.4, 1.3)
                self.unstick_dir = (math.cos(ang), math.sin(ang))
                self.unstick_until = t + 1.2
        if t < self.unstick_until:
            mode += "+unstick"
            move = self.unstick_dir
            if me.get("grounded") and energy > 30:
                act["jump"] = True
            boost = energy > 5
        act["move"] = [round(move[0], 3), round(move[1], 3)]
        act["boost"] = bool(boost)

        # ---- aim and fire
        weapons = me.get("weapons") or [None, None]
        gun = weapons[0] if weapons and weapons[0] and weapons[0].get("kind") == "gun" else None
        sec = weapons[1] if len(weapons) > 1 else None
        fire = [False, False]
        aim = None
        if target is not None:
            aim, fire = self._engage(me, target, gun, sec, weapons, w, escaping=mode.startswith(("zone!", "retreat")))
        else:
            recent = [d for d in self.dmg if t - d[0] < 2.5]
            if recent:
                aim = [recent[-1][1], 0.0]
            elif self.heard:
                aim = [self.heard[-1][1], 0.0]
            elif mode == "hold":
                base = self.hold[1] if self.hold else self.scan0
                aim = [wrap(base + 1.0 * math.sin(t * 0.7)), 0.0]
            elif move != (0.0, 0.0):
                aim = [math.atan2(move[1], move[0]), 0.0]
            # laser charging with nothing to shoot at: let it go
            if sec and sec.get("kind") == "laser" and sec.get("charge", 0) > 0:
                fire[1] = False
            if gun and not mechs and gun.get("reload", 0) == 0 and gun.get("ammo", 99) < gun["stats"].get("mag", 12) * 0.5:
                act["reload"] = True
        if aim is not None:
            act["aim"] = [round(aim[0], 4), round(max(-0.6, min(1.2, aim[1])), 4)]
        if fire[0] or fire[1]:
            act["fire"] = fire

        # ---- upgrades
        offer = obs.get("offer")
        if offer and offer.get("choices"):
            ids = [c.get("id") for c in offer["choices"]]
            best = min(range(len(ids)), key=lambda i: UPGRADE_ORDER.index(ids[i]) if ids[i] in UPGRADE_ORDER else 99)
            act["pick"] = best

        mf.draw.label(f"{mode} hp{me['hp']:.0f} e{energy:.0f}")
        if target is not None:
            mf.draw.line([x, y, z + 7], [target["pos"][0], target["pos"][1], target["pos"][2] + 4.5], "#ff5040")
        if self.goal:
            mf.draw.point([self.goal[0], self.goal[1], z], "#40c0ff", 6)
        return act

    # ------------------------------------------------------------ pieces

    def _late(self, zone, t, x, y) -> bool:
        """Will walking leave us outside the circle when the shrink catches up?"""
        gx, gy, r = self.goal
        need = max(0.0, math.hypot(x - gx, y - gy) - r)
        walk = self.cfg["mech"].get("walk_speed", 11.7)
        left = zone.get("shrink_end", t + 999) - t
        return need / max(walk, 1) > 0.7 * left

    def _hold_move(self, x, y):
        w = self.world
        if w is None or self.dist is None:
            return (0.0, 0.0)
        if self.hold is None or self.hold_key != self.field_key:
            spots = [s for s in w.cover_spots(self.dist, x, y) if s[1] not in self.bad_spots]
            if spots:
                pickn = min(len(spots), 3)
                s = spots[self.rng.randrange(pickn)]
                self.hold = (s[1], s[2])
            else:
                self.hold = (w.cell(x, y), self.scan0)
            self.hold_key = self.field_key
        hx, hy = w.center(self.hold[0])
        if math.hypot(hx - x, hy - y) < 2.5:
            return (0.0, 0.0)
        if len(self.trail) > 10 and math.dist(self.trail[0][1:], (x, y)) < 1.0 and math.hypot(hx - x, hy - y) > 6:
            self.bad_spots.add(self.hold[0])     # can't get there: pick another next time
            self.hold = None
            return (0.0, 0.0)
        dx, dy = self._move_toward(x, y, hx, hy)
        return (dx * 0.4, dy * 0.4)   # about 4.7 m/s: under the step-sound speed

    def _drop_goal(self, obs, mechs, x, y):
        best = None
        walk = self.cfg["mech"].get("walk_speed", 11.7)
        for d in obs.get("drops", []):
            if d.get("holder") not in (None, self.me_id):
                continue
            dd = math.dist(d["pos"][:2], (x, y))
            if dd > 140 or d.get("lands_in", 0) > dd / walk + 8:
                continue
            if any(math.dist(m["pos"][:2], d["pos"][:2]) < 60 for m in mechs):
                continue
            if not obs["self"].get("in_zone", True) or not self._safe(obs["zone"], *d["pos"][:2], margin=0.0):
                continue
            if best is None or dd < best[0]:
                best = (dd, d)
        return best[1] if best else None

    def _pickup_goal(self, obs, me, mechs, x, y, t):
        weapons = me.get("weapons") or [None, None]
        if len(weapons) > 1 and weapons[1] is not None:
            return None
        if any(math.dist(m["pos"][:2], (x, y)) < 70 for m in mechs):
            return None
        held = {wpn.get("kind") for wpn in weapons if wpn}
        best = None
        for p in obs.get("pickups", []):
            if p.get("kind") not in WANT or p.get("kind") in held:
                continue
            if t - self.bad_pickups.get(p["id"], -99) < 10:
                continue
            if abs(p["pos"][2] - me["pos"][2]) > 3.5:
                continue      # on a roof or below us: not worth the climb
            dd = math.dist(p["pos"][:2], (x, y))
            if dd > 60 or not me.get("in_zone", True) or not self._safe(obs["zone"], *p["pos"][:2]):
                continue   # loot never despawns, so the zone can close over it
            if best is None or dd < best[0]:
                best = (dd, p)
        return best[1] if best else None

    def _engage(self, me, tgt, gun, sec, weapons, w, escaping=False):
        x, y, z = me["pos"]
        mz = z + me["stats"].get("muzzle_height", 7)
        ayaw, apitch = me.get("aim", (0.0, 0.0))
        fire = [False, False]
        tx, ty, tz = tgt["pos"]
        is_npc = tgt.get("kind") == "npc"
        th = 3.5 if is_npc else 9.0
        center = (tx, ty, tz + th / 2)
        dh = math.hypot(tx - x, ty - y)
        clear = True
        if w is not None:
            clear = w.los((x, y, mz), center) or w.los((x, y, mz), (tx, ty, tz + th * 0.9))
        gspeed = gun["stats"].get("speed", 252) if gun else 252
        yaw, pitch, _ = self._lead(me, tgt, gspeed)
        if is_npc:
            pitch = math.atan2(tz + 1.75 - mz, max(dh, 1e-3))
        rad = 1.8 if is_npc else 2.5 * (1 + 0.05 * tgt.get("cubes", 0))
        tol_y = math.atan2(rad * 0.8, max(dh, 1.0)) + 0.004
        tol_p = math.atan2(th * 0.4, max(dh, 1.0)) + 0.004
        on = abs(wrap(ayaw - yaw)) < tol_y and abs(apitch - pitch) < tol_p
        if gun and clear and dh < gun["stats"].get("range", 230) - 5 and gun.get("ammo", 1) > 0 and gun.get("reload", 0) == 0 and on:
            fire[0] = True
        aim = [yaw, pitch]
        if sec:
            kind = sec.get("kind")
            st = sec.get("stats", {})
            if kind == "lightning":
                rng = st.get("range", 45)
                cone = math.radians(st.get("cone_deg", 30)) / 2
                direct = math.atan2(ty - y, tx - x)
                if dh < rng - 2 and abs(wrap(ayaw - direct)) < cone * 0.8 and clear and me.get("energy", 0) > 8:
                    fire[1] = True
            elif kind == "laser" and not escaping:
                full = st.get("charge", 0.75)
                ch = sec.get("charge", 0)
                if ch >= full * 0.98:
                    # hitscan: aim straight at it, and let go (fire false) once on target
                    dyaw = math.atan2(ty - y, tx - x)
                    dpitch = math.atan2(tz + th / 2 - mz, max(dh, 1e-3))
                    aim = [dyaw, dpitch]
                    laser_on = abs(wrap(ayaw - dyaw)) < tol_y and abs(apitch - dpitch) < tol_p
                    fire[1] = not (laser_on and clear)
                elif ch > 0:
                    fire[1] = True
                elif not escaping and clear and dh < st.get("range", 280) - 5 and sec.get("cooldown", 0) <= 0.05:
                    fire[1] = True
        return aim, fire


if __name__ == "__main__":
    mf.run(Bot)
