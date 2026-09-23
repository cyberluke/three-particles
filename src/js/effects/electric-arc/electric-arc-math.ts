/**
 * Scalar math helpers shared by the CPU and (conceptually mirrored on) GPU
 * electric-arc paths.
 *
 * The integer hash mirrors the engine PCG machinery
 * (`three-particles/webgpu/compute-modifiers.ts`: `pcgRawU32` / `pcg01` /
 * `mixBirthSeed`) using `Math.imul` + `>>> 0`, so the same
 * `(seed, epoch, cell, axis)` inputs produce a (bit-wise compatible) stable
 * topology between the CPU and GPU backends. Behavioral parity, not
 * bit-perfect parity, is the requirement.
 *
 * @module
 */

/** 32-bit unsigned normalisation matching the TSL `u32` chain. */
export const u32 = (n: number): number => n >>> 0;

/** Raw uint PCG word, scalar twin of the TSL `pcgRawU32`. */
export const pcgRawU32Scalar = (seedU: number): number => {
  const s = u32(Math.imul(u32(seedU), 747796405) + 2891336453);
  const shifted = s >>> ((s >>> 28) + 4);
  let word = u32(shifted ^ s);
  word = u32(Math.imul(word, 277803737));
  return u32((word >>> 22) ^ word);
};

/** PCG hash in [0,1), scalar twin of `pcg01`. */
export const pcg01Scalar = (seedU: number): number =>
  pcgRawU32Scalar(seedU) * (1 / 4294967296);

/** Integer channel mix, scalar twin of `mixBirthSeed` (Knuth odd multiply + xor). */
export const mixSeedScalar = (a: number, b: number, c: number): number =>
  u32(u32(Math.imul(u32(a), 2654435761)) ^ u32(b) ^ u32(c));

/**
 * Deterministic [0,1) hash from the four discharge-epoch inputs (§9/§10):
 * `seed`, `epoch`, `cellIndex`, `axis`.
 */
export const dischargeHash = (
  seed: number,
  epoch: number,
  cellIndex: number,
  axis: number
): number =>
  pcg01Scalar(
    mixSeedScalar(mixSeedScalar(cellIndex, seed, 1), epoch, axis + 2)
  );

export const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

export const lerp = (a: number, b: number, t: number): number =>
  a + (b - a) * t;

export const fract = (v: number): number => v - Math.floor(v);

/**
 * Endpoint-pinning envelope (§6): `pow(sin(pi * t), pinPower)`.
 * Exact zero at t=0 and t=1 is additionally enforced by the callers via the
 * explicit endpoint override (no reliance on floating-point `sin(PI)`).
 */
export const pinEnvelope = (t: number, pinPower: number): number => {
  if (t <= 0) return 0;
  if (t >= 1) return 0;
  const s = Math.sin(Math.PI * t);
  return Math.pow(s < 0 ? 0 : s, pinPower);
};

/** Coarse displacement weight for a lattice cell pair (linear, §7). */
export const coarseOffset = (
  seed: number,
  epoch: number,
  t: number,
  coarseKnots: number,
  axis: number
): number => {
  const cellF = t * coarseKnots;
  const cell0 = Math.floor(cellF);
  const f = cellF - cell0;
  const h0 = dischargeHash(seed, epoch, cell0, axis) * 2 - 1;
  const h1 = dischargeHash(seed, epoch, cell0 + 1, axis) * 2 - 1;
  return h0 + (h1 - h0) * f;
};

/** §11 chaos-derived displacement envelope scale (clamped for pathological arcs). */
export const chaosAmplitude = (distance: number, chaosity: number): number => {
  const a =
    distance * lerp(0.0025, 0.045, Math.pow(clamp(chaosity, 0, 1), 1.6));
  return clamp(a, 0.0025, Math.max(0.05, distance * 0.25));
};

/** Maximum slot count of the pulse chaos model (coarseKnots <= 20 + guard). */
export const PULSE_MAX_SLOTS = 21;

/** Rise part of the spike envelope (fraction of one slot). */
export const PULSE_RISE = 0.15;

/** Exponential decay constant of the spike tail. */
export const PULSE_DECAY = 3;

/** Degradation multipliers: amplitude halves, and half-widths shrink. */
export const PULSE_DEG_GAIN = 0.5;
export const PULSE_DEG_WIDTH = 0.5;

/**
 * Asymmetric pulse-slot envelope for `f` in [0..1] (§11 v2). Three slot
 * classes (selected from the slot hash in `pulseOffset`, not here):
 *
 * ```text
 * spike    : 0..0.15 linear rise, then an exponential slow tail  (zap)
 * plateau  : constant 1 across the slot                          (bzzz)
 * gap      : constant 0                                          (stop)
 * ```
 *
 * This function returns the *spike* shape; `pulseOffset` / TSL twins apply
 * the class selection on top of it. 1 at the rise end, ~0.08 at f=1.
 */
export const pulseEnvelope = (f: number): number => {
  if (f <= 0) return 0;
  if (f < PULSE_RISE) return f / PULSE_RISE;
  return Math.exp(-PULSE_DECAY * (f - PULSE_RISE));
};

/** module scratch for `pulseOffset` (allocation-free; single-threaded hot loop) */
const _pulseW = new Float32Array(PULSE_MAX_SLOTS);

/**
 * Second chaos model (§11 v2): a strongly asymmetric 3-class ladder per
 * non-uniform PCG slot — `0.2` gap / `0.4` plateau / `>=0.4` spike —
 * producing the `zap zap … bzzz … zap zap` discharge rhythm, with
 * occasional (≈25%) "chaos degradation": gain x0.5 and half slot width,
 * so the ladder visibly flattens for a few slots. Degraded slots do not
 * overlap gaps. Scalar twin of `coarsePulseOf` in the compute kernel
 * (same integer mixing, same classes).
 */
export const pulseOffset = (
  seed: number,
  epoch: number,
  t: number,
  slots: number,
  axis: number
): number => {
  const n = Math.min(PULSE_MAX_SLOTS, Math.max(1, Math.round(slots)));
  const wXor = u32(11 * (Math.round(axis) + 1));
  const lvlXor = u32(Math.round(axis) + 2);
  const clsXor = u32(Math.round(axis) + 5);
  const degXor = u32(Math.round(axis) + 9);

  let total = 0;
  for (let i = 0; i < n; i++) {
    const w = 0.3 + 0.7 * pcg01Scalar(mixSeedScalar(i + 1, seed, epoch) ^ wXor);
    _pulseW[i] = w;
    total += w;
  }
  const inv = total > 0 ? 1 / total : 1;

  let acc = 0;
  for (let i = 0; i < n; i++) {
    const b0 = acc;
    acc += _pulseW[i] * inv;
    const wN = acc - b0;
    if (t < acc || i === n - 1) {
      // occasional chaos degradation (gain x0.5, half slot width)
      const degOn =
        pcg01Scalar(mixSeedScalar(i + 900, seed, epoch) ^ degXor) < 0.25;
      const slotW = degOn ? wN * 0.5 : wN;
      if (t >= b0 + slotW && t < acc) return 0; // dead part of degraded slot

      const cls = pcg01Scalar(mixSeedScalar(i + 128, seed, epoch) ^ clsXor);
      const fRaw = t < b0 ? 0 : Math.min(1, (t - b0) / Math.max(slotW, 1e-6));
      const f = Math.min(fRaw, 0.999999);
      const level = pcg01Scalar(mixSeedScalar(i, seed, epoch) ^ lvlXor) * 2 - 1;
      let out: number;
      if (cls < 0.2) {
        out = 0; // hard gap ("stop")
      } else if (cls < 0.4) {
        out = level; // signed rectangular plateau ("bzzz")
      } else {
        out = level * pulseEnvelope(f); // sharp zig + slow exp tail ("zap")
      }
      if (degOn) out *= PULSE_DEG_GAIN;
      return out;
    }
  }
  return 0;
};

/** Organic hold window: from 45%… */
export const ORGANIC_HOLD_START = 0.45;
/** …with up to 40% extra random hold. */
export const ORGANIC_HOLD_SPAN = 0.4;

/**
 * Third chaos model (§11, "organic"): smooth cosine-eased lattice kinks that
 * randomly stop. Each knot cell eases from level `i` to level `i+1` with a
 * `smoothstep` over its first part, then *holds* the level for a random
 * fraction of the slot (pause). Produces a breathing, organic line with
 * occasional flat rests — different from both `linear` and `pulse`. Scalar
 * twin of `coarseOrganicOf` in the TSL kernel.
 */
export const organicOffset = (
  seed: number,
  epoch: number,
  t: number,
  knots: number,
  axis: number
): number => {
  const n = Math.min(PULSE_MAX_SLOTS, Math.max(1, Math.round(knots)));
  const ax = Math.round(axis);
  const lvlXor = u32(ax + 2);
  const holdXor = u32(31 * (ax + 1));

  const cellF = t * n;
  const cell0 = Math.floor(cellF);
  const f = cellF - cell0;
  const c1 = cell0 + 1 > n ? n : cell0 + 1;
  const h0 =
    pcg01Scalar(mixSeedScalar(Math.min(cell0, n), seed, epoch) ^ lvlXor) * 2 -
    1;
  const h1 = pcg01Scalar(mixSeedScalar(c1, seed, epoch) ^ lvlXor) * 2 - 1;

  const holdFrac =
    ORGANIC_HOLD_START +
    ORGANIC_HOLD_SPAN *
      pcg01Scalar(
        mixSeedScalar(Math.min(cell0, n) + 700, seed, epoch) ^ holdXor
      );

  if (f >= holdFrac) return h1; // pause at the level
  // smoothstep-eased travel over the first `holdFrac` of the cell
  const g = f / holdFrac;
  const e = g * g * (3 - 2 * g);
  return h0 + (h1 - h0) * e;
};

/** Default `flickerHz` derived from chaos (§11). */
export const chaosFlickerHz = (chaosity: number): number =>
  lerp(8, 42, clamp(chaosity, 0, 1));

/**
 * Thin-out factor of the variable thickness model: `1` = full core width
 * for a straight segment, `0.55` for a fully-deflected (rotated) sample.
 */
export const PULSE_THIN_MIN = 0.55;

/**
 * Variable-width factor for a centerline sample given the two in-plane
 * displacement components (u/v basis) of the chaos model.
 * Straight (zero displacement) -> 1, strongest deflection -> 0.55.
 */
export const widthFactor = (ou: number, ov: number): number => {
  const rot = Math.sqrt(ou * ou + ov * ov);
  return 1 - (1 - PULSE_THIN_MIN) * clamp(Math.min(1.4142136, rot), 0, 1);
};

/** Rotate a point around Z (degrees), in place, returning the same tuple. */
const DEG = Math.PI / 180;
export const rotateZ2 = (
  v: { x: number; y: number },
  deg: number
): { x: number; y: number } => {
  if (!deg) return v;
  const r = deg * DEG;
  const c = Math.cos(r);
  const s = Math.sin(r);
  const x = v.x * c - v.y * s;
  const y = v.x * s + v.y * c;
  v.x = x;
  v.y = y;
  return v;
};

/**
 * Global discharge flicker for an epoch (§9 + §15): one small value per
 * epoch, identical on CPU and GPU (derived from the same PCG machinery).
 */
export const globalFlicker = (seed: number, epoch: number): number =>
  0.78 + 0.27 * pcg01Scalar(mixSeedScalar(seed, epoch, 7));
