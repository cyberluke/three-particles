/**
 * Scalar 3D simplex noise for the CPU fallback path.
 *
 * Mirrors the TSL simplex used on the GPU (`three-particles/webgpu/tsl-noise.ts`,
 * Ashima / Gustavson scheme, ~42.0 normalisation) with the classic 12-gradient
 * simplex construction (70.0 normalisation). Same statistical band [-1..1];
 * bit-perfect parity is not required.
 *
 * Allocation-free: all state is module-local scratch.
 *
 * @module
 */

const GRAD3 = new Float32Array([
  1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0, 1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0,
  -1, 0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,
]);

const F3 = 1 / 3;
const G3 = 1 / 6;

/** 3D simplex noise in approximately [-1, 1]. */
export const snoise3 = (x: number, y: number, z: number): number => {
  const s = (x + y + z) * F3;
  const i = Math.floor(x + s);
  const j = Math.floor(y + s);
  const k = Math.floor(z + s);
  const t = (i + j + k) * G3;
  const x0 = x - (i - t);
  const y0 = y - (j - t);
  const z0 = z - (k - t);

  let i1: number, j1: number, k1: number;
  let i2: number, j2: number, k2: number;
  if (x0 >= y0) {
    if (y0 >= z0) {
      i1 = 1;
      j1 = 0;
      k1 = 0;
      i2 = 1;
      j2 = 1;
      k2 = 0;
    } else if (x0 >= z0) {
      i1 = 1;
      j1 = 0;
      k1 = 0;
      i2 = 1;
      j2 = 0;
      k2 = 1;
    } else {
      i1 = 0;
      j1 = 0;
      k1 = 1;
      i2 = 1;
      j2 = 0;
      k2 = 1;
    }
  } else {
    if (y0 < z0) {
      i1 = 0;
      j1 = 0;
      k1 = 1;
      i2 = 0;
      j2 = 1;
      k2 = 1;
    } else if (x0 < z0) {
      i1 = 0;
      j1 = 1;
      k1 = 0;
      i2 = 0;
      j2 = 1;
      k2 = 1;
    } else {
      i1 = 0;
      j1 = 1;
      k1 = 0;
      i2 = 1;
      j2 = 1;
      k2 = 0;
    }
  }

  const x1 = x0 - i1 + G3;
  const y1 = y0 - j1 + G3;
  const z1 = z0 - k1 + G3;
  const x2 = x0 - i2 + 2 * G3;
  const y2 = y0 - j2 + 2 * G3;
  const z2 = z0 - k2 + 2 * G3;
  const x3 = x0 - 1 + 3 * G3;
  const y3 = y0 - 1 + 3 * G3;
  const z3 = z0 - 1 + 3 * G3;

  const ii = i & 255;
  const jj = j & 255;
  const kk = k & 255;

  let n = 0;
  let m = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
  if (m > 0) {
    const g = (ii % 12) * 3;
    // corner gradient hash mixes all three lattice axes
    m *= m;
    n += m * m * (GRAD3[g] * x0 + GRAD3[g + 1] * y0 + GRAD3[g + 2] * z0);
  }
  m = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
  if (m > 0) {
    const g = (((ii + i1) % 12) + (jj + j1) + (kk + k1)) % 12;
    const gg = g * 3;
    m *= m;
    n += m * m * (GRAD3[gg] * x1 + GRAD3[gg + 1] * y1 + GRAD3[gg + 2] * z1);
  }
  m = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
  if (m > 0) {
    const g = ((((ii + i2) % 12) + (jj + j2) + (kk + k2)) % 12) | 0;
    const gg = g * 3;
    m *= m;
    n += m * m * (GRAD3[gg] * x2 + GRAD3[gg + 1] * y2 + GRAD3[gg + 2] * z2);
  }
  m = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
  if (m > 0) {
    const g = (((ii + 1) % 12) + (jj + 1) + (kk + 1)) % 12;
    const gg = g * 3;
    m *= m;
    n += m * m * (GRAD3[gg] * x3 + GRAD3[gg + 1] * y3 + GRAD3[gg + 2] * z3);
  }
  return n * 0.8389; // ~70 -> [-1,1] band matching the TSL implementation
};

/**
 * Micro-noise layer (§8): 3 octaves of simplex evaluated in the
 * (along-path, time, channel) frame so only the micro layer swims.
 */
export const microNoise = (
  t: number,
  time: number,
  microFrequency: number,
  channel: number
): number => {
  const p = t * microFrequency;
  return (
    snoise3(p, time, channel) * 0.6 +
    snoise3(p * 2.13, time * 2.13, channel) * 0.27 +
    snoise3(p * 4.71, time * 4.71, channel) * 0.13
  );
};

/**
 * Impulse-noise layer: one octave at a detuned phase (weight 0.08 in the
 * centerline composition, §7).
 */
export const impulseNoise = (
  t: number,
  time: number,
  microFrequency: number,
  channel: number
): number =>
  snoise3(t * microFrequency * 2.7 + 11.37, time * 1.7 + 3.1, channel + 9.7);
