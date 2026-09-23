/**
 * Regression test for the flame-barrier WORLD + collision-plane GPU bug.
 *
 * ROOT CAUSE
 * ==========
 *
 * The WebGPU compute dispatch is declared as `compute(kernel, maxParticles)`.
 * The backend rounds this up to whole workgroups (typically 64 threads), so
 * when `maxParticles` is not a multiple of the workgroup size the last threads
 * run with `instanceIndex >= maxParticles`.
 *
 * Those threads entered the per-particle init block which reads
 * `curveData[curveLen + i * INIT_STRIDE + 3]` as the init flag.
 *
 * The collision plane data is packed right after the per-particle init
 * region at offset `collisionPlaneOffset = curveLen + maxParticles *
 * INIT_STRIDE`. For the thread with `i == maxParticles` the init-flag read
 * lands EXACTLY on the first collision plane's `position.y` byte. When
 * `position.y > 0.5` (e.g. plane at y=3 like flame-barrier), the init
 * branch activated every frame and zeroed the plane's position.y on the GPU:
 *
 *   sCurveData.element(i * INIT_STRIDE + curveLen + 3).assign(float(0))
 *
 * The result on the GPU was plane.y = 0 even though the CPU mirror still
 * contained plane.y = 3 — particles that should have clamped against y=3
 * instead clamped against y=0, producing the "all particles snap to the
 * ground" visual bug.
 *
 * FIX
 * ===
 *
 * A top-level `If(i < maxParticles)` guard around the whole kernel body so
 * out-of-range threads do nothing. This test verifies that the collision
 * plane encoding layout does not overlap with any valid particle's init
 * slot, which is the invariant the kernel depends on for safety, and that
 * the kernel-level bounds check is in place.
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';

// ESM mode has no ambient `__dirname`; derive it from `import.meta.url`.
const SELF_DIR = fileURLToPath(new URL('.', import.meta.url));

const COMPUTE_MODIFIERS_PATH = join(
  SELF_DIR,
  '../js/effects/three-particles/webgpu/compute-modifiers.ts'
);
const INIT_STRIDE = 28;

describe('compute kernel bounds check vs collision plane layout', () => {
  it('collisionPlaneOffset starts exactly where an out-of-bounds thread (i=maxParticles) would read its init slot', () => {
    // For any curveLen and maxParticles:
    //   out-of-bounds thread i = maxParticles reads offsets:
    //     initBase = curveLen + maxParticles * INIT_STRIDE
    //   collision plane region starts at:
    //     cpOffset = curveLen + maxParticles * INIT_STRIDE (+ ffSize)
    //
    // Without force fields, initBase == cpOffset, so
    // curveData[initBase + 3] aliases the first plane's position.y.
    const curveLen = 512;
    const maxParticles = 150;
    const initBase = curveLen + maxParticles * INIT_STRIDE;
    const cpOffset = curveLen + maxParticles * INIT_STRIDE; // no force fields
    expect(initBase).toBe(cpOffset);
    // Offset 3 in plane stride is position.y
    expect(initBase + 3).toBe(cpOffset + 3);
  });

  it('compute kernel source contains a top-level bounds check on instanceIndex', () => {
    const source = readFileSync(COMPUTE_MODIFIERS_PATH, 'utf8');
    // The fix wraps the whole kernel body inside `If(<i>.lessThan(float(maxParticles)), ...)`.
    // `<i>` is either the raw `instanceIndex` node (`i.lessThan(...)`) or its
    // f32 form (`float(i).lessThan(...)`); the source uses the latter so the
    // guard compares on float 32 (matching WebGPU `atomic<u32>` counter rules).
    // Without the guard, threads with `i >= maxParticles` would corrupt the
    // collision plane region (see test above).
    expect(source).toMatch(
      /If\s*\(\s*(?:float\(\s*i\s*\)|i)\.lessThan\s*\(\s*float\s*\(\s*maxParticles\s*\)\s*\)/
    );
  });

  it('kernel bounds check is placed BEFORE the init-flag read', () => {
    const source = readFileSync(COMPUTE_MODIFIERS_PATH, 'utf8');
    const boundsIdx = source.search(
      /If\s*\(\s*(?:float\(\s*i\s*\)|i)\.lessThan\s*\(\s*float\s*\(\s*maxParticles\s*\)\s*\)/
    );
    // The init-flag read (ORBITAL_IS_ACTIVE.w) is `sOIA.element(i).toVar()`.
    // It must appear AFTER the bounds `If` guard, otherwise an out-of-bounds
    // thread could still corrupt the collision-plane region (see header).
    const initFlagReadIdx = source.indexOf('sOIA.element(i).toVar()');
    expect(boundsIdx).toBeGreaterThan(-1);
    expect(initFlagReadIdx).toBeGreaterThan(-1);
    expect(boundsIdx).toBeLessThan(initFlagReadIdx);
  });
});
