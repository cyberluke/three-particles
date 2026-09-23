/**
 * WebGPU compute kernel for the Electric Arc centerline (§12).
 *
 * One storage buffer of `vec4 * totalSamples` (xyz = centerline position,
 * w = local brightness), 16-byte aligned, written by a single 1D dispatch.
 *
 * The model mirrors `electric-arc-cpu.ts` exactly:
 *   - coarse piecewise-linear random lattice kinks (PCG hash, epoch-based)
 *   - TSL simplex micro noise (2 octaves, continuous swim)
 *   - impulse noise (single octave, detuned phase)
 *   - sin(pi*t)^pin envelope with exact endpoints
 *   - chaos-derived amplitudes
 *
 * PCG machinery is the engine's shared TSL implementation
 * (`compute-modifiers.ts`: `pcgRawU32` / `pcg01` / `mixBirthSeed`), so the
 * same seed/epoch/cell/axis inputs yield the same approximate topology as
 * the scalar CPU twin.
 *
 * @module
 */
import { Vector3 } from 'three';
import {
  Fn,
  PI,
  float,
  vec3,
  vec4,
  sqrt,
  storage,
  instanceIndex,
  uniform,
  compute,
  mix,
  floor,
  sin,
  pow,
  exp,
  If,
  Loop,
  uint as tuint,
  type ShaderNodeObject,
  type Node,
} from 'three/tsl';
import { StorageBufferAttribute } from 'three/webgpu';
import {
  pcg01,
  mixBirthSeed,
} from '../../three-particles/webgpu/compute-modifiers.js';
import { snoise3D } from '../../three-particles/webgpu/tsl-noise.js';
import type { NormalizedElectricArcConfig } from '../electric-arc-types.js';

/** Samples per branch polyline. */
export const ARC_BRANCH_SAMPLES = 8;

export type ElectricArcComputePipeline = {
  computeNode: unknown;
  arcBuffer: StorageBufferAttribute;
  totalSamples: number;
  /** Main-arc sample count. */
  mainCount: number;
  /** Per-frame uniform accessors (written from CPU, no readback). */
  uniforms: {
    start: ShaderNodeObject<Node>;
    end: ShaderNodeObject<Node>;
    basisU: ShaderNodeObject<Node>;
    basisV: ShaderNodeObject<Node>;
    time: ShaderNodeObject<Node>;
    pin: ShaderNodeObject<Node>;
    amp: ShaderNodeObject<Node>;
    knots: ShaderNodeObject<Node>;
    microF: ShaderNodeObject<Node>;
    epoch: ShaderNodeObject<Node>;
    seed: ShaderNodeObject<Node>;
    seedInv: ShaderNodeObject<Node>;
    brightnessVar: ShaderNodeObject<Node>;
    globalFlicker: ShaderNodeObject<Node>;
    intensity: ShaderNodeObject<Node>;
    segLast: ShaderNodeObject<Node>;
    branchBase: ShaderNodeObject<Node>;
    branchLen0: ShaderNodeObject<Node>;
    branchLen1: ShaderNodeObject<Node>;
    branchProb: ShaderNodeObject<Node>;
  };
  dispose: () => void;
};

export function createElectricArcCompute(
  cfg: NormalizedElectricArcConfig
): ElectricArcComputePipeline {
  const segments = cfg.segments;
  const branchCount = cfg.branches.enabled
    ? Math.max(0, Math.min(8, Math.round(cfg.branches.maxCount)))
    : 0;
  const totalSamples = segments + branchCount * ARC_BRANCH_SAMPLES;

  // ── one 16-byte-aligned vec4 storage buffer (§12) ─────────────────────────
  const arcBuffer = new StorageBufferAttribute(
    new Float32Array(totalSamples * 4),
    4
  );

  // ── variable-width factor per sample (1 = full core thickness) ────────────
  const widthBuffer = new StorageBufferAttribute(
    new Float32Array(totalSamples),
    1
  );

  // ── per-frame uniforms ─────────────────────────────────────────────────────
  const nStart = uniform(new Vector3().copy(cfg.start));
  const nEnd = uniform(new Vector3().copy(cfg.end));
  const nU = uniform(new Vector3(1, 0, 0));
  const nV = uniform(new Vector3(0, 0, 1));
  const nTime = uniform(float(0));
  const nPin = uniform(float(cfg.endpointPinning));
  const nAmp = uniform(float(cfg.amplitude));
  const nKnots = uniform(float(cfg.coarseKnots));
  const nMicroF = uniform(float(cfg.microFrequency));
  const nEpoch = uniform(float(0));
  const nSeed = uniform(float(cfg.seed));
  const nSeedInv = uniform(float(cfg.seed * 0.001 + 1));
  const nBvar = uniform(float(cfg.brightnessVariation));
  const nGlobalF = uniform(float(1));
  const nIntensity = uniform(float(cfg.intensity));
  const nSegLast = uniform(float(segments - 1));
  const nBranchBase = uniform(float(branchCount > 0 ? segments : 0));
  const nLen0 = uniform(
    float(
      cfg.branches.length[0] * Math.max(cfg.end.distanceTo(cfg.start), 0.01)
    )
  );
  const nLen1 = uniform(
    float(
      cfg.branches.length[1] * Math.max(cfg.end.distanceTo(cfg.start), 0.01)
    )
  );
  const nProb = uniform(
    float(cfg.branchProbability || cfg.branches.probability)
  );

  // ── centerline helpers ─────────────────────────────────────────────────────

  /** Coarse piecewise-linear lattice kink (§7). */
  const coarseOf = Fn(
    ({
      tN,
      axisU,
    }: {
      tN: ShaderNodeObject<Node>;
      axisU: ShaderNodeObject<Node>;
    }) => {
      const seedU = nSeed.toUint();
      const epochU = nEpoch.toUint();
      const cellF = tN.mul(nKnots);
      const c0 = floor(cellF);
      const f = cellF.sub(c0);
      const c1 = c0.add(1);
      const h0 = pcg01(mixBirthSeed(c0.toUint(), seedU, epochU).bitXor(axisU))
        .mul(2)
        .sub(1);
      const h1 = pcg01(mixBirthSeed(c1.toUint(), seedU, epochU).bitXor(axisU))
        .mul(2)
        .sub(1);
      return mix(h0, h1, f);
    }
  );

  /** Non-uniform PCG slot width for pulse chaos v2 (§11), iu = 0-based uint. */
  const pulseWidth = (
    iu: ShaderNodeObject<Node>,
    seedU: ShaderNodeObject<Node>,
    epochU: ShaderNodeObject<Node>,
    wXor: ShaderNodeObject<Node>
  ): ShaderNodeObject<Node> =>
    pcg01(mixBirthSeed(iu.add(tuint(1)), seedU, epochU).bitXor(wXor))
      .mul(0.7)
      .add(0.3);

  /**
   * Non-linear asymmetric chaos model (§11 v2, "pulse"). Each non-uniform
   * slot takes one of three classes from its hash (§11 doc):
   *
   * ```text
   * cls < 0.2  ->  gap      0 (stop)
   * cls < 0.4  ->  plateau  signed constant (bzzz)
   * else       ->  spike    15% linear rise, exponential slow tail (zap)
   * ```
   *
   * plus ≈25% "chaos degradation": gain x0.5 and half slot width. The
   * `t == 1` boundary falls into the last slot. Scalar twin: `pulseOffset`
   * in `electric-arc-math.ts` (same integer mixing, same classes).
   */
  const coarsePulseOf = Fn(
    ({
      tN,
      axisU,
    }: {
      tN: ShaderNodeObject<Node>;
      axisU: ShaderNodeObject<Node>;
    }) => {
      const seedU = nSeed.toUint();
      const epochU = nEpoch.toUint();
      const ax = axisU.toUint().sub(tuint(1)); // 0 / 1 channel index
      const wXor = tuint(11).mul(ax.add(tuint(1))); // 11 / 22
      const lvlXor = ax.add(tuint(2));
      const clsXor = ax.add(tuint(5));
      const degXor = ax.add(tuint(9));
      const nU = nKnots.toUint();

      // pass 1: normalized total slot width
      const total = float(0).toVar();
      Loop(21, ({ i }: { i: ShaderNodeObject<Node> }) => {
        If(i.lessThan(nKnots), () => {
          total.addAssign(pulseWidth(i.toUint(), seedU, epochU, wXor));
        });
      });
      const inv = float(1).div(total.max(float(1e-6)));

      // pass 2: locate the slot and its normalized position (full width)
      const acc0 = float(0).toVar();
      const cellU = tuint(0).toVar();
      const fN = float(0).toVar();
      const found = float(0).toVar();
      Loop(21, ({ i }: { i: ShaderNodeObject<Node> }) => {
        If(i.lessThan(nKnots).and(found.lessThan(0.5)), () => {
          const wN = pulseWidth(i.toUint(), seedU, epochU, wXor).mul(inv);
          const acc1 = acc0.add(wN);
          If(tN.lessThan(acc1), () => {
            found.assign(1);
            cellU.assign(i.toUint());
            fN.assign(
              tN
                .sub(acc0)
                .div(wN.max(float(1e-6)))
                .max(float(0))
                .min(float(1))
            );
          });
          acc0.assign(acc1);
        });
      });
      If(tN.equal(float(1)), () => {
        // t == 1: the last slot, position 1, never dead
        cellU.assign(nU.sub(tuint(1)));
        fN.assign(float(1));
      });

      const deg01 = pcg01(
        mixBirthSeed(cellU.add(tuint(900)), seedU, epochU).bitXor(degXor)
      );
      const degOn = deg01.lessThan(float(0.25));
      const slotFrac = degOn.select(float(0.5), float(1));

      // position inside the (possibly half-width) slot + dead-tail mask
      const fS = fN.div(slotFrac).min(float(1));
      const dead = fN.greaterThan(slotFrac).and(fN.lessThan(float(1)));

      const level = pcg01(mixBirthSeed(cellU, seedU, epochU).bitXor(lvlXor))
        .mul(2)
        .sub(1);
      const cls = pcg01(
        mixBirthSeed(cellU.add(tuint(128)), seedU, epochU).bitXor(clsXor)
      );

      const riseV = fS.div(float(0.15)).min(float(1));
      const tailV = exp(float(-3).mul(fS.sub(float(0.15))));
      const spike = level.mul(fS.lessThan(float(0.15)).select(riseV, tailV));

      const cls3 = cls
        .lessThan(float(0.2))
        .select(float(0), cls.lessThan(float(0.4)).select(level, spike));
      const val = dead.select(float(0), cls3);
      return degOn.select(val.mul(float(0.5)), val);
    }
  );

  /**
   * Third chaos model (§11, "organic"): smooth cosine-eased kinks with a
   * random per-cell hold (pause) at the arrived level. Scalar twin:
   * `organicOffset` (same integer mixing and hold window 0.45..0.85).
   */
  const coarseOrganicOf = Fn(
    ({
      tN,
      axisU,
    }: {
      tN: ShaderNodeObject<Node>;
      axisU: ShaderNodeObject<Node>;
    }) => {
      const seedU = nSeed.toUint();
      const epochU = nEpoch.toUint();
      const ax = axisU.toUint().sub(tuint(1)); // 0 / 1 channel index
      const lvlXor = ax.add(tuint(2));
      const holdXor = tuint(31).mul(ax.add(tuint(1))); // 31 / 62
      const nU = nKnots.toUint();

      const cellF = tN.mul(nKnots);
      const c0f = floor(cellF);
      const f = cellF.sub(c0f);
      const c0u = c0f.toUint().min(nU);
      const c1u = c0u.add(tuint(1)).min(nU);

      const l0 = pcg01(mixBirthSeed(c0u, seedU, epochU).bitXor(lvlXor))
        .mul(2)
        .sub(1);
      const l1 = pcg01(mixBirthSeed(c1u, seedU, epochU).bitXor(lvlXor))
        .mul(2)
        .sub(1);

      const holdFrac = pcg01(
        mixBirthSeed(c0u.add(tuint(700)), seedU, epochU).bitXor(holdXor)
      )
        .mul(float(0.4))
        .add(float(0.45));

      const travel = f.div(holdFrac).min(float(1));
      const e = travel.mul(travel).mul(float(3).sub(travel.mul(float(2)))); // smoothstep
      const eased = l0.add(l1.sub(l0).mul(e));
      return f.greaterThanEqual(holdFrac).select(l1, eased);
    }
  );

  /** Active chaos model (kernel built once — structural on switch). */
  const coarseFn =
    cfg.chaosAlgorithm === 'pulse'
      ? coarsePulseOf
      : cfg.chaosAlgorithm === 'organic'
        ? coarseOrganicOf
        : coarseOf;
  const microOf = Fn(
    ({
      tN,
      ch,
    }: {
      tN: ShaderNodeObject<Node>;
      ch: ShaderNodeObject<Node>;
    }) => {
      const p = vec3(tN.mul(nMicroF), nTime, ch.mul(nSeedInv));
      return snoise3D({ v: p })
        .mul(float(0.72))
        .add(snoise3D({ v: p.mul(2.13) }).mul(float(0.28)));
    }
  );

  /** Impulse layer: single octave, detuned phase. */
  const impulseOf = Fn(
    ({ tN, ch }: { tN: ShaderNodeObject<Node>; ch: ShaderNodeObject<Node> }) =>
      snoise3D({
        v: vec3(
          tN.mul(nMicroF).mul(2.7).add(float(11.37)),
          nTime.mul(1.7).add(float(3.1)),
          ch.add(float(9.7))
        ),
      })
  );

  /** Sampled position on the main (parent) arc at normalized t (§5–§9). */
  const mainPosAt = Fn(({ tN }: { tN: ShaderNodeObject<Node> }) => {
    const base = mix(nStart, nEnd, tN);
    const sT = tN.equal(float(1)).select(float(0.9999975), tN).toVar();
    const env = pow(sin(PI.mul(sT)), nPin);

    const ou = coarseFn({ tN: sT, axisU: tuint(1) })
      .mul(float(0.68))
      .add(microOf({ tN: sT, ch: tuint(1) }).mul(float(0.24)))
      .add(impulseOf({ tN: sT, ch: tuint(1) }).mul(float(0.08)));
    const ov = coarseFn({ tN: sT, axisU: tuint(2) })
      .mul(float(0.68))
      .add(microOf({ tN: sT, ch: tuint(2) }).mul(float(0.24)))
      .add(impulseOf({ tN: sT, ch: tuint(2) }).mul(float(0.08)));

    const ampEnv = nAmp.mul(env);
    const x = base.x.add(nU.x.mul(ou).add(nV.x.mul(ov)).mul(ampEnv));
    const y = base.y.add(nU.y.mul(ou).add(nV.y.mul(ov)).mul(ampEnv));
    const z = base.z.add(nU.z.mul(ou).add(nV.z.mul(ov)).mul(ampEnv));
    return vec3(
      tN.equal(float(0)).select(nStart.x, tN.equal(float(1)).select(nEnd.x, x)),
      tN.equal(float(0)).select(nStart.y, tN.equal(float(1)).select(nEnd.y, y)),
      tN.equal(float(0)).select(nStart.z, tN.equal(float(1)).select(nEnd.z, z))
    );
  });

  /**
   * Variable-width model: magnitude of the in-plane deflection (rotated
   * tangent amount) at normalized `tN` on the main arc, 0 for a straight
   * segment and ->1 with chaos deflection. Used to thin the ribbon where
   * the filament rotates and keep it full where it is straight.
   */
  const rotMagAt = Fn(({ tN }: { tN: ShaderNodeObject<Node> }) => {
    const sT = tN.equal(float(1)).select(float(0.9999975), tN).toVar();
    const env = pow(sin(PI.mul(sT)), nPin).toVar();
    const ou = coarseFn({ tN: sT, axisU: tuint(1) })
      .mul(float(0.68))
      .add(microOf({ tN: sT, ch: tuint(1) }).mul(float(0.24)))
      .add(impulseOf({ tN: sT, ch: tuint(1) }).mul(float(0.08)))
      .mul(env);
    const ov = coarseFn({ tN: sT, axisU: tuint(2) })
      .mul(float(0.68))
      .add(microOf({ tN: sT, ch: tuint(2) }).mul(float(0.24)))
      .add(impulseOf({ tN: sT, ch: tuint(2) }).mul(float(0.08)))
      .mul(env);
    return sqrt(ou.mul(ou).add(ov.mul(ov))).min(float(1));
  });

  /** §12 kernel: one vec4 per centerline sample. */
  const kernel = Fn(() => {
    const i = instanceIndex; // uint, exact integer in [0, totalSamples)
    const iF = i.toFloat();
    const sArc = storage(arcBuffer, 'vec4', totalSamples);
    const sW = storage(widthBuffer, 'float', totalSamples);

    const isMain = iF.lessThan(float(segments));

    // main arc
    const tN = iF.div(nSegLast);
    const mainPos = mainPosAt({ tN });
    const hb = pcg01(mixBirthSeed(i, nSeed.toUint(), tuint(5)));
    const mainBrightness = nGlobalF.mul(
      float(1).add(nBvar.mul(hb.mul(2).sub(1)))
    );

    // branches (only evaluated when present)
    const rel = iF.sub(nBranchBase);
    const bN = floor(rel.div(float(ARC_BRANCH_SAMPLES)));
    const jN = rel.sub(bN.mul(float(ARC_BRANCH_SAMPLES)));
    const tB = jN.div(float(ARC_BRANCH_SAMPLES - 1));

    const bh = pcg01(mixBirthSeed(bN, nSeed.toUint(), tuint(3)));
    const active = bh.lessThan(nProb).select(float(1), float(0));

    const dir1r = pcg01(mixBirthSeed(bN, nSeed.toUint(), tuint(13)))
      .mul(2)
      .sub(1);
    const dir2r = pcg01(mixBirthSeed(bN, nSeed.toUint(), tuint(14)))
      .mul(2)
      .sub(1);
    const dir3r = pcg01(mixBirthSeed(bN, nSeed.toUint(), tuint(15)))
      .mul(2)
      .sub(1);
    const dLen = sqrt(
      dir1r.mul(dir1r).add(dir2r.mul(dir2r)).add(dir3r.mul(dir3r))
    ).max(float(1e-6));
    const dir1 = dir1r.div(dLen);
    const dir2 = dir2r.div(dLen);
    const dir3 = dir3r.div(dLen);
    const hLen = pcg01(mixBirthSeed(bN, nSeed.toUint(), tuint(16)));
    const len = mix(nLen0, nLen1, hLen).mul(active);

    const tS = floor(bh.mul(nSegLast)).div(nSegLast);
    const origin = mainPosAt({ tN: tS });
    // same quadratic curve as the CPU branch (control = d*0.5 + U*bend)
    const bend = pcg01(mixBirthSeed(bN, nSeed.toUint(), tuint(18)))
      .mul(2)
      .sub(1)
      .mul(float(0.28));
    const cX = origin.x.add(
      dir1.mul(len).add(nU.x.mul(bend).mul(len)).mul(float(0.5))
    );
    const cY = origin.y.add(
      dir2.mul(len).add(nU.y.mul(bend).mul(len)).mul(float(0.5))
    );
    const cZ = origin.z.add(
      dir3.mul(len).add(nU.z.mul(bend).mul(len)).mul(float(0.5))
    );
    const eX = origin.x.add(dir1.mul(len));
    const eY = origin.y.add(dir2.mul(len));
    const eZ = origin.z.add(dir3.mul(len));
    const om = float(1).sub(tB);
    const w0 = om.mul(om);
    const w1 = om.mul(tB).mul(2);
    const w2 = tB.mul(tB);
    const bx = origin.x.mul(w0).add(cX.mul(w1)).add(eX.mul(w2));
    const by = origin.y.mul(w0).add(cY.mul(w1)).add(eY.mul(w2));
    const bz = origin.z.mul(w0).add(cZ.mul(w1)).add(eZ.mul(w2));

    const x = isMain.select(mainPos.x, bx);
    const y = isMain.select(mainPos.y, by);
    const z = isMain.select(mainPos.z, bz);
    const bright = isMain.select(mainBrightness, active);

    sArc.element(i).assign(vec4(x, y, z, bright));

    // ── variable width: thin where the filament rotates, full when straight
    const rotMag = rotMagAt({ tN });
    const wThin = float(1).sub(float(0.45).mul(rotMag));
    const wMain = isMain.select(wThin, float(1));
    sW.element(i).assign(wMain);
  });

  // `compute` expects the INVOKED kernel node (same as the particle
  // module's `compute(emitKernel(), …)`) — passing the bare `Fn` would make
  // the node builder emit the "declared but not invoked" placeholder.
  const computeNode = compute(kernel(), totalSamples);

  let disposed = false;
  return {
    computeNode,
    arcBuffer,
    widthBuffer,
    totalSamples,
    mainCount: segments,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      arcBuffer.array = new Float32Array(0) as never;
      widthBuffer.array = new Float32Array(0) as never;
    },
    uniforms: {
      start: nStart,
      end: nEnd,
      basisU: nU,
      basisV: nV,
      time: nTime,
      pin: nPin,
      amp: nAmp,
      knots: nKnots,
      microF: nMicroF,
      epoch: nEpoch,
      seed: nSeed,
      seedInv: nSeedInv,
      brightnessVar: nBvar,
      globalFlicker: nGlobalF,
      intensity: nIntensity,
      segLast: nSegLast,
      branchBase: nBranchBase,
      branchLen0: nLen0,
      branchLen1: nLen1,
      branchProb: nProb,
    },
  };
}
