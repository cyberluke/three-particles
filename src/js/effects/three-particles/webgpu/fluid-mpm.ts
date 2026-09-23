/**
 * MLS-MPM (Moving Least Squares Material Point Method) fluid solver.
 *
 * TSL port of `matsuoka-601/webgpu-ocean` (`mls-mpm/*.wgsl` and `mls-mpm/mls-mpm.ts`):
 *   `clearGrid -> p2g_1 -> p2g_2 -> updateGrid -> g2p`, repeated
 *   {@link MLS_MPM_SUBSTEPS} times per rendered frame.
 *
 * Two adaptations to this engine:
 *   - the upstream `copyPosition` pass is a no-op: the engine's `position` /
 *     `velocity` storage attributes double as compute state and render
 *     attributes, so no second buffer has to be mirrored;
 *   - the lattice keeps the upstream fixed-point `i32` layout (a `Cell` holds
 *     `vx, vy, vz, mass`) so the 27-cell scatter stays on integer `atomicAdd`.
 *
 * Per-particle records use the engine `vec4` convention:
 *   `position.xyz`, `velocity.xyz`, and `C` as three `vec4` columns.
 *
 * @module
 */
import {
  Fn,
  If,
  atomicAdd,
  atomicLoad,
  atomicStore,
  compute,
  float,
  floor,
  instanceIndex,
  uint as tuint,
  max as tslMax,
  min as tslMin,
  pow,
  storage,
  uniform,
  vec3,
  vec4,
  type Node,
  type ShaderNodeObject,
} from 'three/tsl';
import {
  StorageBufferAttribute,
  StorageInstancedBufferAttribute,
} from 'three/webgpu';

import type { MLSMPMConfig, PassLayout } from '../types.js';

// ─── Constants ────────────────────────────────────────────────────────────────

/** `@workgroup_size(64)` of every upstream kernel. */
export const MLS_MPM_WORKGROUP_SIZE = 64;

/** Fixed-point scale shared by the encode / decode pair (`1e7` upstream). */
export const MLS_MPM_FIXED_POINT_MULTIPLIER = 1e7;

/** Simulation sub-steps executed per rendered frame. */
export const MLS_MPM_SUBSTEPS = 2;

/** Per-axis cell budget of the `64^3` lattice. */
export const MLS_MPM_MAX_GRID_DIM = 64;

/** Dambreak lattice spacing in world units. */
export const MLS_MPM_PARTICLE_SPACING = 0.65;

/** Words stored per grid cell: `(vx, vy, vz, mass)`. */
export const MLS_MPM_CELL_WORDS = 4;

/** `vec4` words stored per particle for the affine velocity matrix `C`. */
export const MLS_MPM_C_WORDS = 3;

/** Defaults lifted from the reference `main.ts` MLS-MPM parameter block. */
export const MLS_MPM_DEFAULTS = {
  stiffness: 3,
  restDensity: 4,
  dynamicViscosity: 0.1,
  dt: 0.2,
  gravity: -0.3,
  sphereSize: 1.2,
  fov: Math.PI / 4,
  minBoxSize: 1,
  boxSize: [40, 30, 60] as [number, number, number],
} as const;

/** Wall / clamp constants of the upstream `g2p` + `updateGrid` passes. */
export const MLS_MPM_WALL = {
  min: 3,
  maxOffset: 4,
  stiffness: 0.3,
  extrapolationK: 3,
  clampLower: 1,
  clampUpperOffset: 2,
  cellBorder: 2,
  cellMargin: 3,
} as const;

// ─── Pure helpers (unit-testable) ─────────────────────────────────────────────

/** `i32( f * multiplier )` — truncation, matching the WGSL `i32()` cast. */
export const encodeFixedPoint = (
  value: number,
  multiplier: number = MLS_MPM_FIXED_POINT_MULTIPLIER
): number => Math.trunc(value * multiplier);

/** `f32( encoded ) / multiplier`. */
export const decodeFixedPoint = (
  encoded: number,
  multiplier: number = MLS_MPM_FIXED_POINT_MULTIPLIER
): number => encoded / multiplier;

/** Cell counts per axis: `ceil(boxSize)`, clipped to the `64^3` budget. */
export const computeMLSMPMGridDims = (
  boxSize: readonly [number, number, number]
): [number, number, number] =>
  [
    Math.min(MLS_MPM_MAX_GRID_DIM, Math.ceil(boxSize[0])),
    Math.min(MLS_MPM_MAX_GRID_DIM, Math.ceil(boxSize[1])),
    Math.min(MLS_MPM_MAX_GRID_DIM, Math.ceil(boxSize[2])),
  ] as [number, number, number];

/** Total lattice size `nx * ny * nz` (upstream `gridCount`). */
export const computeMLSMPMGridCount = (
  boxSize: readonly [number, number, number]
): number => {
  const [nx, ny, nz] = computeMLSMPMGridDims(boxSize);
  return nx * ny * nz;
};

/** Quadratic B-spline weights `(w[-1], w[0], w[+1])` along one axis. */
export const mlsmpmQuadraticWeights = (
  diff: number
): [number, number, number] => {
  const minus = 0.5 - diff;
  const plus = 0.5 + diff;
  return [0.5 * minus * minus, 0.75 - diff * diff, 0.5 * plus * plus];
};

/** Flat index of a cell inside the `nx * ny * nz` lattice. */
export const mlsmpmCellIndex = (
  ix: number,
  iy: number,
  iz: number,
  ny: number,
  nz: number
): number => ix * ny * nz + iy * nz + iz;

/** Flat `i32` word offset of the first component of a cell. */
export const mlsmpmCellWordBase = (
  ix: number,
  iy: number,
  iz: number,
  ny: number,
  nz: number
): number => mlsmpmCellIndex(ix, iy, iz, ny, nz) * MLS_MPM_CELL_WORDS;

/**
 * CPU initial state of the dambreak, mirroring the reference `initDambreak`.
 * `position` / `velocity` use the engine `vec4` (xyz + padding) contract;
 * `coefficients` stores `C` as three identity columns.
 */
export const initMLSMPMDambreak = (
  boxSize: readonly [number, number, number],
  capacity: number,
  spacing: number = MLS_MPM_PARTICLE_SPACING,
  random: () => number = Math.random
): {
  count: number;
  position: Float32Array;
  velocity: Float32Array;
  coefficients: Float32Array;
} => {
  const slots = Math.max(1, Math.floor(capacity));
  const position = new Float32Array(slots * 4);
  const velocity = new Float32Array(slots * 4);
  const coefficients = new Float32Array(slots * MLS_MPM_C_WORDS * 4);
  const yLimit = boxSize[1] * 0.8;
  let count = 0;

  for (let y = 0; y < yLimit && count < slots; y += spacing) {
    for (let x = 3; x < boxSize[0] - 4 && count < slots; x += spacing) {
      for (let z = 3; z < boxSize[2] / 2 && count < slots; z += spacing) {
        const jitter = 2 * random();
        const base = count * 4;
        position[base] = x + jitter;
        position[base + 1] = y + jitter;
        position[base + 2] = z + jitter;
        velocity[base] = 0;
        velocity[base + 1] = 0;
        velocity[base + 2] = 0;
        const cBase = count * MLS_MPM_C_WORDS * 4;
        coefficients[cBase] = 1;
        coefficients[cBase + 5] = 1;
        coefficients[cBase + 10] = 1;
        count++;
      }
    }
  }

  return { count, position, velocity, coefficients };
};

/** Capacity actually filled by {@link initMLSMPMDambreak} (deterministic). */
export const countMLSMPMDambreak = (
  boxSize: readonly [number, number, number],
  capacity: number,
  spacing: number = MLS_MPM_PARTICLE_SPACING
): number => initMLSMPMDambreak(boxSize, capacity, spacing, () => 0).count;

// ─── Storage pool ─────────────────────────────────────────────────────────────

/** GPU storage owned by the MLS-MPM kernels. */
export type MLSMPMBuffers = {
  /** Particle position (vec4: xyz, w = padding). */
  position: StorageBufferAttribute | StorageInstancedBufferAttribute;
  /** Particle velocity (vec4: xyz, w = padding). */
  velocity: StorageBufferAttribute | StorageInstancedBufferAttribute;
  /** Affine velocity matrix `C`: three `vec4` columns per particle. */
  coefficients: StorageBufferAttribute;
  /** Fixed-point lattice: {@link MLS_MPM_CELL_WORDS} i32 words per cell. */
  cells: StorageBufferAttribute;
};

/**
 * Optional externally-owned per-particle vec4 buffers the solver should share
 * in place, so the MLS-MPM pass writes into the same storage that the base
 * modifier pipeline (and thus the FLUID render material) reads. `coefficients`
 * and `cells` always live inside the solver pool because the modifier
 * pipeline has no analogue for them.
 */
export type MLSMPMSharedBuffers = {
  position: StorageBufferAttribute | StorageInstancedBufferAttribute;
  velocity: StorageBufferAttribute | StorageInstancedBufferAttribute;
};

/** Creates the MLS-MPM storage pool (identity `C`, zeroed lattice). */
export function createMLSMPMBuffers(
  maxParticles: number,
  gridCount: number,
  shared?: MLSMPMSharedBuffers
): MLSMPMBuffers {
  const particles = Math.max(1, Math.floor(maxParticles));
  const cells = Math.max(1, Math.floor(gridCount));
  const coefficients = new Float32Array(particles * MLS_MPM_C_WORDS * 4);
  for (let i = 0; i < particles; i++) {
    const base = i * MLS_MPM_C_WORDS * 4;
    coefficients[base] = 1;
    coefficients[base + 5] = 1;
    coefficients[base + 10] = 1;
  }
  const zeroedCells = new StorageBufferAttribute(
    new Uint32Array(cells * MLS_MPM_CELL_WORDS),
    1
  );
  const freshPos = new StorageBufferAttribute(
    new Float32Array(particles * 4),
    4
  );
  const freshVel = new StorageBufferAttribute(
    new Float32Array(particles * 4),
    4
  );
  const coefficientBuffer = new StorageBufferAttribute(coefficients, 4);
  return shared
    ? {
        position: shared.position,
        velocity: shared.velocity,
        coefficients: coefficientBuffer,
        cells: zeroedCells,
      }
    : {
        position: freshPos,
        velocity: freshVel,
        coefficients: coefficientBuffer,
        cells: zeroedCells,
      };
}

// ─── Parameter resolution ─────────────────────────────────────────────────────

/** Scalar parameter block resolved from the user config (defaults applied). */
export type MLSMPMParams = {
  stiffness: number;
  restDensity: number;
  dynamicViscosity: number;
  dt: number;
  gravity: number;
  cellSize: number;
  gridSize: number;
  sphereSize: number;
  boxSize: [number, number, number];
  gridDims: [number, number, number];
  wallStiffness: number;
  extrapolationK: number;
};

/** Applies the documented MLS-MPM defaults on top of a partial config. */
export function resolveMLSMPMParams(
  config: MLSMPMConfig | undefined,
  boxSize: readonly [number, number, number]
): MLSMPMParams {
  const [nx, ny, nz] = computeMLSMPMGridDims(boxSize);
  return {
    stiffness: config?.stiffness ?? MLS_MPM_DEFAULTS.stiffness,
    restDensity: config?.restDensity ?? MLS_MPM_DEFAULTS.restDensity,
    dynamicViscosity:
      config?.dynamicViscosity ?? MLS_MPM_DEFAULTS.dynamicViscosity,
    dt: config?.dt ?? MLS_MPM_DEFAULTS.dt,
    gravity: config?.gravity ?? MLS_MPM_DEFAULTS.gravity,
    cellSize: config?.cellSize ?? 1,
    gridSize: config?.gridSize ?? MLS_MPM_MAX_GRID_DIM,
    sphereSize: config?.sphereSize ?? MLS_MPM_DEFAULTS.sphereSize,
    boxSize: [boxSize[0], boxSize[1], boxSize[2]],
    gridDims: [nx, ny, nz],
    wallStiffness: MLS_MPM_WALL.stiffness,
    extrapolationK: MLS_MPM_WALL.extrapolationK,
  };
}

// ─── Kernel building blocks ──────────────────────────────────────────────────

type KernelContext = {
  count: number;
  gridCount: number;
  ny: number;
  nz: number;
  sPos: ShaderNodeObject<Node>;
  sVel: ShaderNodeObject<Node>;
  sC: ShaderNodeObject<Node>;
  sCells: ShaderNodeObject<Node>;
  fp: ShaderNodeObject<Node>;
  k: ShaderNodeObject<Node>;
  d0: ShaderNodeObject<Node>;
  mu: ShaderNodeObject<Node>;
  dt: ShaderNodeObject<Node>;
  gravity: ShaderNodeObject<Node>;
  wallStiffness: ShaderNodeObject<Node>;
  extrapolationK: ShaderNodeObject<Node>;
  /** Animated `z` squeeze ratio, written once per frame by the host. */
  uBoxWidthRatio: ShaderNodeObject<Node>;
  /** Animated real box size; drives wall clamps like `changeBoxSize()`. */
  rx: ShaderNodeObject<Node>;
  ry: ShaderNodeObject<Node>;
  rz: ShaderNodeObject<Node>;
};

/** Offsets of the `3 x 3 x 3` quadratic stencil, in `(gx, gy, gz)` order. */
const STENCIL: ReadonlyArray<readonly [number, number, number]> =
  /* 27 triples */ Array.from(
    { length: 27 },
    (_, n) =>
      [Math.floor(n / 9), Math.floor(n / 3) % 3, n % 3] as [
        number,
        number,
        number,
      ]
  );

/** Quadratic weight triple of one axis as three scalar nodes. */
const axisWeights = (
  diff: ShaderNodeObject<Node>
): ShaderNodeObject<Node>[] => {
  const minus = float(0.5).sub(diff);
  const plus = float(0.5).add(diff);
  return [
    minus.mul(minus).mul(float(0.5)),
    float(0.75).sub(diff.mul(diff)),
    plus.mul(plus).mul(float(0.5)),
  ];
};

/** `(cellIdx, wx, wy, wz)` — the per-particle stencil frame. */
const stencilFrame = (
  pos: ShaderNodeObject<Node>
): {
  cellIdx: ShaderNodeObject<Node>;
  wx: ShaderNodeObject<Node>[];
  wy: ShaderNodeObject<Node>[];
  wz: ShaderNodeObject<Node>[];
} => {
  const cellIdx = floor(pos);
  const diff = pos.sub(cellIdx.add(float(0.5)));
  return {
    cellIdx,
    wx: axisWeights(diff.x),
    wy: axisWeights(diff.y),
    wz: axisWeights(diff.z),
  };
};

/** Stencil cell coordinates and its separable weight. */
const stencilCell = (
  cellIdx: ShaderNodeObject<Node>,
  offset: readonly [number, number, number],
  wx: ShaderNodeObject<Node>[],
  wy: ShaderNodeObject<Node>[],
  wz: ShaderNodeObject<Node>[]
): {
  cx: ShaderNodeObject<Node>;
  cy: ShaderNodeObject<Node>;
  cz: ShaderNodeObject<Node>;
  weight: ShaderNodeObject<Node>;
} => {
  const [gx, gy, gz] = offset;
  return {
    cx: cellIdx.x.add(float(gx - 1)),
    cy: cellIdx.y.add(float(gy - 1)),
    cz: cellIdx.z.add(float(gz - 1)),
    weight: wx[gx].mul(wy[gy]).mul(wz[gz]),
  };
};

/** Flat word offset of a stencil cell (4 fixed-point words per cell). */
const wordBase = (
  cx: ShaderNodeObject<Node>,
  cy: ShaderNodeObject<Node>,
  cz: ShaderNodeObject<Node>,
  ctx: KernelContext
): ShaderNodeObject<Node> =>
  cx
    .mul(float(ctx.ny * ctx.nz))
    .add(cy.mul(float(ctx.nz)))
    .add(cz)
    .mul(float(MLS_MPM_CELL_WORDS));

/** Cell-centre distance `(cell + 0.5) - position`. */
const cellDistance = (
  cx: ShaderNodeObject<Node>,
  cy: ShaderNodeObject<Node>,
  cz: ShaderNodeObject<Node>,
  pos: ShaderNodeObject<Node>
): ShaderNodeObject<Node> =>
  vec3(cx.add(0.5).sub(pos.x), cy.add(0.5).sub(pos.y), cz.add(0.5).sub(pos.z));

/** Fixed-point decode: two's-complement `u32` word -> signed f32. */
const decodeWord = (
  word: ShaderNodeObject<Node>,
  fp: ShaderNodeObject<Node>
): ShaderNodeObject<Node> => {
  const value = float(word);
  return value
    .lessThan(float(2147483648))
    .select(value, value.sub(float(4294967296)))
    .div(fp);
};

/** Decodes one lattice word (fixed-point -> f32). */
const loadWord = (
  ctx: KernelContext,
  base: ShaderNodeObject<Node>,
  component: number
): ShaderNodeObject<Node> =>
  decodeWord(atomicLoad(ctx.sCells.element(base.add(component))), ctx.fp);

/** Encodes and accumulates one scalar into a lattice word. */
const addWord = (
  ctx: KernelContext,
  base: ShaderNodeObject<Node>,
  component: number,
  value: ShaderNodeObject<Node>
): void => {
  atomicAdd(ctx.sCells.element(base.add(component)), tuint(value.mul(ctx.fp)));
};

// ─── Kernels ─────────────────────────────────────────────────────────────────

/** `clearGrid`: zero the four fixed-point words of every cell. */
const createClearGridKernel = (ctx: KernelContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.gridCount)), () => {
      const base = i.mul(float(MLS_MPM_CELL_WORDS));
      atomicStore(ctx.sCells.element(base), tuint(0));
      atomicStore(ctx.sCells.element(base.add(1)), tuint(0));
      atomicStore(ctx.sCells.element(base.add(2)), tuint(0));
      atomicStore(ctx.sCells.element(base.add(3)), tuint(0));
    });
  });

/** `p2g_1`: scatter mass and momentum `w * (v + C * d)`. */
const createP2G1Kernel = (ctx: KernelContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.count)), () => {
      const pos = ctx.sPos.element(i).xyz.toVar();
      const vel = ctx.sVel.element(i).xyz.toVar();
      const frame = stencilFrame(pos);
      const cBase = i.mul(float(MLS_MPM_C_WORDS));
      // `C` is column-major: column j lives in `C[i * 3 + j].xyz`.
      const c0 = ctx.sC.element(cBase).xyz;
      const c1 = ctx.sC.element(cBase.add(1)).xyz;
      const c2 = ctx.sC.element(cBase.add(2)).xyz;

      for (const offset of STENCIL) {
        const cell = stencilCell(
          frame.cellIdx,
          offset,
          frame.wx,
          frame.wy,
          frame.wz
        );
        const d = cellDistance(cell.cx, cell.cy, cell.cz, pos);
        const q = c0.mul(d.x).add(c1.mul(d.y)).add(c2.mul(d.z));
        const add = vel.add(q).mul(cell.weight);
        const base = wordBase(cell.cx, cell.cy, cell.cz, ctx);
        addWord(ctx, base, 0, add.x);
        addWord(ctx, base, 1, add.y);
        addWord(ctx, base, 2, add.z);
        addWord(ctx, base, 3, cell.weight);
      }
    });
  });

/** `p2g_2`: pressure + viscosity momentum (upstream omits the `C^T` term). */
const createP2G2Kernel = (ctx: KernelContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.count)), () => {
      const pos = ctx.sPos.element(i).xyz.toVar();
      const frame = stencilFrame(pos);
      const cBase = i.mul(float(MLS_MPM_C_WORDS));
      const c0 = ctx.sC.element(cBase).xyz;
      const c1 = ctx.sC.element(cBase.add(1)).xyz;
      const c2 = ctx.sC.element(cBase.add(2)).xyz;

      const density = float(0).toVar();
      for (const offset of STENCIL) {
        const cell = stencilCell(
          frame.cellIdx,
          offset,
          frame.wx,
          frame.wy,
          frame.wz
        );
        const mass = loadWord(ctx, wordBase(cell.cx, cell.cy, cell.cz, ctx), 3);
        density.assign(density.add(mass.mul(cell.weight)));
      }

      const volume = float(1).div(density);
      const pressure = tslMax(
        ctx.k.mul(pow(density.div(ctx.d0), float(5.0)).sub(float(1.0))),
        float(0.0)
      );

      // `stress = diag(-p) + mu * (C + C^T)`, still column-major.
      const negP = float(-1).mul(pressure);
      const s0 = ctx.mu
        .mul(c0.add(vec3(c0.x, c1.x, c2.x)))
        .add(vec3(negP, float(0), float(0)));
      const s1 = ctx.mu
        .mul(c1.add(vec3(c0.y, c1.y, c2.y)))
        .add(vec3(float(0), negP, float(0)));
      const s2 = ctx.mu
        .mul(c2.add(vec3(c0.z, c1.z, c2.z)))
        .add(vec3(float(0), float(0), negP));

      const eq16 = float(-4).mul(volume).mul(ctx.dt);

      for (const offset of STENCIL) {
        const cell = stencilCell(
          frame.cellIdx,
          offset,
          frame.wx,
          frame.wy,
          frame.wz
        );
        const d = cellDistance(cell.cx, cell.cy, cell.cz, pos);
        // `momentum = (-dt * volume * 4 * stress * w) * d`
        const momentum = s0
          .mul(d.x)
          .add(s1.mul(d.y))
          .add(s2.mul(d.z))
          .mul(cell.weight)
          .mul(eq16);
        const base = wordBase(cell.cx, cell.cy, cell.cz, ctx);
        addWord(ctx, base, 0, momentum.x);
        addWord(ctx, base, 1, momentum.y);
        addWord(ctx, base, 2, momentum.z);
      }
    });
  });

/** `updateGrid`: normalise momentum, integrate gravity, zero the walls. */
const createUpdateGridKernel = (ctx: KernelContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.gridCount)), () => {
      const base = i.mul(float(MLS_MPM_CELL_WORDS));
      const mass = decodeWord(
        atomicLoad(ctx.sCells.element(base.add(3))),
        ctx.fp
      );
      If(mass.greaterThan(float(0)), () => {
        const invMass = float(1).div(mass);
        const vx = loadWord(ctx, base, 0).mul(invMass);
        const vy = loadWord(ctx, base, 1)
          .mul(invMass)
          .add(ctx.gravity.mul(ctx.dt));
        const vz = loadWord(ctx, base, 2).mul(invMass);
        atomicStore(ctx.sCells.element(base), tuint(vx.mul(ctx.fp)));
        atomicStore(ctx.sCells.element(base.add(1)), tuint(vy.mul(ctx.fp)));
        atomicStore(ctx.sCells.element(base.add(2)), tuint(vz.mul(ctx.fp)));

        // Integer lattice decomposition of the flat cell index.
        const fi = float(i);
        const iz = fi.mod(float(ctx.nz));
        const iy = fi.div(float(ctx.nz)).floor().mod(float(ctx.ny));
        const ix = fi.div(float(ctx.ny * ctx.nz)).floor();
        zeroWallCell(ctx, base, ix, ctx.rx);
        zeroWallCell(ctx, base.add(1), iy, ctx.ry);
        zeroWallCell(ctx, base.add(2), iz, ctx.rz);
      });
    });
  });

/** Zeroes an out-of-range wall velocity: `n < 2 || n > ceil(boxAxis) - 3`. */
const zeroWallCell = (
  ctx: KernelContext,
  base: ShaderNodeObject<Node>,
  index: ShaderNodeObject<Node>,
  boxAxis: ShaderNodeObject<Node>
): void => {
  If(index.lessThan(float(MLS_MPM_WALL.cellBorder)), () => {
    atomicStore(ctx.sCells.element(base), tuint(0));
  });
  If(index.greaterThan(boxAxis.sub(float(MLS_MPM_WALL.cellMargin))), () => {
    atomicStore(ctx.sCells.element(base), tuint(0));
  });
};

/** `g2p`: gather velocity, refit `C`, integrate, clamp, apply wall forces. */
const createG2PKernel = (ctx: KernelContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.count)), () => {
      const pos = ctx.sPos.element(i).xyz.toVar();
      const frame = stencilFrame(pos);
      const cBase = i.mul(float(MLS_MPM_C_WORDS));
      const vel = vec3(float(0), float(0), float(0)).toVar();
      const b0 = vec3(float(0), float(0), float(0)).toVar();
      const b1 = vec3(float(0), float(0), float(0)).toVar();
      const b2 = vec3(float(0), float(0), float(0)).toVar();

      for (const offset of STENCIL) {
        const cell = stencilCell(
          frame.cellIdx,
          offset,
          frame.wx,
          frame.wy,
          frame.wz
        );
        const d = cellDistance(cell.cx, cell.cy, cell.cz, pos);
        const base = wordBase(cell.cx, cell.cy, cell.cz, ctx);
        const gvx = loadWord(ctx, base, 0).mul(cell.weight);
        const gvy = loadWord(ctx, base, 1).mul(cell.weight);
        const gvz = loadWord(ctx, base, 2).mul(cell.weight);
        vel.assign(vel.add(vec3(gvx, gvy, gvz)));
        // `B += outer(v_w, d)` (columns of `B`).
        b0.assign(b0.add(vec3(gvx.mul(d.x), gvy.mul(d.x), gvz.mul(d.x))));
        b1.assign(b1.add(vec3(gvx.mul(d.y), gvy.mul(d.y), gvz.mul(d.y))));
        b2.assign(b2.add(vec3(gvx.mul(d.z), gvy.mul(d.z), gvz.mul(d.z))));
      }

      const four = float(4.0);
      ctx.sC.element(cBase).assign(vec4(b0.mul(four), float(0)));
      ctx.sC.element(cBase.add(1)).assign(vec4(b1.mul(four), float(0)));
      ctx.sC.element(cBase.add(2)).assign(vec4(b2.mul(four), float(0)));

      const next = pos.add(vel.mul(ctx.dt));
      const lower = float(MLS_MPM_WALL.clampLower);
      const clamped = vec3(
        tslMin(
          tslMax(next.x, lower),
          ctx.rx.sub(float(MLS_MPM_WALL.clampUpperOffset))
        ),
        tslMin(
          tslMax(next.y, lower),
          ctx.ry.sub(float(MLS_MPM_WALL.clampUpperOffset))
        ),
        tslMin(
          tslMax(next.z, lower),
          ctx.rz.sub(float(MLS_MPM_WALL.clampUpperOffset))
        )
      );
      pos.assign(clamped);
      ctx.sPos.element(i).assign(vec4(clamped, float(0)));

      // Wall penalty driven by the extrapolated next-step position.
      const step = ctx.dt.mul(ctx.extrapolationK);
      const ex = clamped.add(vel.mul(step));
      const minWall = float(MLS_MPM_WALL.min);
      const maxOffset = float(MLS_MPM_WALL.maxOffset);
      const newVel = vel.toVar();
      If(ex.x.lessThan(minWall), () => {
        newVel.x.addAssign(ctx.wallStiffness.mul(minWall.sub(ex.x)));
      });
      If(ex.x.greaterThan(ctx.rx.sub(maxOffset)), () => {
        newVel.x.addAssign(
          ctx.wallStiffness.mul(ctx.rx.sub(maxOffset).sub(ex.x))
        );
      });
      If(ex.y.lessThan(minWall), () => {
        newVel.y.addAssign(ctx.wallStiffness.mul(minWall.sub(ex.y)));
      });
      If(ex.y.greaterThan(ctx.ry.sub(maxOffset)), () => {
        newVel.y.addAssign(
          ctx.wallStiffness.mul(ctx.ry.sub(maxOffset).sub(ex.y))
        );
      });
      If(ex.z.lessThan(minWall), () => {
        newVel.z.addAssign(ctx.wallStiffness.mul(minWall.sub(ex.z)));
      });
      If(ex.z.greaterThan(ctx.rz.sub(maxOffset)), () => {
        newVel.z.addAssign(
          ctx.wallStiffness.mul(ctx.rz.sub(maxOffset).sub(ex.z))
        );
      });
      ctx.sVel.element(i).assign(vec4(newVel, float(0)));
    });
  });

// ─── Pipeline builder ────────────────────────────────────────────────────────

/** MLS-MPM compute pipeline handle (dispatch order = `computeNodes`). */
export type MLSMPMPipeline = {
  /** Every kernel in strict dispatch order. */
  computeNodes: ReturnType<typeof compute>[];
  /** Semantic pass names, aligned with {@link MLSMPMPipeline.computeNodes}. */
  passNames: string[];
  /** Real per-pass storage / uniform budgets. */
  passLayouts: PassLayout[];
  /** GPU storage owned by the solver. */
  buffers: MLSMPMBuffers;
  /** Lattice size `nx * ny * nz`. */
  gridCount: number;
  /** Particle capacity of the pool. */
  numParticles: number;
  /** Host-written scalars (`boxWidthRatio` = `z` squeeze of the box). */
  uniforms: { boxWidthRatio: { value: number } };
};

/** Per-pass accounting identical to the modifier kernels (`<= 8` storages). */
const layout = (
  name: string,
  storages: Array<ShaderNodeObject<Node>>,
  uniforms: Array<ShaderNodeObject<Node>>
): PassLayout => ({
  name,
  storageBindings: storages.length,
  uniformBindings: uniforms.length,
});

/**
 * Builds the MLS-MPM compute pipeline.
 *
 * @param maxParticles - Particle capacity (dambreak lattice count).
 * @param params - Resolved scalar parameter block.
 * @param realBox - Animated box size; drives the wall clamps exactly like the
 *   reference `changeBoxSize()` (defaults to the initial box).
 * @param shared - Optional external position / velocity storage to write
 *   in-place (so the base modifier pipeline + render material see the exact
 *   same buffer the solver updates).
 * @returns Pipeline with kernels in strict dispatch order.
 */
export function createMLSMPMPipeline(
  maxParticles: number,
  params: MLSMPMParams,
  realBox?: readonly [number, number, number],
  shared?: MLSMPMSharedBuffers
): MLSMPMPipeline {
  const count = Math.max(1, Math.floor(maxParticles));
  const [nx, ny, nz] = params.gridDims;
  const gridCount = nx * ny * nz;
  const box = realBox ?? params.boxSize;
  const buffers = createMLSMPMBuffers(count, gridCount, shared);
  // Live `z` squeeze of the simulation box (`changeBoxSize` in the reference).
  const uBoxWidthRatio = uniform(
    params.boxSize[2] > 0 ? box[2] / params.boxSize[2] : 1
  );

  const sPos = storage(buffers.position, 'vec4', count);
  const sVel = storage(buffers.velocity, 'vec4', count);
  const sC = storage(buffers.coefficients, 'vec4', count * MLS_MPM_C_WORDS);
  const sCells = storage(
    buffers.cells,
    'uint',
    gridCount * MLS_MPM_CELL_WORDS
  ).toAtomic();

  const ctx: KernelContext = {
    count,
    gridCount,
    ny,
    nz,
    sPos,
    sVel,
    sC,
    sCells,
    fp: float(MLS_MPM_FIXED_POINT_MULTIPLIER),
    k: float(params.stiffness),
    d0: float(params.restDensity),
    mu: float(params.dynamicViscosity),
    dt: float(params.dt),
    gravity: float(params.gravity),
    wallStiffness: float(params.wallStiffness),
    extrapolationK: float(params.extrapolationK),
    uBoxWidthRatio,
    rx: float(box[0]),
    ry: float(box[1]),
    // Animated `z` extent = init extent * `uBoxWidthRatio` (`changeBoxSize`).
    rz: float(params.boxSize[2]).mul(uBoxWidthRatio),
  };

  const clearGrid = createClearGridKernel(ctx);
  const p2g1 = createP2G1Kernel(ctx);
  const p2g2 = createP2G2Kernel(ctx);
  const updateGrid = createUpdateGridKernel(ctx);
  const g2p = createG2PKernel(ctx);

  const passNames: string[] = [];
  const computeNodes: ReturnType<typeof compute>[] = [];
  const push = (name: string, node: ReturnType<typeof compute>): void => {
    passNames.push(name);
    computeNodes.push(node);
  };
  for (let step = 0; step < MLS_MPM_SUBSTEPS; step++) {
    const suffix = MLS_MPM_SUBSTEPS > 1 ? `_${step + 1}` : '';
    push(`clearGrid${suffix}`, compute(clearGrid(), gridCount));
    push(`p2g1${suffix}`, compute(p2g1(), count));
    push(`p2g2${suffix}`, compute(p2g2(), count));
    push(`updateGrid${suffix}`, compute(updateGrid(), gridCount));
    push(`g2p${suffix}`, compute(g2p(), count));
  }

  const pool = [sPos, sVel, sC, sCells];
  return {
    computeNodes,
    passNames,
    passLayouts: passNames.map((name) => layout(name, pool, [])),
    buffers,
    gridCount,
    numParticles: count,
    uniforms: { boxWidthRatio: uBoxWidthRatio },
  };
}
