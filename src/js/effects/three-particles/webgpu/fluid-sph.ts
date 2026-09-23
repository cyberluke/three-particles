/**
 * SPH (Smoothed Particle Hydrodynamics) fluid solver with fixed-radius
 * neighbour search and double-density relaxation.
 *
 * TSL port of `matsuoka-601/webgpu-ocean` (`sph/*.wgsl` and `sph/sph.ts`):
 *   `gridClear -> gridBuild -> prefixSum(3 passes) -> reorder(x2) ->
 *    density -> reorder(x2) -> force -> integrate`, iterated
 *   {@link SPH_SUBSTEPS} times per rendered frame.
 *
 * Adaptations to this engine:
 *   - the upstream 64-byte per-particle record maps onto three `vec4`
 *     storage attributes: `position (xyz, w = nearDensity)`,
 *     `velocity (xyz)`, `forceDensity (xyz = force, w = density)`;
 *   - `webgpu-radix-sort`'s `PrefixSumKernel` becomes three TSL passes
 *     (per-block partials, single-workgroup block scan, apply), because the
 *     engine owns every binding through `StorageBufferAttribute`;
 *   - the 27-cell neighbour slab is expressed with clamped per-axis extents
 *     (`min(coord, grids - coord - 1)`), i.e. the same limits the upstream
 *     nested loops use;
 *   - the upstream `copyPosition` pass is a no-op (state and render
 *     attributes are the same buffers).
 *
 * @module
 */
import {
  Fn,
  If,
  Loop,
  add,
  atomicAdd,
  atomicLoad,
  atomicStore,
  compute,
  dot,
  float,
  floor,
  instanceIndex,
  invocationLocalIndex,
  length,
  min as tslMin,
  normalize,
  sqrt,
  storage,
  sub,
  uint,
  uniform,
  vec3,
  vec4,
  workgroupArray,
  workgroupBarrier,
  type Node,
  type ShaderNodeObject,
} from 'three/tsl';
import { Vector4 } from 'three';
import {
  StorageBufferAttribute,
  StorageInstancedBufferAttribute,
} from 'three/webgpu';

import type { PassLayout, SPHConfig } from '../types.js';

// ─── Constants ────────────────────────────────────────────────────────────────

/** `@workgroup_size(64)` of every upstream kernel. */
export const SPH_WORKGROUP_SIZE = 64;

/** Simulation sub-steps executed per rendered frame. */
export const SPH_SUBSTEPS = 2;

/** Neighbourhood search radius `h` of the reference implementation. */
export const SPH_DEFAULT_KERNEL_RADIUS = 0.07;

/** Cell size = `1.0 * kernelRadius` (reference `cellSize`). */
export const SPH_CELL_SIZE_FACTOR = 1.0;

/** Guard band around the simulation volume, in cell units. */
export const SPH_SENTINEL_CELLS = 4;

/** Half-extent of the maximal simulation box per axis. */
export const SPH_MAX_HALF_BOX = 2.0;

/** Wall penalty stiffness of the upstream `integrate` pass. */
export const SPH_WALL_STIFFNESS = 8000;

/** Lower bound of the upstream `1e-64 < r2` guard. */
export const SPH_R2_EPSILON = 1e-64;

/** Lanes per workgroup for the prefix-sum block scan. */
export const SPH_SCAN_CHUNK = SPH_WORKGROUP_SIZE;

/** Hillis-Steele doubling stages for {@link SPH_SCAN_CHUNK} lanes. */
export const SPH_SCAN_STAGES = 6;

/** Slab radius of the fixed-cell-count neighbour search (27 cells). */
export const SPH_SLAB_RADIUS = 1;

/** Defaults lifted from the reference `sph.ts` parameter block. */
export const SPH_DEFAULTS = {
  kernelRadius: SPH_DEFAULT_KERNEL_RADIUS,
  mass: 1,
  restDensity: 15000,
  stiffness: 20,
  nearStiffness: 1,
  viscosity: 100,
  dt: 0.006,
  gravity: -9.8,
  sphereSize: 0.08,
  fov: Math.PI / 4,
  halfBoxSize: [1, 2, 1] as [number, number, number],
} as const;

/** Dambreak lattice stride, in units of `kernelRadius`. */
export const SPH_LATTICE_FACTOR = 0.5;

/** Fill margin of the reference dambreak lattice. */
export const SPH_LATTICE_MARGIN = 0.95;

// ─── Pure helpers (unit-testable) ─────────────────────────────────────────────

/** Cell budget per axis: `ceil((2 * halfMax + sentinel) / cellSize)`. */
export const computeSPHGridDims = (
  kernelRadius: number = SPH_DEFAULT_KERNEL_RADIUS,
  halfMax: number = SPH_MAX_HALF_BOX
): [number, number, number] => {
  const cellSize = kernelRadius * SPH_CELL_SIZE_FACTOR;
  const dims = Math.ceil(
    (2 * halfMax + SPH_SENTINEL_CELLS * cellSize) / cellSize
  );
  return [dims, dims, dims];
};

/** Total lattice size `xGrids * yGrids * zGrids` (upstream `gridCount`). */
export const computeSPHGridCount = (
  kernelRadius?: number,
  halfMax?: number
): number => {
  const [x, y, z] = computeSPHGridDims(kernelRadius, halfMax);
  return x * y * z;
};

/** Lattice offset `sentinel / 2` used by the cell-coordinate mapping. */
export const computeSPHOffset = (
  kernelRadius: number = SPH_DEFAULT_KERNEL_RADIUS
): number => (SPH_SENTINEL_CELLS * kernelRadius * SPH_CELL_SIZE_FACTOR) / 2;

/** Neighbour-cell flat index `xi + yi * xGrids + zi * xGrids * yGrids`. */
export const sphCellId = (
  xi: number,
  yi: number,
  zi: number,
  xGrids: number,
  yGrids: number
): number => xi + yi * xGrids + zi * xGrids * yGrids;

/** Blocks covered by the per-block partial-sum pass. */
export const computeSPHScanBlocks = (cellCount: number): number =>
  Math.ceil(Math.max(1, cellCount) / SPH_SCAN_CHUNK);

/** Strided entries each lane folds in the block scan. */
export const computeSPHScanInnerSteps = (blockCount: number): number =>
  Math.ceil(Math.max(1, blockCount) / SPH_SCAN_CHUNK);

/** `r^2, r^5, r^6, r^9` of the smoothing radius. */
export type SPHKernelPowers = {
  pow2: number;
  pow5: number;
  pow6: number;
  pow9: number;
};

/** Kernel-radius powers shared by the density / force kernels. */
export const sphKernelPowers = (
  kernelRadius: number = SPH_DEFAULT_KERNEL_RADIUS
): SPHKernelPowers => ({
  pow2: Math.pow(kernelRadius, 2),
  pow5: Math.pow(kernelRadius, 5),
  pow6: Math.pow(kernelRadius, 6),
  pow9: Math.pow(kernelRadius, 9),
});

/** Density-kernel scale `315 / (64 * pi * r^9)`. */
export const sphDensityKernelScale = (powers: SPHKernelPowers): number =>
  315 / (64 * Math.PI * powers.pow9);

/** Near-density-kernel scale `15 / (pi * r^6)`. */
export const sphNearDensityKernelScale = (powers: SPHKernelPowers): number =>
  15 / (Math.PI * powers.pow6);

/** Density-kernel gradient scale `45 / (pi * r^6)`. */
export const sphDensityGradientScale = (powers: SPHKernelPowers): number =>
  45 / (Math.PI * powers.pow6);

/** Viscosity-kernel Laplacian scale `45 / (pi * r^6)`. */
export const sphViscosityLaplacianScale = (powers: SPHKernelPowers): number =>
  45 / (Math.PI * powers.pow6);

/**
 * CPU initial state of the dambreak, mirroring the reference `initDambreak`.
 * `position.w` carries `nearDensity`; `forceDensity.w` carries `density`.
 */
export const initSPHDambreak = (
  halfBoxSize: readonly [number, number, number],
  capacity: number,
  kernelRadius: number = SPH_DEFAULT_KERNEL_RADIUS,
  random: () => number = Math.random,
  seedSphere?: { center: readonly [number, number, number]; radius: number }
): {
  count: number;
  position: Float32Array;
  velocity: Float32Array;
  forceDensity: Float32Array;
} => {
  const slots = Math.max(1, Math.floor(capacity));
  const position = new Float32Array(slots * 4);
  const velocity = new Float32Array(slots * 4);
  const forceDensity = new Float32Array(slots * 4);
  const step = SPH_LATTICE_FACTOR * kernelRadius;
  const mx = SPH_LATTICE_MARGIN * halfBoxSize[0];
  const my = SPH_LATTICE_MARGIN * halfBoxSize[1];
  const mz = SPH_LATTICE_MARGIN * halfBoxSize[2];
  const r2 = seedSphere ? seedSphere.radius * seedSphere.radius : 0;
  let count = 0;

  const inSphere = (px: number, py: number, pz: number): boolean => {
    if (!seedSphere) return true;
    const dx = px - seedSphere.center[0];
    const dy = py - seedSphere.center[1];
    const dz = pz - seedSphere.center[2];
    return dx * dx + dy * dy + dz * dz <= r2;
  };

  for (let y = -my; count < slots; y += step) {
    for (let x = -mx; x < mx && count < slots; x += step) {
      for (let z = -mz; z < 0 && count < slots; z += step) {
        const jitter = 0.001 * random();
        const px = x + jitter;
        const py = y + jitter;
        const pz = z + jitter;
        if (!inSphere(px, py, pz)) continue;
        const base = count * 4;
        position[base] = px;
        position[base + 1] = py;
        position[base + 2] = pz;
        count++;
      }
    }
  }

  return { count, position, velocity, forceDensity };
};

/** Capacity actually filled by {@link initSPHDambreak} (deterministic). */
export const countSPHDambreak = (
  halfBoxSize: readonly [number, number, number],
  capacity: number,
  kernelRadius?: number,
  seedSphere?: { center: readonly [number, number, number]; radius: number }
): number =>
  initSPHDambreak(halfBoxSize, capacity, kernelRadius, () => 0, seedSphere).count;

// ─── Storage pool ─────────────────────────────────────────────────────────────

/** GPU storage owned by the SPH kernels. */
export type SPHBuffers = {
  /** Position (xyz, w = nearDensity in-place after `density`). */
  position: StorageBufferAttribute | StorageInstancedBufferAttribute;
  /** Velocity (xyz). */
  velocity: StorageBufferAttribute | StorageInstancedBufferAttribute;
  /** Force (xyz) + density (w). */
  forceDensity: StorageBufferAttribute;
  /** Sorted copy of {@link SPHBuffers.position}. */
  sortedPosition: StorageBufferAttribute;
  /** Sorted copy of {@link SPHBuffers.velocity}. */
  sortedVelocity: StorageBufferAttribute;
  /** Sorted copy of {@link SPHBuffers.forceDensity}. */
  sortedForceDensity: StorageBufferAttribute;
  /** Raw per-cell population (atomic counter). */
  cellCounts: StorageBufferAttribute;
  /** Exclusive prefix sum over {@link SPHBuffers.cellCounts}. */
  prefixSums: StorageBufferAttribute;
  /** Per-particle in-cell offset assigned by `gridBuild`. */
  particleCellOffsets: StorageBufferAttribute;
  /** Per-block partial sums of the cell counters. */
  blockPartials: StorageBufferAttribute;
  /** Lane-local inclusive prefixes of the block scan. */
  blockInclusive: StorageBufferAttribute;
  /** Exclusive per-block offsets (consumed by the apply pass). */
  blockOffsets: StorageBufferAttribute;
};

/**
 * Optional external per-particle vec4 buffers (position + velocity) the solver
 * should use in place, so the SPH pass and the base modifier pipeline share
 * one storage backing.
 */
export type SPHSharedBuffers = {
  position: StorageBufferAttribute | StorageInstancedBufferAttribute;
  velocity: StorageBufferAttribute | StorageInstancedBufferAttribute;
};

/** Creates the SPH storage pool (integer maps + f32 block tables). */
export function createSPHBuffers(
  maxParticles: number,
  gridCount: number,
  shared?: SPHSharedBuffers
): SPHBuffers {
  const particles = Math.max(1, Math.floor(maxParticles));
  const cells = Math.max(1, Math.floor(gridCount));
  const blocks = computeSPHScanBlocks(cells + 1);
  const particleVec4 = (): StorageBufferAttribute =>
    new StorageBufferAttribute(new Float32Array(particles * 4), 4);
  const pos: StorageBufferAttribute | StorageInstancedBufferAttribute = shared
    ? shared.position
    : particleVec4();
  const vel: StorageBufferAttribute | StorageInstancedBufferAttribute = shared
    ? shared.velocity
    : particleVec4();
  return {
    position: pos,
    velocity: vel,
    forceDensity: particleVec4(),
    sortedPosition: particleVec4(),
    sortedVelocity: particleVec4(),
    sortedForceDensity: particleVec4(),
    cellCounts: new StorageBufferAttribute(new Uint32Array(cells), 1),
    prefixSums: new StorageBufferAttribute(new Float32Array(cells + 1), 1),
    particleCellOffsets: new StorageBufferAttribute(
      new Uint32Array(particles),
      1
    ),
    blockPartials: new StorageBufferAttribute(new Float32Array(blocks), 1),
    blockInclusive: new StorageBufferAttribute(new Float32Array(blocks), 1),
    blockOffsets: new StorageBufferAttribute(new Float32Array(blocks), 1),
  };
}

// ─── Parameter resolution ─────────────────────────────────────────────────────

/** Scalar parameter block resolved from the user config (defaults applied). */
export type SPHParams = {
  kernelRadius: number;
  mass: number;
  restDensity: number;
  stiffness: number;
  nearStiffness: number;
  viscosity: number;
  dt: number;
  gravity: number;
  sphereSize: number;
  halfBoxSize: [number, number, number];
  /** Half-extents used by `integrate` (follows the animated `z` squeeze). */
  realHalfBox: [number, number, number];
  cellSize: number;
  offset: number;
  gridDims: [number, number, number];
  powers: SPHKernelPowers;
  densityScale: number;
  nearDensityScale: number;
  gradientScale: number;
  laplacianScale: number;
};

/** Applies the documented SPH defaults on top of a partial config. */
export function resolveSPHParams(
  config: SPHConfig | undefined,
  halfBoxSize: readonly [number, number, number],
  realHalfBox?: readonly [number, number, number]
): SPHParams {
  const kernelRadius = config?.kernelRadius ?? SPH_DEFAULTS.kernelRadius;
  const [nx, ny, nz] = computeSPHGridDims(kernelRadius, SPH_MAX_HALF_BOX);
  const powers = sphKernelPowers(kernelRadius);
  const box = realHalfBox ?? halfBoxSize;
  return {
    kernelRadius,
    mass: config?.mass ?? SPH_DEFAULTS.mass,
    restDensity: config?.restDensity ?? SPH_DEFAULTS.restDensity,
    stiffness: config?.stiffness ?? SPH_DEFAULTS.stiffness,
    nearStiffness: config?.nearStiffness ?? SPH_DEFAULTS.nearStiffness,
    viscosity: config?.viscosity ?? SPH_DEFAULTS.viscosity,
    dt: config?.dt ?? SPH_DEFAULTS.dt,
    gravity: config?.gravity ?? SPH_DEFAULTS.gravity,
    sphereSize: config?.sphereSize ?? SPH_DEFAULTS.sphereSize,
    halfBoxSize: [halfBoxSize[0], halfBoxSize[1], halfBoxSize[2]],
    realHalfBox: [box[0], box[1], box[2]],
    cellSize: kernelRadius * SPH_CELL_SIZE_FACTOR,
    offset: computeSPHOffset(kernelRadius),
    gridDims: [nx, ny, nz],
    powers,
    densityScale: sphDensityKernelScale(powers),
    nearDensityScale: sphNearDensityKernelScale(powers),
    gradientScale: sphDensityGradientScale(powers),
    laplacianScale: sphViscosityLaplacianScale(powers),
  };
}

// ─── Kernel context and helpers ──────────────────────────────────────────────

type SPHContext = {
  count: number;
  gridCount: number;
  xGrids: number;
  yGrids: number;
  zGrids: number;
  scanBlocks: number;
  scanSteps: number;
  sPos: ShaderNodeObject<Node>;
  sVel: ShaderNodeObject<Node>;
  sForce: ShaderNodeObject<Node>;
  sSortedPos: ShaderNodeObject<Node>;
  sSortedVel: ShaderNodeObject<Node>;
  sSortedForce: ShaderNodeObject<Node>;
  sCells: ShaderNodeObject<Node>;
  sPrefix: ShaderNodeObject<Node>;
  sOffsets: ShaderNodeObject<Node>;
  sPartials: ShaderNodeObject<Node>;
  sInclusive: ShaderNodeObject<Node>;
  sBlockOffsets: ShaderNodeObject<Node>;
  cellSizeInv: ShaderNodeObject<Node>;
  offset: ShaderNodeObject<Node>;
  halfX: ShaderNodeObject<Node>;
  halfY: ShaderNodeObject<Node>;
  halfZ: ShaderNodeObject<Node>;
  /** Animated `z` squeeze ratio of the walls / lattice (`changeBoxSize`). */
  uBoxWidthRatio: ShaderNodeObject<Node>;
  /** Spherical boundary `(center.xyz, radius)`; radius `0` keeps the box. */
  uSphere: ShaderNodeObject<Node>;
  /** Pointer force `(position.xyz, radius)`; radius `0` disables it. */
  uPointerPos: ShaderNodeObject<Node>;
  /** Pointer velocity `(vx, vy, vz, 0)`. */
  uPointerVel: ShaderNodeObject<Node>;
  radius: ShaderNodeObject<Node>;
  radiusPow2: ShaderNodeObject<Node>;
  r2Epsilon: ShaderNodeObject<Node>;
  mass: ShaderNodeObject<Node>;
  stiffness: ShaderNodeObject<Node>;
  nearStiffness: ShaderNodeObject<Node>;
  restDensity: ShaderNodeObject<Node>;
  viscosity: ShaderNodeObject<Node>;
  dt: ShaderNodeObject<Node>;
  gravity: ShaderNodeObject<Node>;
  densityScale: ShaderNodeObject<Node>;
  nearDensityScale: ShaderNodeObject<Node>;
  gradientScale: ShaderNodeObject<Node>;
  laplacianScale: ShaderNodeObject<Node>;
  /** Shared storage handle list for per-pass accounting. */
  pool: Array<ShaderNodeObject<Node>>;
};

/** Lattice coordinates of a position (half box + sentinel offset). */
const cellCoords = (
  pos: ShaderNodeObject<Node>,
  ctx: SPHContext
): ShaderNodeObject<Node> =>
  floor(
    pos
      .add(vec3(ctx.halfX, ctx.halfY, ctx.halfZ))
      .add(ctx.offset)
      .mul(ctx.cellSizeInv)
  );

/** Flat cell index of {@link cellCoords} output. */
const cellIdOf = (
  coords: ShaderNodeObject<Node>,
  ctx: SPHContext
): ShaderNodeObject<Node> =>
  coords.x
    .add(coords.y.mul(float(ctx.xGrids)))
    .add(coords.z.mul(float(ctx.xGrids * ctx.yGrids)));

/** Inside-lattice test for a cell coordinate triple. */
const insideLattice = (
  coords: ShaderNodeObject<Node>,
  ctx: SPHContext
): ShaderNodeObject<Node> =>
  coords.x
    .greaterThanEqual(float(0))
    .and(coords.y.greaterThanEqual(float(0)))
    .and(coords.z.greaterThanEqual(float(0)))
    .and(coords.x.lessThan(float(ctx.xGrids)))
    .and(coords.y.lessThan(float(ctx.yGrids)))
    .and(coords.z.lessThan(float(ctx.zGrids)));

/** Clamped per-axis slab extent (`min(coord, grids - coord - 1)`). */
const slabExtent = (
  coord: ShaderNodeObject<Node>,
  grids: number
): ShaderNodeObject<Node> =>
  tslMin(coord, float(grids).sub(coord).sub(float(1)));

/** Raw cell counter at a lattice index (atomic u32 -> f32). */
const cellCountAt = (
  ctx: SPHContext,
  index: ShaderNodeObject<Node>
): ShaderNodeObject<Node> => float(atomicLoad(ctx.sCells.element(index)));

/** One slab term: `[firstId, lastId]` of a `(dx, dy, dz)` triple. */
const slabTermIds = (
  coords: ShaderNodeObject<Node>,
  dx: ShaderNodeObject<Node>,
  dy: ShaderNodeObject<Node>,
  dz: ShaderNodeObject<Node>,
  ctx: SPHContext
): {
  first: ShaderNodeObject<Node>;
  last: ShaderNodeObject<Node>;
} => {
  const xg = float(ctx.xGrids);
  const yg = float(ctx.yGrids);
  const xy = xg.mul(yg);
  const first = coords.x
    .sub(dx)
    .add(coords.y.sub(dy).mul(xg))
    .add(coords.z.sub(dz).mul(xy));
  const last = coords.x
    .add(dx)
    .add(coords.y.add(dy).mul(xg))
    .add(coords.z.add(dz).mul(xy));
  return { first, last };
};

/** Iterates the packed particles of one slab cell, guarding each axis extent. */
const forEachSlabCell = (
  coords: ShaderNodeObject<Node>,
  ex: ShaderNodeObject<Node>,
  ey: ShaderNodeObject<Node>,
  ez: ShaderNodeObject<Node>,
  ctx: SPHContext,
  body: (start: ShaderNodeObject<Node>, end: ShaderNodeObject<Node>) => void
): void => {
  for (let dz = 0; dz < 3; dz++) {
    for (let dy = 0; dy < 3; dy++) {
      for (let dx = 0; dx < 3; dx++) {
        const kx = float(dx);
        const ky = float(dy);
        const kz = float(dz);
        const ids = slabTermIds(
          coords,
          tslMin(kx, ex),
          tslMin(ky, ey),
          tslMin(kz, ez),
          ctx
        );
        const inRange = kx
          .lessThanEqual(ex)
          .and(ky.lessThanEqual(ey).and(kz.lessThanEqual(ez)));
        If(inRange, () => {
          body(
            ctx.sPrefix.element(ids.first),
            ctx.sPrefix.element(ids.last.add(1))
          );
        });
      }
    }
  }
};

// ─── Kernels ─────────────────────────────────────────────────────────────────

/** `gridClear`: zero the per-cell population counters. */
const createGridClearKernel = (ctx: SPHContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.gridCount)), () => {
      atomicStore(ctx.sCells.element(i), uint(0));
    });
  });

/** `gridBuild`: atomic per-cell population + per-particle in-cell offset. */
const createGridBuildKernel = (ctx: SPHContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.count)), () => {
      const coords = cellCoords(ctx.sPos.element(i).xyz, ctx);
      If(insideLattice(coords, ctx), () => {
        const old = atomicAdd(
          ctx.sCells.element(cellIdOf(coords, ctx)),
          uint(1)
        );
        ctx.sOffsets.element(i).assign(old);
      });
    });
  });

/** Prefix-sum pass 1: per-block partial sums over the `gridCount + 1` range. */
const createScanPartialsKernel = (ctx: SPHContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.scanBlocks)), () => {
      const total = float(0).toVar();
      for (let k = 0; k < SPH_SCAN_CHUNK; k++) {
        const idx = float(i).mul(float(SPH_SCAN_CHUNK)).add(float(k));
        If(idx.lessThan(float(ctx.gridCount)), () => {
          total.assign(total.add(cellCountAt(ctx, idx)));
        });
      }
      ctx.sPartials.element(i).assign(total);
    });
  });

/**
 * Prefix-sum pass 2: exclusive scan of the per-block partials inside ONE
 * workgroup of {@link SPH_SCAN_CHUNK} lanes. Each lane folds its strided
 * entries into a lane-local inclusive prefix, the lane totals are scanned with
 * Hillis-Steele (ping-pong workgroup arrays), and the exclusive offsets are
 * written to `blockOffsets`.
 */
const createBlockScanKernel = (ctx: SPHContext) =>
  Fn(() => {
    const lane = invocationLocalIndex;
    const steps = float(ctx.scanSteps);
    const laneTotal = float(0).toVar();

    // Fold 1: lane-local inclusive prefixes.
    for (let k = 0; k < ctx.scanSteps; k++) {
      const idx = float(lane).mul(steps).add(float(k));
      If(idx.lessThan(float(ctx.scanBlocks)), () => {
        laneTotal.assign(laneTotal.add(ctx.sPartials.element(idx)));
        ctx.sInclusive.element(idx).assign(laneTotal);
      });
    }

    const a = workgroupArray('float', SPH_SCAN_CHUNK);
    const b = workgroupArray('float', SPH_SCAN_CHUNK);
    a.element(lane).assign(laneTotal);
    workgroupBarrier();

    // Hillis-Steele inclusive scan over the lane totals.
    let src = a;
    let dst = b;
    for (let stage = 0; stage < SPH_SCAN_STAGES; stage++) {
      const s = float(Math.pow(2, stage));
      If(float(lane).greaterThanEqual(s), () => {
        dst
          .element(lane)
          .assign(
            add(
              float(src.element(lane)),
              float(src.element(float(lane).sub(s)))
            ) as never
          );
      });
      If(float(lane).lessThan(s), () => {
        dst.element(lane).assign(src.element(lane));
      });
      workgroupBarrier();
      const next = dst;
      dst = src;
      src = next;
    }

    // `src` holds the inclusive lane totals; the exclusive offset subtracts
    // this lane's own total.
    const exclusive = float(0).toVar();
    exclusive.assign(sub(float(src.element(lane)), laneTotal) as never);

    // Fold 2: `offset = laneExcl + laneInclusive - partial`.
    for (let k = 0; k < ctx.scanSteps; k++) {
      const idx = float(lane).mul(steps).add(float(k));
      If(idx.lessThan(float(ctx.scanBlocks)), () => {
        ctx.sBlockOffsets
          .element(idx)
          .assign(
            ctx.sInclusive
              .element(idx)
              .sub(ctx.sPartials.element(idx))
              .add(exclusive)
          );
      });
    }
  });

/** Prefix-sum pass 3: fold the block offsets into the exclusive cell scan. */
const createScanApplyKernel = (ctx: SPHContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.scanBlocks)), () => {
      const base = float(i).mul(float(SPH_SCAN_CHUNK));
      const blockStart = ctx.sBlockOffsets.element(i);
      const running = float(0).toVar();
      ctx.sPrefix.element(base).assign(blockStart);
      for (let k = 0; k < SPH_SCAN_CHUNK; k++) {
        const idx = base.add(float(k));
        If(idx.lessThan(float(ctx.gridCount + 1)), () => {
          If(idx.lessThan(float(ctx.gridCount)), () => {
            running.assign(running.add(cellCountAt(ctx, idx)));
          });
          ctx.sPrefix.element(idx.add(1)).assign(blockStart.add(running));
        });
      }
    });
  });

/**
 * Shared body of both reorder passes: destination slot is
 * `prefix[id + 1] - cellOffset[i] - 1` (stable, written once per particle).
 */
const reorderBody = (
  ctx: SPHContext,
  writeFields: (target: ShaderNodeObject<Node>) => void
): void => {
  const i = instanceIndex;
  If(float(i).lessThan(float(ctx.count)), () => {
    const coords = cellCoords(ctx.sPos.element(i).xyz, ctx);
    If(insideLattice(coords, ctx), () => {
      const target = ctx.sPrefix
        .element(cellIdOf(coords, ctx).add(1))
        .sub(ctx.sOffsets.element(i))
        .sub(float(1));
      If(target.lessThan(float(ctx.count)), () => {
        writeFields(target);
      });
    });
  });
};

/** `reorderParticles` for the position + velocity pair. */
const createReorderPositionKernel = (ctx: SPHContext) =>
  Fn(() => {
    reorderBody(ctx, (target) => {
      const i = instanceIndex;
      ctx.sSortedPos.element(target).assign(ctx.sPos.element(i));
      ctx.sSortedVel.element(target).assign(ctx.sVel.element(i));
    });
  });

/** `reorderParticles` for the force + density pair. */
const createReorderForceKernel = (ctx: SPHContext) =>
  Fn(() => {
    reorderBody(ctx, (target) => {
      ctx.sSortedForce
        .element(target)
        .assign(ctx.sForce.element(instanceIndex));
    });
  });

/** `computeDensity`: double-density relaxation over the clamped 27-cell slab. */
const createDensityKernel = (ctx: SPHContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.count)), () => {
      const pos = ctx.sPos.element(i).xyz;
      const coords = cellCoords(pos, ctx);
      If(insideLattice(coords, ctx), () => {
        const density = float(0).toVar();
        const nearDensity = float(0).toVar();
        const ex = slabExtent(coords.x, ctx.xGrids);
        const ey = slabExtent(coords.y, ctx.yGrids);
        const ez = slabExtent(coords.z, ctx.zGrids);

        forEachSlabCell(coords, ex, ey, ez, ctx, (start, end) => {
          Loop(end.sub(start), ({ i: j }) => {
            const other = ctx.sSortedPos.element(start.add(j)).xyz;
            const delta = pos.sub(other);
            const r2 = dot(delta, delta);
            If(r2.lessThan(ctx.radiusPow2), () => {
              const r = sqrt(r2);
              const gap = ctx.radius.sub(r);
              density.assign(
                density.add(
                  ctx.mass
                    .mul(ctx.densityScale)
                    .mul(cube(ctx.radiusPow2.sub(r2)))
                )
              );
              nearDensity.assign(
                nearDensity.add(
                  ctx.mass.mul(ctx.nearDensityScale).mul(cube(gap))
                )
              );
            });
          });
        });

        const posVec = ctx.sPos.element(i).toVar();
        posVec.w.assign(nearDensity);
        ctx.sPos.element(i).assign(posVec);
        const forceVec = ctx.sForce.element(i).toVar();
        forceVec.w.assign(density);
        ctx.sForce.element(i).assign(forceVec);
      });
    });
  });

/** `computeForce`: pressure (near + far) plus viscosity and gravity. */
const createForceKernel = (ctx: SPHContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.count)), () => {
      const posVec = ctx.sPos.element(i);
      const posI = posVec.xyz;
      const velI = ctx.sVel.element(i).xyz;
      const densityI = ctx.sForce.element(i).w;
      const nearI = posVec.w;
      const coords = cellCoords(posI, ctx);
      const fPress = vec3(float(0), float(0), float(0)).toVar();
      const fVisc = vec3(float(0), float(0), float(0)).toVar();

      If(insideLattice(coords, ctx), () => {
        const ex = slabExtent(coords.x, ctx.xGrids);
        const ey = slabExtent(coords.y, ctx.yGrids);
        const ez = slabExtent(coords.z, ctx.zGrids);

        forEachSlabCell(coords, ex, ey, ez, ctx, (start, end) => {
          Loop(end.sub(start), ({ i: j }) => {
            const slot = start.add(j);
            const densityJ = ctx.sSortedForce.element(slot).w;
            const posJ = ctx.sSortedPos.element(slot).xyz;
            const nearJ = ctx.sSortedPos.element(slot).w;
            const velJ = ctx.sSortedVel.element(slot).xyz;
            const delta = posI.sub(posJ);
            const r2 = dot(delta, delta);
            If(
              densityJ.greaterThan(float(0)).and(nearJ.greaterThan(float(0))),
              () => {
                If(
                  r2
                    .greaterThan(ctx.r2Epsilon)
                    .and(r2.lessThan(ctx.radiusPow2)),
                  () => {
                    const r = sqrt(r2);
                    const gap = ctx.radius.sub(r);
                    const pressureI = ctx.stiffness.mul(
                      densityI.sub(ctx.restDensity)
                    );
                    const pressureJ = ctx.stiffness.mul(
                      densityJ.sub(ctx.restDensity)
                    );
                    const nearPressureI = ctx.nearStiffness.mul(nearI);
                    const nearPressureJ = ctx.nearStiffness.mul(nearJ);
                    const dir = normalize(posJ.sub(posI));
                    const grad = ctx.gradientScale.mul(gap.mul(gap));
                    const laplacian = ctx.laplacianScale.mul(gap);
                    const shared = float(0.5).mul(pressureI.add(pressureJ));
                    const nearShared = float(0.5).mul(
                      nearPressureI.add(nearPressureJ)
                    );
                    // `- grad(P) * m * 0.5 * (Pi + Pj) / dj`
                    const pressTerm = dir
                      .mul(shared.mul(grad).mul(ctx.mass).div(densityJ))
                      .negate();
                    // `- grad_near(Pn) * m * 0.5 * (Pni + Pnj) / nj`
                    const nearTerm = dir
                      .mul(nearShared.mul(grad).mul(ctx.mass).div(nearJ))
                      .negate();
                    // `mu * v_lap * m / dj`
                    const viscTerm = velJ
                      .sub(velI)
                      .mul(laplacian.mul(ctx.mass).div(densityJ));
                    fPress.assign(fPress.add(pressTerm).add(nearTerm));
                    fVisc.assign(fVisc.add(viscTerm));
                  }
                );
              }
            );
          });
        });
      });

      const gravityVec = vec3(float(0), ctx.gravity.mul(densityI), float(0));
      const force = fPress.add(fVisc.mul(ctx.viscosity)).add(gravityVec);
      ctx.sForce.element(i).assign(vec4(force.x, force.y, force.z, densityI));
    });
  });

/** `integrate`: `a = f / d` + wall penalty, then the semi-implicit Euler step. */
const createIntegrateKernel = (ctx: SPHContext) =>
  Fn(() => {
    const i = instanceIndex;
    If(float(i).lessThan(float(ctx.count)), () => {
      const posVec = ctx.sPos.element(i).toVar();
      const forceVec = ctx.sForce.element(i);
      const density = forceVec.w;
      If(density.notEqual(float(0)), () => {
        const accel = vec4(
          forceVec.x.div(density),
          forceVec.y.div(density),
          forceVec.z.div(density),
          float(0)
        ).toVar();
        const wall = float(SPH_WALL_STIFFNESS);
        // Six signed distances to the (animated) box walls. The sphere domain
        // (`domain.kind === 'sphere'`) replaces them with one radial distance.
        If(ctx.uSphere.w.greaterThan(float(0)), () => {
          const rel = posVec.xyz.sub(ctx.uSphere.xyz);
          const d = tslMin(ctx.uSphere.w.sub(length(rel)), float(0));
          const n = normalize(rel);
          accel.x.addAssign(wall.mul(d).mul(n.x));
          accel.y.addAssign(wall.mul(d).mul(n.y));
          accel.z.addAssign(wall.mul(d).mul(n.z));
        });
        If(ctx.uSphere.w.equal(float(0)), () => {
          signedWall(accel.x, wall, ctx.halfX.sub(posVec.x));
          signedWall(accel.x, wall, ctx.halfX.add(posVec.x));
          signedWall(accel.y, wall, ctx.halfY.sub(posVec.y));
          signedWall(accel.y, wall, ctx.halfY.add(posVec.y));
          signedWall(accel.z, wall, ctx.halfZ.sub(posVec.z));
          signedWall(accel.z, wall, ctx.halfZ.add(posVec.z));
        });

        // Pointer force: linear-falloff transfer of the pointer velocity.
        const velVec = ctx.sVel.element(i).toVar();
        If(ctx.uPointerPos.w.greaterThan(float(0)), () => {
          const rel = posVec.xyz.sub(ctx.uPointerPos.xyz);
          const d = length(rel);
          If(d.lessThan(ctx.uPointerPos.w), () => {
            const f = float(1).sub(d.div(ctx.uPointerPos.w));
            velVec.x.addAssign(ctx.uPointerVel.x.mul(f));
            velVec.y.addAssign(ctx.uPointerVel.y.mul(f));
            velVec.z.addAssign(ctx.uPointerVel.z.mul(f));
          });
        });

        velVec.x.addAssign(accel.x.mul(ctx.dt));
        velVec.y.addAssign(accel.y.mul(ctx.dt));
        velVec.z.addAssign(accel.z.mul(ctx.dt));
        posVec.x.addAssign(velVec.x.mul(ctx.dt));
        posVec.y.addAssign(velVec.y.mul(ctx.dt));
        posVec.z.addAssign(velVec.z.mul(ctx.dt));
        // Hard sphere projection keeps the particle inside the domain.
        If(ctx.uSphere.w.greaterThan(float(0)), () => {
          const rel = posVec.xyz.sub(ctx.uSphere.xyz);
          const d = length(rel);
          If(d.greaterThan(ctx.uSphere.w), () => {
            const n = rel.div(d);
            posVec.x.assign(ctx.uSphere.x.add(n.mul(ctx.uSphere.w).x));
            posVec.y.assign(ctx.uSphere.y.add(n.mul(ctx.uSphere.w).y));
            posVec.z.assign(ctx.uSphere.z.add(n.mul(ctx.uSphere.w).z));
          });
        });
        ctx.sVel.element(i).assign(velVec);
        ctx.sPos.element(i).assign(posVec);
      });
    });
  });

/** Adds `k * min(distance, 0)` to one axis (the upstream signed-wall trick). */
const signedWall = (
  axis: ShaderNodeObject<Node>,
  stiffness: ShaderNodeObject<Node>,
  distance: ShaderNodeObject<Node>
): void => {
  axis.addAssign(stiffness.mul(tslMin(distance, float(0))));
};

/** `x^3` of a scalar node. */
const cube = (n: ShaderNodeObject<Node>): ShaderNodeObject<Node> =>
  n.mul(n).mul(n);

// ─── Pipeline builder ────────────────────────────────────────────────────────

/** SPH compute pipeline handle (dispatch order = `computeNodes`). */
export type SPHPipeline = {
  /** Every kernel in strict dispatch order. */
  computeNodes: ReturnType<typeof compute>[];
  /** Semantic pass names, aligned with {@link SPHPipeline.computeNodes}. */
  passNames: string[];
  /** Real per-pass storage / uniform budgets. */
  passLayouts: PassLayout[];
  /** GPU storage owned by the solver. */
  buffers: SPHBuffers;
  /** Lattice size `xGrids * yGrids * zGrids`. */
  gridCount: number;
  /** Particle capacity of the pool. */
  numParticles: number;
  /** Host-written scalars (`boxWidthRatio` = `z` squeeze of the box). */
  uniforms: {
    boxWidthRatio: { value: number };
    /** `(cx, cy, cz, radius)`; radius `0` = box domain. */
    sphereDomain: { value: Vector4 };
    /** `(px, py, pz, influenceRadius)`; radius `0` disables the force. */
    pointerPos: { value: Vector4 };
    /** `(vx, vy, vz, 0)` pointer velocity. */
    pointerVel: { value: Vector4 };
  };
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
 * Builds the SPH compute pipeline.
 *
 * @param maxParticles - Particle capacity (dambreak lattice count).
 * @param params - Resolved scalar parameter block.
 * @param realHalfBox - Animated half-extents; drive the `integrate` walls
 *   exactly like the reference `changeBoxSize()` (defaults to the init box).
 * @param shared - Optional external position / velocity storage shared with
 *   the base modifier pipeline so the SPH result reaches the render material.
 * @returns Pipeline with kernels in strict dispatch order.
 */
export function createSPHPipeline(
  maxParticles: number,
  params: SPHParams,
  realHalfBox?: readonly [number, number, number],
  shared?: SPHSharedBuffers
): SPHPipeline {
  const count = Math.max(1, Math.floor(maxParticles));
  const [nx, ny, nz] = params.gridDims;
  const gridCount = nx * ny * nz;
  const scanBlocks = computeSPHScanBlocks(gridCount + 1);
  const scanSteps = computeSPHScanInnerSteps(scanBlocks);
  const box = realHalfBox ?? params.realHalfBox;
  const buffers = createSPHBuffers(count, gridCount, shared);
  // Live `z` squeeze of the box (`changeBoxSize` in the reference).
  const uBoxWidthRatio = uniform(
    params.halfBoxSize[2] > 0 ? box[2] / params.halfBoxSize[2] : 1
  );
  // Spherical boundary + pointer force (both `0`-radius = inert defaults).
  // `Vector4`-backed uniforms so the host mutates them in place per frame.
  const uSphere = uniform(new Vector4(0, 0, 0, 0));
  const uPointerPos = uniform(new Vector4(0, 0, 0, 0));
  const uPointerVel = uniform(new Vector4(0, 0, 0, 0));

  const sPos = storage(buffers.position, 'vec4', count);
  const sVel = storage(buffers.velocity, 'vec4', count);
  const sForce = storage(buffers.forceDensity, 'vec4', count);
  const sSortedPos = storage(buffers.sortedPosition, 'vec4', count);
  const sSortedVel = storage(buffers.sortedVelocity, 'vec4', count);
  const sSortedForce = storage(buffers.sortedForceDensity, 'vec4', count);
  const sCells = storage(buffers.cellCounts, 'uint', gridCount).toAtomic();
  const sPrefix = storage(buffers.prefixSums, 'float', gridCount + 1);
  const sOffsets = storage(buffers.particleCellOffsets, 'uint', count);
  const sPartials = storage(buffers.blockPartials, 'float', scanBlocks);
  const sInclusive = storage(buffers.blockInclusive, 'float', scanBlocks);
  const sBlockOffsets = storage(buffers.blockOffsets, 'float', scanBlocks);

  const ctx: SPHContext = {
    count,
    gridCount,
    xGrids: nx,
    yGrids: ny,
    zGrids: nz,
    scanBlocks,
    scanSteps,
    sPos,
    sVel,
    sForce,
    sSortedPos,
    sSortedVel,
    sSortedForce,
    sCells,
    sPrefix,
    sOffsets,
    sPartials,
    sInclusive,
    sBlockOffsets,
    cellSizeInv: float(1 / params.cellSize),
    offset: float(params.offset),
    uBoxWidthRatio,
    uSphere,
    uPointerPos,
    uPointerVel,
    // One half-max set feeds both the lattice coordinates and the walls
    // (`xHalfMax` / `yHalfMax` / `zHalfMax` of the reference params block);
    // the `z` axis carries the animated `boxWidthRatio` squeeze.
    halfX: float(box[0]),
    halfY: float(box[1]),
    halfZ: float(params.halfBoxSize[2]).mul(uBoxWidthRatio),
    radius: float(params.kernelRadius),
    radiusPow2: float(params.powers.pow2),
    r2Epsilon: float(SPH_R2_EPSILON),
    mass: float(params.mass),
    stiffness: float(params.stiffness),
    nearStiffness: float(params.nearStiffness),
    restDensity: float(params.restDensity),
    viscosity: float(params.viscosity),
    dt: float(params.dt),
    gravity: float(params.gravity),
    densityScale: float(params.densityScale),
    nearDensityScale: float(params.nearDensityScale),
    gradientScale: float(params.gradientScale),
    laplacianScale: float(params.laplacianScale),
    pool: [
      sPos,
      sVel,
      sForce,
      sSortedPos,
      sSortedVel,
      sSortedForce,
      sCells,
      sPrefix,
      sOffsets,
      sPartials,
      sInclusive,
      sBlockOffsets,
    ],
  };

  // Per-pass storage subsets (the real binding budget of each kernel).
  const subsets: Record<string, Array<ShaderNodeObject<Node>>> = {
    gridClear: [sCells],
    gridBuild: [sPos, sCells, sOffsets],
    scanPartials: [sCells, sPartials],
    scanBlocks: [sPartials, sInclusive, sBlockOffsets],
    scanApply: [sCells, sPrefix, sBlockOffsets],
    reorderPosition: [sPos, sVel, sSortedPos, sSortedVel, sPrefix, sOffsets],
    reorderForce: [sPos, sForce, sSortedForce, sPrefix, sOffsets],
    density: [sPos, sForce, sSortedPos, sPrefix],
    force: [sPos, sVel, sForce, sSortedPos, sSortedVel, sSortedForce, sPrefix],
    integrate: [sPos, sVel, sForce],
  };

  const clear = createGridClearKernel(ctx);
  const build = createGridBuildKernel(ctx);
  const scan1 = createScanPartialsKernel(ctx);
  const scan2 = createBlockScanKernel(ctx);
  const scan3 = createScanApplyKernel(ctx);
  const reorderPos = createReorderPositionKernel(ctx);
  const reorderForce = createReorderForceKernel(ctx);
  const density = createDensityKernel(ctx);
  const force = createForceKernel(ctx);
  const integrate = createIntegrateKernel(ctx);

  const passNames: string[] = [];
  const computeNodes: ReturnType<typeof compute>[] = [];
  const passLayouts: PassLayout[] = [];
  const push = (
    name: string,
    node: ReturnType<typeof compute>,
    subset: keyof typeof subsets | 'pool'
  ): void => {
    passNames.push(name);
    computeNodes.push(node);
    passLayouts.push(
      layout(name, subset === 'pool' ? ctx.pool : subsets[subset], [])
    );
  };

  for (let step = 0; step < SPH_SUBSTEPS; step++) {
    const suffix = SPH_SUBSTEPS > 1 ? `_${step + 1}` : '';
    push(`gridClear${suffix}`, compute(clear(), gridCount), 'gridClear');
    push(`gridBuild${suffix}`, compute(build(), count), 'gridBuild');
    push('scanPartials', compute(scan1(), scanBlocks), 'scanPartials');
    push('scanBlocks', compute(scan2(), SPH_SCAN_CHUNK), 'scanBlocks');
    push('scanApply', compute(scan3(), scanBlocks), 'scanApply');
    push(
      `reorderPosition${suffix}`,
      compute(reorderPos(), count),
      'reorderPosition'
    );
    push(
      `reorderForce${suffix}`,
      compute(reorderForce(), count),
      'reorderForce'
    );
    push(`density${suffix}`, compute(density(), count), 'density');
    // The density pass rewrites `nearDensity` / `density`, so the packed order
    // is refreshed once more before the force kernel consumes it.
    push(
      `reorderPositionB${suffix}`,
      compute(reorderPos(), count),
      'reorderPosition'
    );
    push(
      `reorderForceB${suffix}`,
      compute(reorderForce(), count),
      'reorderForce'
    );
    push(`force${suffix}`, compute(force(), count), 'force');
    push(`integrate${suffix}`, compute(integrate(), count), 'integrate');
  }

  return {
    computeNodes,
    passNames,
    passLayouts,
    buffers,
    gridCount,
    numParticles: count,
    uniforms: {
      boxWidthRatio: uBoxWidthRatio,
      sphereDomain: uSphere,
      pointerPos: uPointerPos,
      pointerVel: uPointerVel,
    },
  };
}
