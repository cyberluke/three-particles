/**
 * Electric Arc spark emission through the existing particle engine (§22).
 *
 * When enabled, three lightweight `createParticleSystem()` instances emit
 * from endpoint A, endpoint B and an occasional centerline position. Both
 * electric-arc backends reuse whichever particle backend is currently
 * selected — the arc and the sparks are separate simulation concerns.
 *
 * @module
 */
import * as THREE from 'three';
import { mixSeedScalar } from './electric-arc-math.js';
import type { NormalizedElectricArcConfig } from './electric-arc-types.js';
import type { ParticleSystem } from '../three-particles/types.js';

// The particle module carries the shared `createParticleSystem` entry once
// the engine is loaded; the spark factory resolves it lazily (one-time) so
// the electric-arc module graph parses independently in every loader.
let _createParticleSystem:
  ((config: Record<string, unknown>) => ParticleSystem) | null = null;
async function resolveParticleSystemFactory(): Promise<
  typeof _createParticleSystem
> {
  if (_createParticleSystem) return _createParticleSystem;
  const mod = (await import('../three-particles/three-particles.js')) as {
    createParticleSystem: (config: Record<string, unknown>) => ParticleSystem;
  };
  _createParticleSystem = mod.createParticleSystem;
  return _createParticleSystem;
}

export type ArcSparks = {
  group: THREE.Group;
  systems: ParticleSystem[];
  update: (
    cycle: { now: number; delta: number; elapsed: number },
    start: THREE.Vector3,
    end: THREE.Vector3
  ) => void;
  dispose: () => void;
};

const sparkConfig = (
  cfg: NormalizedElectricArcConfig
): Record<string, unknown> => {
  const s = cfg.sparks;
  return {
    duration: 0,
    looping: true,
    startLifetime: {
      min: Math.max(0.02, s.lifetime[0]),
      max: Math.max(0.03, s.lifetime[1]),
    },
    startSpeed: { min: s.speed[0], max: s.speed[1] },
    startSize: { min: s.size[0], max: s.size[1] },
    startOpacity: 1,
    startColor: {
      min: { r: 1.0, g: 1.0, b: 0.85 },
      max: { r: 0.72, g: 1.0, b: 0.39 },
    },
    maxParticles: Math.min(64, Math.max(8, Math.round(s.rate * 0.6))),
    gravity: 1.5,
    emission: { rateOverTime: s.rate },
    shape: { shape: 'SPHERE', sphere: { radius: 0.02, radiusThickness: 1 } },
    renderer: {
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthTest: true,
      depthWrite: false,
    },
    opacityOverLifetime: {
      isActive: true,
      lifetimeCurve: {
        type: 'BEZIER',
        bezierPoints: [
          { x: 0, y: 1 },
          { x: 1, y: 0 },
        ],
      },
    },
  };
};

/**
 * Push the (already merged) numeric spark fields into existing live spark
 * systems. Particle `startSize` / `startSpeed` / `startLifetime` scalars
 * are re-read from the normalized particle config on every emission, so a
 * deepMerge through `updateConfig` takes effect immediately — including
 * the min/max size pair.
 */
export const pushSparkLiveConfig = (
  systems: ParticleSystem[],
  cfg: NormalizedElectricArcConfig
): void => {
  const s = cfg.sparks;
  const patch = {
    startLifetime: {
      min: Math.max(0.02, s.lifetime[0]),
      max: Math.max(0.03, s.lifetime[1]),
    },
    startSpeed: { min: s.speed[0], max: s.speed[1] },
    startSize: { min: s.size[0], max: s.size[1] },
    maxParticles: Math.min(64, Math.max(8, Math.round(s.rate * 0.6))),
    emission: { rateOverTime: s.rate },
  };
  for (const sys of systems) {
    try {
      sys.updateConfig(patch);
    } catch {
      /* detached / already disposed child */
    }
  }
};

export function createArcSparks(
  cfg: NormalizedElectricArcConfig
): ArcSparks | null {
  if (!cfg.sparks.enabled || cfg.sparks.rate <= 0) return null;

  const base = sparkConfig(cfg);
  const group = new THREE.Group();
  const systems: ParticleSystem[] = [];

  // Lazy (non-blocking) creation: resolveParticleSystemFactory() returns a
  // promise the first time; `update()` keeps the system positions in sync
  // once the child systems exist. 3 fixed endpoints: start, end, midpoint.
  void resolveParticleSystemFactory().then((factory) => {
    if (!factory || systems.length > 0) return;
    try {
      for (let i = 0; i < 3; i++) systems.push(factory({ ...base }));
      for (const s of systems) group.add(s.instance as THREE.Object3D);
    } catch {
      systems.length = 0;
    }
  });

  const posA = new THREE.Vector3();
  const posB = new THREE.Vector3();
  const posC = new THREE.Vector3();

  let disposed = false;
  return {
    group,
    systems,
    update: (cycle, start, end) => {
      if (disposed) return;
      // deterministic jitter so the third emitter does not sit exactly on 0.5
      const mix = mixSeedScalar(cfg.seed, 1, 17) * (1 / 4294967296);
      const t = 0.3 + mix * 0.4;
      posC.set(
        start.x + (end.x - start.x) * t,
        start.y + (end.y - start.y) * t,
        start.z + (end.z - start.z) * t
      );
      posA.copy(start);
      posB.copy(end);
      if (systems.length < 3) return; // async creation may lag one frame
      systems[0].instance.position.copy(posA);
      systems[1].instance.position.copy(posB);
      systems[2].instance.position.copy(posC);
      systems[0].update(cycle);
      systems[1].update(cycle);
      systems[2].update(cycle);
    },
    dispose: () => {
      if (disposed) return;
      disposed = true;
      for (const s of systems) {
        try {
          s.dispose();
        } catch {
          /* already disposed */
        }
      }
    },
  };
}
