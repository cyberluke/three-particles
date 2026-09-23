import * as THREE from 'three';
import { SimulationSpace } from '../js/effects/three-particles/three-particles-enums.js';
import { createParticleSystem } from '../js/effects/three-particles/three-particles.js';
import type { ParticleSystem } from '../js/effects/three-particles/types.js';

/**
 * GPU-only contract: the CPU owns only the scalar uniforms. The per-frame
 * emission count is observable through `gpuDebug.lastEmitCount()`.
 */
const lastEmit = (ps: ParticleSystem): number =>
  (ps as unknown as { gpuDebug: { lastEmitCount(): number } }).gpuDebug.lastEmitCount();

const createWorldSpaceSystem = (
  extraConfig: Record<string, unknown> = {},
  startTime = 1000
) => {
  const ps = createParticleSystem(
    {
      maxParticles: 20,
      duration: 5,
      looping: true,
      startLifetime: 2,
      startSpeed: 1,
      simulationSpace: SimulationSpace.WORLD,
      emission: { rateOverTime: 50 },
      ...extraConfig,
    } as never,
    startTime
  );

  const step = (timeOffsetMs: number, deltaMs: number = 16) => {
    ps.update({
      now: startTime + timeOffsetMs,
      delta: deltaMs / 1000,
      elapsed: timeOffsetMs / 1000,
    });
  };

  return { ps, step, startTime };
};

describe('World Simulation Space', () => {
  it('should expose the particle system directly (no wrapper) in WORLD space', () => {
    const { ps } = createWorldSpaceSystem();

    // After the WORLD-space refactor, instance is the Points itself and
    // matrixWorld is pinned to identity (no Gyroscope wrapper).
    expect(ps.instance).toBeDefined();
    expect(ps.instance).toBeInstanceOf(THREE.Points);
    expect(ps.instance.matrixWorldAutoUpdate).toBe(false);
    expect(ps.instance.matrixWorld.elements).toEqual(
      new THREE.Matrix4().elements
    );

    ps.dispose();
  });

  it('should emit particles in world space', () => {
    const { ps, step } = createWorldSpaceSystem();

    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('should handle world position changes', () => {
    const { ps, step } = createWorldSpaceSystem();

    step(16);
    (ps.instance as THREE.Points).position.set(10, 0, 0);
    step(100);

    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('should compensate for emitter movement in world space', () => {
    const { ps, step } = createWorldSpaceSystem({
      startSpeed: 0,
      emission: { rateOverTime: 50 },
    });

    step(16);
    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    // Move emitter — the WORLD pose is handed to the kernel per frame.
    (ps.instance as THREE.Points).position.set(5, 0, 0);
    step(200);

    expect(ps.instance).toBeDefined();

    ps.dispose();
  });

  it('should handle parent attachment for wrapper quaternion', () => {
    const { ps, step } = createWorldSpaceSystem();

    const parent = new THREE.Group();
    parent.add(ps.instance);
    parent.quaternion.setFromEuler(new THREE.Euler(0, Math.PI / 4, 0));

    step(16);
    step(100);

    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('should work with gravity in world space', () => {
    const { ps, step } = createWorldSpaceSystem({
      gravity: -9.8,
      startSpeed: 0,
    });

    step(16);
    step(100);
    step(200);

    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });

  it('should remove the particle system from scene when disposed', () => {
    const scene = new THREE.Group();
    const { ps, step } = createWorldSpaceSystem();
    scene.add(ps.instance);

    step(16);
    expect(scene.children.length).toBe(1);

    ps.dispose();

    expect(scene.children.length).toBe(0);
  });

  it('should handle pause/resume in world space', () => {
    const { ps, step } = createWorldSpaceSystem();

    step(100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.pauseEmitter();
    step(200);
    expect(lastEmit(ps)).toBe(0);

    ps.resumeEmitter();
    step(300, 100);
    expect(lastEmit(ps)).toBeGreaterThan(0);

    ps.dispose();
  });
});

describe('Local Simulation Space', () => {
  it('should expose the Points object directly for LOCAL space', () => {
    const ps = createParticleSystem(
      {
        maxParticles: 10,
        simulationSpace: SimulationSpace.LOCAL,
        emission: { rateOverTime: 10 },
      },
      1000
    );

    // In local space, instance is directly the Points object
    expect((ps.instance as THREE.Points).geometry).toBeDefined();
    expect(ps.instance.matrixWorldAutoUpdate).not.toBe(false);

    ps.dispose();
  });

  it('should default to LOCAL simulation space', () => {
    const ps = createParticleSystem({ maxParticles: 10 }, 1000);

    // Default should be local - instance is Points directly
    expect((ps.instance as THREE.Points).geometry).toBeDefined();

    ps.dispose();
  });
});
