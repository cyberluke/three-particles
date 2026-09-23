/**
 * Focused regression tests for the Electric Arc CPU subsystem.
 */
import * as THREE from 'three';
import {
  normalizeElectricArcConfig,
  mergeLiveConfig,
  touchesStructuralField,
} from '../js/effects/electric-arc/electric-arc-config.js';
import {
  radialProfile,
  triangleProfile,
} from '../js/effects/electric-arc/electric-arc-contact.js';
import { createElectricArcCpu } from '../js/effects/electric-arc/electric-arc-cpu.js';
import { ELECTRIC_ARC_PRESET_CINEMATIC } from '../js/effects/electric-arc/electric-arc-defaults.js';
import {
  PULSE_THIN_MIN,
  chaosFlickerHz,
  globalFlicker,
  organicOffset,
  pcg01Scalar,
  pcgRawU32Scalar,
  pinEnvelope,
  pulseEnvelope,
  pulseOffset,
  rotateZ2,
  widthFactor,
} from '../js/effects/electric-arc/electric-arc-math.js';
import {
  createArcSparks,
  pushSparkLiveConfig,
} from '../js/effects/electric-arc/electric-arc-sparks.js';
import { createElectricArc } from '../js/effects/electric-arc/electric-arc.js';
import { SimulationBackend } from '../js/effects/three-particles/three-particles-enums.js';

const baseConfig = {
  start: { x: -1.1, y: 0, z: 0 },
  end: { x: 1.1, y: 0, z: 0 },
  chaos: 0.19,
  seed: 271,
};

describe('electric-arc-math', () => {
  it('pcg scalar stays in uint32 and [0,1)', () => {
    for (const s of [0, 1, 271, 4294967295]) {
      const w = pcgRawU32Scalar(s);
      expect(w >>> 0).toBe(w);
      const f = pcg01Scalar(s);
      expect(f).toBeGreaterThanOrEqual(0);
      expect(f).toBeLessThan(1);
    }
  });

  it('endpoint envelope is exactly 0 at both ends', () => {
    expect(pinEnvelope(0, 0.72)).toBe(0);
    expect(pinEnvelope(1, 0.72)).toBe(0);
  });

  it('chaos macros derive monotonically and deterministically', () => {
    expect(chaosFlickerHz(0)).toBe(8);
    expect(chaosFlickerHz(1)).toBe(42);
    expect(globalFlicker(271, 4)).toBeCloseTo(globalFlicker(271, 4), 10);
    expect(
      Math.abs(globalFlicker(271, 5) - globalFlicker(271, 4))
    ).toBeGreaterThan(0);
  });
});

describe('normalizeElectricArcConfig', () => {
  it('derives chaos parameters (§11)', () => {
    const n = normalizeElectricArcConfig({ ...baseConfig, chaos: 0.5 });
    expect(n.chaos).toBe(0.5);
    // amplitude = distance * lerp(0.0025,0.045, pow(c,1.6)), distance = 2.2
    expect(n.amplitude).toBeCloseTo(
      2.2 * (0.0025 + (0.045 - 0.0025) * Math.pow(0.5, 1.6)),
      6
    );
    expect(n.coarseKnots).toBe(Math.round(4 + 16 * 0.5));
    expect(n.microFrequency).toBeCloseTo(15 + 60 * 0.5, 6);
    expect(n.branchProbability).toBeCloseTo(Math.pow(0.5, 2) * 0.32, 6);
  });

  it('honors quality tiers but explicit values win (§43)', () => {
    const n = normalizeElectricArcConfig({
      ...baseConfig,
      quality: 'low',
    });
    expect(n.segments).toBe(32);
    const n2 = normalizeElectricArcConfig({
      ...baseConfig,
      quality: 'low',
      segments: 96,
    });
    expect(n2.segments).toBe(96);
  });

  it('detects structural fields (§25)', () => {
    expect(touchesStructuralField({ segments: 64 })).toBe(true);
    expect(touchesStructuralField({ chaos: 0.4 })).toBe(false);
    expect(
      touchesStructuralField({ simulationBackend: SimulationBackend.CPU })
    ).toBe(true);
    expect(touchesStructuralField({ chaosAlgorithm: 'pulse' })).toBe(true);
  });

  it('pulse chaos model v2: three classes + degradation (§11)', () => {
    const n = normalizeElectricArcConfig({
      ...baseConfig,
      chaosAlgorithm: 'pulse',
    });
    expect(n.chaosAlgorithm).toBe('pulse');

    // spike envelope: 0.15 linear rise, then exponential slow tail
    expect(pulseEnvelope(0)).toBe(0);
    expect(pulseEnvelope(0.075)).toBeCloseTo(0.5, 5);
    expect(pulseEnvelope(0.15)).toBeCloseTo(1, 5);
    expect(pulseEnvelope(0.5)).toBeCloseTo(Math.exp(-3 * (0.5 - 0.15)), 5);
    expect(pulseEnvelope(1)).toBeCloseTo(Math.exp(-3 * 0.85), 5);

    // determinism + range
    const a = pulseOffset(271, 4, 0.37, n.coarseKnots, 0);
    const b = pulseOffset(271, 4, 0.37, n.coarseKnots, 0);
    expect(a).toBe(b);
    expect(Math.abs(a)).toBeLessThanOrEqual(1);

    // one epoch over 8 slots: gaps AND plateau runs must both appear
    const arr: number[] = [];
    for (let k = 1; k < 160; k++) arr.push(pulseOffset(97, 2, k / 160, 8, 0));
    const zeros = arr.filter((v) => v === 0).length;
    expect(zeros).toBeGreaterThan(0);
    let plateauRun = 0;
    for (let k = 0; k < arr.length - 2; k++) {
      if (arr[k] === arr[k + 1] && arr[k] === arr[k + 2] && arr[k] !== 0) {
        plateauRun = 3;
        break;
      }
    }
    expect(plateauRun).toBe(3);
  });

  it('normalizes glow profile and merges it live', () => {
    const n = normalizeElectricArcConfig({ ...baseConfig });
    expect(n.glow.profile).toBe('gaussian');
    mergeLiveConfig(n, { glow: { profile: 'triangle' } });
    expect(n.glow.profile).toBe('triangle');
    mergeLiveConfig(n, { glow: { profile: 'gaussian', width: 4 } });
    expect(n.glow.profile).toBe('gaussian');
    expect(n.glow.width).toBe(4);
  });

  it('triangle tent: peak 1 at center, 0 at its 1/e width', () => {
    expect(triangleProfile(0, 700)).toBe(1);
    const w = 1 / Math.sqrt(0.55 * 700); // 1/e-ish half width of gaussian k=700
    expect(triangleProfile(w, 700)).toBeCloseTo(0, 1);
    expect(triangleProfile(1.2, 700)).toBe(0);
    expect(radialProfile(0, 700)).toBe(1);
  });

  it('organic chaos model: deterministic with random pauses (§11)', () => {
    const n = normalizeElectricArcConfig({
      ...baseConfig,
      chaosAlgorithm: 'organic',
    });
    expect(n.chaosAlgorithm).toBe('organic');

    const a = organicOffset(271, 4, 0.32, n.coarseKnots, 0);
    const b = organicOffset(271, 4, 0.32, n.coarseKnots, 0);
    expect(a).toBe(b);
    expect(Math.abs(a)).toBeLessThanOrEqual(1);

    // pauses: within one epoch some consecutive samples repeat exactly
    const arr: number[] = [];
    for (let k = 1; k < 96; k++) arr.push(organicOffset(57, 6, k / 96, 9, 0));
    let pause = 0;
    for (let k = 0; k < arr.length - 1; k++) {
      if (arr[k] === arr[k + 1]) {
        pause = 2;
        break;
      }
    }
    expect(pause).toBe(2);

    // live merge recognizes the third model
    const m = normalizeElectricArcConfig({ ...baseConfig });
    mergeLiveConfig(m, { chaosAlgorithm: 'organic' });
    expect(m.chaosAlgorithm).toBe('organic');
    expect(touchesStructuralField({ chaosAlgorithm: 'organic' })).toBe(true);
  });

  it('variable width thins rotated samples only', () => {
    expect(widthFactor(0, 0)).toBe(1);
    expect(widthFactor(1, 1)).toBeCloseTo(PULSE_THIN_MIN, 5);
    expect(widthFactor(1, 1)).toBeLessThan(widthFactor(0, 0));
  });

  it('rotationZ rotates the arc endpoints around Z', () => {
    const n = normalizeElectricArcConfig({ ...baseConfig, rotationZ: 90 });
    const a = n.start.clone();
    const b = n.end.clone();
    rotateZ2(a, n.rotationZ);
    rotateZ2(b, n.rotationZ);
    expect(a.x).toBeCloseTo(0, 5);
    expect(a.y).toBeCloseTo(-1.1, 5);
    expect(b.x).toBeCloseTo(0, 5);
    expect(b.y).toBeCloseTo(1.1, 5);
    // live merge is non-structural
    const m = normalizeElectricArcConfig({ ...baseConfig });
    mergeLiveConfig(m, { rotationZ: 180 });
    expect(m.rotationZ).toBe(180);
    expect(touchesStructuralField({ rotationZ: 90 })).toBe(false);
  });

  it('sparks size/lifetime/speed min-max merge live (§22)', () => {
    const n = normalizeElectricArcConfig({
      ...baseConfig,
      sparks: { enabled: true, rate: 6 },
    });
    mergeLiveConfig(n, {
      sparks: {
        size: [0.2, 0.8],
        lifetime: [0.1, 0.5],
        speed: [1, 4],
      },
    });
    expect(n.sparks.size).toEqual([0.2, 0.8]);
    expect(n.sparks.lifetime).toEqual([0.1, 0.5]);
    expect(n.sparks.speed).toEqual([1, 4]);
    // incomplete pair keeps previous values
    mergeLiveConfig(n, { sparks: { size: [0.3] as [number, number] } });
    expect(n.sparks.size).toEqual([0.2, 0.8]);
  });
});

function invokeOnBeforeRender(meshGroup: THREE.Object3D[]): void {
  const camera = new THREE.PerspectiveCamera(45, 320 / 220, 1, 100);
  camera.position.set(0, 0, 15);
  camera.updateMatrixWorld(true);
  const fakeRenderer = {} as unknown as THREE.WebGLRenderer;
  const fakeScene = new THREE.Scene();
  for (const g of meshGroup) {
    g.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh && typeof m.onBeforeRender === 'function') {
        m.onBeforeRender(
          fakeRenderer,
          fakeScene,
          camera,
          m.geometry,
          m.material as THREE.Material,
          m as unknown as THREE.Group
        );
      }
    });
  }
}

describe('createElectricArc CPU fallback', () => {
  it('constructs without WebGPU and reports CPU backend (§3)', () => {
    const arc = createElectricArc({ ...baseConfig });
    expect(arc.instance).toBeInstanceOf(THREE.Group);
    expect(arc.backend).toBe(SimulationBackend.CPU);
    expect(arc.computeNode).toBeNull();
    arc.dispose();
    arc.dispose();
  });

  it('endpoints stay exactly on the wired points and follow moves (§6, §24)', () => {
    const arc = createElectricArc({ ...baseConfig, segments: 32 });
    const start = new THREE.Vector3(-1.1, 0, 0);
    const end = new THREE.Vector3(1.1, 0, 0);
    arc.setEndpoints(start, end);
    arc.update({ now: 1, delta: 0.016, elapsed: 0.016 });
    invokeOnBeforeRender(arc.instance.children);

    const cpuGroup = arc.instance.children[0];
    let found: THREE.BufferAttribute | undefined;
    cpuGroup.traverse((o) => {
      const mesh2 = o as THREE.Mesh;
      if (mesh2.isMesh) {
        const attr = mesh2.geometry.getAttribute(
          'position'
        ) as THREE.BufferAttribute;
        if (attr && (attr.array as Float32Array).length === 32 * 2 * 3)
          found = attr;
      }
    });
    expect(found).toBeDefined();
    const arr = found!.array as Float32Array;
    // centerline exact: midpoint of each left/right billboard pair (§6)
    const midX0 = (arr[0] + arr[3]) / 2;
    const midY0 = (arr[1] + arr[4]) / 2;
    expect(midX0).toBeCloseTo(-1.1, 5);
    expect(midY0).toBeCloseTo(0, 5);
    const li = (32 * 2 - 2) * 3;
    expect((arr[li] + arr[li + 3]) / 2).toBeCloseTo(1.1, 5);

    // move endpoints
    arc.setEndpoints(new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 0.5, 0));
    arc.update({ now: 2, delta: 0.016, elapsed: 0.032 });
    invokeOnBeforeRender(arc.instance.children);
    expect((arr[0] + arr[3]) / 2).toBeCloseTo(0, 5);
    expect((arr[1] + arr[4]) / 2).toBeCloseTo(0, 5);
    expect((arr[li] + arr[li + 3]) / 2).toBeCloseTo(1, 5);
    expect((arr[li + 1] + arr[li + 4]) / 2).toBeCloseTo(0.5, 5);
    arc.dispose();
  });

  it('bindEndpoints follows live scene objects (§24)', () => {
    const a = new THREE.Object3D();
    a.position.set(-2, 0, 0);
    const b = new THREE.Object3D();
    b.position.set(2, 0, 0);
    const arc = createElectricArc({ ...baseConfig, segments: 16 });
    arc.bindEndpoints({
      start: { object: a, offset: new THREE.Vector3(0.5, 0, 0) },
      end: { object: b, offset: new THREE.Vector3(-0.5, 0, 0) },
    });
    arc.update({ now: 1, delta: 0.016, elapsed: 0.016 });
    invokeOnBeforeRender(arc.instance.children);
    const cpuGroup = arc.instance.children[0];
    let arr: Float32Array | undefined;
    cpuGroup.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) {
        const aPos = m.geometry.getAttribute('position');
        if (aPos && (aPos.array as Float32Array).length === 16 * 2 * 3)
          arr = aPos.array as Float32Array;
      }
    });
    expect(arr).toBeDefined();
    // pair midpoint = exact bound endpoint (§6)
    expect((arr[0] + arr[3]) / 2).toBeCloseTo(-1.5, 4);
    const li = (16 * 2 - 2) * 3;
    expect((arr[li] + arr[li + 3]) / 2).toBeCloseTo(1.5, 4);
    a.position.x = -3;
    arc.update({ now: 2, delta: 0.016, elapsed: 0.032 });
    invokeOnBeforeRender(arc.instance.children);
    expect((arr[0] + arr[3]) / 2).toBeCloseTo(-2.5, 4);
    arc.clearEndpointBinding();
    arc.setEndpoints({ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 });
    arc.update({ now: 3, delta: 0.016, elapsed: 0.048 });
    invokeOnBeforeRender(arc.instance.children);
    expect((arr[0] + arr[3]) / 2).toBeCloseTo(0, 4);
    arc.dispose();
  });

  it('deterministic per seed and epoch (§9, §10)', () => {
    const n1 = normalizeElectricArcConfig({ ...baseConfig });
    const a = createElectricArcCpu(n1);
    const s = new THREE.Vector3(-1.1, 0, 0);
    const e = new THREE.Vector3(1.1, 0, 0);
    a.update({ now: 1, delta: 0.016, elapsed: 0.1 }, s, e);
    const first = Array.from(
      (a.root.children[0] as THREE.Mesh).geometry.getAttribute('color')
        .array as Float32Array
    );
    // same elapsed -> same epoch -> identical brightness bake
    a.update({ now: 1.1, delta: 0.016, elapsed: 0.1 }, s, e);
    const second = Array.from(
      (a.root.children[0] as THREE.Mesh).geometry.getAttribute('color')
        .array as Float32Array
    );
    expect(first.length).toBe(second.length);
    for (let i = 0; i < first.length; i++) {
      expect(second[i]).toBe(first[i]);
    }
    a.dispose();
  });

  it('chassis preset and live config update without rebuild (§25, §42)', () => {
    const merged = { ...baseConfig, ...ELECTRIC_ARC_PRESET_CINEMATIC };
    const n = normalizeElectricArcConfig(merged);
    expect(n.segments).toBe(128);
    expect(n.intensity).toBe(12);
    const arc = createElectricArc(n);
    const before = arc.instance.children.length;
    arc.updateConfig({ chaos: 0.5, flickerHz: 30, intensity: 9 });
    arc.update({ now: 1, delta: 0.016, elapsed: 0.05 });
    expect(arc.instance.children.length).toBe(before);
    // structural: segments triggers an internal rebuild but stable outer group
    arc.updateConfig({ segments: 64 });
    expect(arc.instance).toBeDefined();
    arc.update({ now: 2, delta: 0.016, elapsed: 0.066 });
    expect(arc.instance.children.length).toBeGreaterThan(0);
    arc.dispose();
    expect(arc.instance.children.length).toBe(0);
  });

  it('spark systems receive merged size/speed/lifetime (§22)', () => {
    const n = normalizeElectricArcConfig({
      ...baseConfig,
      sparks: { enabled: true, rate: 6 },
    });
    // stub child particle systems (the real factory is async and ESM-only)
    const seen: Record<string, unknown>[] = [];
    const stubs = [1, 2, 3].map(
      () =>
        ({
          updateConfig: (p: Record<string, unknown>) => seen.push(p),
        }) as never
    );

    mergeLiveConfig(n, {
      sparks: { size: [0.25, 0.9], speed: [1, 5], lifetime: [0.1, 0.5] },
    });
    expect(() => pushSparkLiveConfig(stubs, n)).not.toThrow();
    expect(seen.length).toBe(3);
    expect(seen[0].startSize).toEqual({ min: 0.25, max: 0.9 });
    expect(seen[0].startSpeed).toEqual({ min: 1, max: 5 });
    expect(seen[0].startLifetime).toEqual({ min: 0.1, max: 0.5 });
    expect(seen[0].emission).toEqual({ rateOverTime: 6 });

    // merging an incomplete pair keeps previous values; empty list is safe
    mergeLiveConfig(n, { sparks: { size: [0.4] as [number, number] } });
    expect(n.sparks.size).toEqual([0.25, 0.9]);
    expect(() => pushSparkLiveConfig([], n)).not.toThrow();
  });
});

describe('persistent endpoint transforms (4.1.2)', () => {
  it('zero transforms reproduce legacy output exactly', () => {
    const legacy = createElectricArc({ ...baseConfig, segments: 32 });
    const modern = createElectricArc({
      ...baseConfig,
      segments: 32,
      startOffset: { x: 0, y: 0, z: 0 },
      endOffset: { x: 0, y: 0, z: 0 },
      startRotation: { pitch: 0, yaw: 0, roll: 0 },
      endRotation: { pitch: 0, yaw: 0, roll: 0 },
    });
    legacy.update({ now: 1, delta: 0.016, elapsed: 0.2 });
    modern.update({ now: 1, delta: 0.016, elapsed: 0.2 });
    invokeOnBeforeRender(legacy.instance.children);
    invokeOnBeforeRender(modern.instance.children);
    const read = (g: THREE.Object3D): Float32Array => {
      let out: Float32Array | undefined;
      g.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) {
          const a = m.geometry.getAttribute('position');
          if (a && (a.array as Float32Array).length === 32 * 2 * 3)
            out = a.array as Float32Array;
        }
      });
      return out!;
    };
    const a = read(legacy.instance.children[0]);
    const b = read(modern.instance.children[0]);
    expect(b.length).toBe(a.length);
    for (let i = 0; i < a.length; i++) expect(b[i]).toBeCloseTo(a[i], 6);
    legacy.dispose();
    modern.dispose();
  });

  it('normalize defaults: offsets zero, rotations resolved', () => {
    const n = normalizeElectricArcConfig({ ...baseConfig });
    expect(n.startOffset.toArray()).toEqual([0, 0, 0]);
    expect(n.endOffset.toArray()).toEqual([0, 0, 0]);
    expect(n.startRotation).toEqual({ pitch: 0, yaw: 0, roll: 0 });
    expect(n.endRotation).toEqual({ pitch: 0, yaw: 0, roll: 0 });
  });

  it('setEndpoints updates base anchors without clearing persistent transforms', () => {
    const arc = createElectricArc({
      ...baseConfig,
      segments: 16,
      startOffset: { x: 0.2, y: 0, z: 0 },
    });
    arc.setEndpoints({ x: 0, y: 0, z: 0 }, { x: 2, y: 0, z: 0 });
    arc.update({ now: 1, delta: 0.016, elapsed: 0.016 });
    arc.setEndpoints({ x: 1, y: 0, z: 0 }, { x: 3, y: 0, z: 0 });
    arc.update({ now: 2, delta: 0.016, elapsed: 0.032 });
    const rt = arc.getRuntimeEndpoints();
    // +X chord: N=(1,0,0), U=N×(0,1,0)=(0,0,1) -> the 0.2 offset lands on Z.
    // base moved to (1,0,0)/(3,0,0); effective must still carry the offset.
    expect(rt.baseStart.x).toBeCloseTo(1, 5);
    expect(rt.effectiveStart.x).toBeCloseTo(1, 5);
    expect(rt.effectiveStart.z).toBeCloseTo(0.2, 5);
    expect(rt.effectiveEnd.x).toBeCloseTo(3, 5);
    expect(rt.effectiveEnd.z).toBeCloseTo(0, 5);
    // rotationZ path untouched: 90 deg maps +X chord onto +Y
    arc.updateConfig({ rotationZ: 90 });
    const r90 = arc.getRuntimeEndpoints();
    expect(r90.effectiveStart.x).toBeCloseTo(0, 4);
    expect(r90.effectiveStart.y).toBeCloseTo(1, 4);
    arc.dispose();
  });

  it('offset survives repeated frame updates and serializes via config', () => {
    const n = normalizeElectricArcConfig({
      ...baseConfig,
      endOffset: { x: 0, y: 0, z: 0.5 },
    });
    // JSON round trip through the persistent surface
    const json = JSON.parse(
      JSON.stringify({
        start: n.start.toArray(),
        end: n.end.toArray(),
        startOffset: n.startOffset.toArray(),
        endOffset: n.endOffset.toArray(),
        startRotation: n.startRotation,
        endRotation: n.endRotation,
        rotationZ: n.rotationZ,
      })
    );
    expect(json.endOffset).toEqual([0, 0, 0.5]);
    const m = normalizeElectricArcConfig({
      start: { x: json.start[0], y: json.start[1], z: json.start[2] },
      end: { x: json.end[0], y: json.end[1], z: json.end[2] },
      endOffset: { x: 0, y: 0, z: json.endOffset[2] },
    });
    expect(m.endOffset.z).toBeCloseTo(0.5, 10);
    const arc = createElectricArc(m);
    for (let f = 0; f < 5; f++)
      arc.update({ now: f, delta: 0.016, elapsed: f * 0.016 });
    const rt = arc.getRuntimeEndpoints();
    // z along the tangent of the +X chord: effective end pushed +0.5 on X
    expect(rt.effectiveEnd.x).toBeCloseTo(1.6, 5);
    arc.dispose();
  });

  it('endpoint rotation changes the offset direction observably', () => {
    const arc = createElectricArc({
      ...baseConfig,
      segments: 16,
      endOffset: { x: 1, y: 0, z: 0 },
      endRotation: { pitch: 0, yaw: 0, roll: 90 },
    });
    arc.update({ now: 1, delta: 0.016, elapsed: 0.016 });
    const a = arc.getRuntimeEndpoints();
    arc.updateConfig({ endRotation: { pitch: 0, yaw: 0, roll: 0 } });
    arc.update({ now: 2, delta: 0.016, elapsed: 0.032 });
    const b = arc.getRuntimeEndpoints();
    // roll around the tangent rotates U->V: effective position must differ
    const d = Math.hypot(
      a.effectiveEnd.x - b.effectiveEnd.x,
      a.effectiveEnd.y - b.effectiveEnd.y,
      a.effectiveEnd.z - b.effectiveEnd.z
    );
    expect(d).toBeGreaterThan(0.05);
    arc.dispose();
  });

  it('partial live patches keep unspecified components; NaN falls back', () => {
    const n = normalizeElectricArcConfig({
      ...baseConfig,
      startOffset: { x: 1, y: 2, z: 3 },
      endRotation: { pitch: 10, yaw: 20, roll: 30 },
    });
    mergeLiveConfig(n, { startOffset: { y: 5 } });
    expect(n.startOffset.toArray()).toEqual([1, 5, 3]);
    mergeLiveConfig(n, {
      endRotation: { pitch: Number.NaN, yaw: 40 },
    });
    expect(n.endRotation.pitch).toBe(10); // NaN keeps previous
    expect(n.endRotation.yaw).toBe(40);
    expect(n.endRotation.roll).toBe(30);
    expect(touchesStructuralField({ startOffset: { x: 1 } })).toBe(false);
    expect(touchesStructuralField({ endRotation: { yaw: 1 } })).toBe(false);
  });

  it('degenerate zero-length chord keeps stable frames (no NaN)', () => {
    const arc = createElectricArc({
      ...baseConfig,
      segments: 8,
      startOffset: { x: 0.1, y: 0, z: 0 },
    });
    arc.setEndpoints({ x: 1, y: 1, z: 1 }, { x: 1, y: 1, z: 1 });
    arc.update({ now: 1, delta: 0.016, elapsed: 0.016 });
    const rt = arc.getRuntimeEndpoints();
    for (const v of [rt.effectiveStart, rt.effectiveEnd]) {
      expect(Number.isFinite(v.x)).toBe(true);
      expect(Number.isFinite(v.y)).toBe(true);
      expect(Number.isFinite(v.z)).toBe(true);
    }
    expect(Number.isFinite(rt.startFrame.tangent.x)).toBe(true);
    arc.dispose();
  });
});
