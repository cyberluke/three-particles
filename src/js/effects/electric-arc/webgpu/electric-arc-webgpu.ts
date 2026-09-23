/**
 * WebGPU implementation of the Electric Arc backend (§3, §12–§16).
 *
 * One compute dispatch writes the centerline (positions + brightness) into
 * a single vec4 storage buffer; camera-facing expansion and the HDR
 * filament profile happen entirely in the TSL materials: the main arc is
 * one draw call, branches are one additional batched draw, contacts are
 * procedural billboard quads. No geometry rebuild, no buffer resize, no
 * GPU readbacks (§41).
 *
 * Registered through `electric-arc-gpu-registry.ts` by `enableWebGPU()`
 * from `@cyberluke/three-particles/webgpu`.
 *
 * @module
 */
import * as THREE from 'three';
import { Vector3 } from 'three';
import { SimulationBackend } from '../../three-particles/three-particles-enums.js';
import { globalFlicker } from '../electric-arc-math.js';
import {
  ARC_BRANCH_SAMPLES,
  createElectricArcCompute,
} from './compute-electric-arc.js';
import { createElectricArcRibbonMaterial } from './tsl-electric-arc-material.js';
import { createElectricContactMaterial } from './tsl-electric-contact-material.js';
import type {
  ElectricArcBackendInstance,
  ElectricArcConfig,
  NormalizedElectricArcConfig,
} from '../electric-arc-types.js';

// module scratch
const _dir = new Vector3();
const _helper = new Vector3();
const _ub = new Vector3();
const _vb = new Vector3();

const nBasis = (out: Vector3, a: Vector3, b: Vector3): void => {
  out.crossVectors(a, b);
  const l = out.length() || 1;
  out.multiplyScalar(1 / l);
};

/** static indexed camera-facing strip (§13). */
function ribbonGeometry(
  start: number,
  count: number,
  u0: number,
  u1: number
): THREE.BufferGeometry {
  const vertCount = count * 2;
  const geometry = new THREE.BufferGeometry();
  const pos = new Float32Array(vertCount * 4);
  const uv = new Float32Array(vertCount * 2);
  const idx = new Uint16Array((count - 1) * 6);
  const inv = 1 / (count - 1);
  for (let j = 0; j < count; j++) {
    const i = start + j;
    const prev = j === 0 ? i : i - 1;
    const next = j === count - 1 ? i : i + 1;
    const li = j * 2;
    const ri = li + 1;
    pos[li * 4] = i;
    pos[li * 4 + 1] = -1;
    pos[li * 4 + 2] = prev;
    pos[li * 4 + 3] = next;
    pos[ri * 4] = i;
    pos[ri * 4 + 1] = 1;
    pos[ri * 4 + 2] = prev;
    pos[ri * 4 + 3] = next;
    const u = u0 + (u1 - u0) * j * inv;
    uv[li * 2] = u;
    uv[li * 2 + 1] = 0;
    uv[ri * 2] = u;
    uv[ri * 2 + 1] = 1;
  }
  for (let j = 0; j < count - 1; j++) {
    const l0 = j * 2;
    const r0 = l0 + 1;
    const l1 = l0 + 2;
    const r1 = l1 + 1;
    const o = j * 6;
    idx[o] = l0;
    idx[o + 1] = r0;
    idx[o + 2] = l1;
    idx[o + 3] = r0;
    idx[o + 4] = r1;
    idx[o + 5] = l1;
  }
  geometry.setAttribute('position', new THREE.BufferAttribute(pos, 4));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geometry.setIndex(new THREE.BufferAttribute(idx, 1));
  geometry.boundingSphere = new THREE.Sphere(new Vector3(), 8);
  return geometry;
}

function contactGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), 2)
  );
  geometry.setAttribute(
    'uv',
    new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), 2)
  );
  geometry.setIndex(
    new THREE.BufferAttribute(new Uint16Array([0, 1, 2, 1, 3, 2]), 1)
  );
  geometry.boundingSphere = new THREE.Sphere(new Vector3(), 1);
  return geometry;
}

const setNum = (u: unknown, v: number): void => {
  (u as { value: number }).value = v;
};

export function createElectricArcGPU(
  cfg: NormalizedElectricArcConfig
): ElectricArcBackendInstance {
  const root = new THREE.Group();
  root.name = 'electric-arc-gpu';

  const pipeline = createElectricArcCompute(cfg);
  const { arcBuffer, widthBuffer, totalSamples, uniforms, mainCount } =
    pipeline;

  const coreColorVec = new Vector3();
  const arcColorVec = new Vector3();
  const setColors = (): void => {
    const c = new THREE.Color(cfg.coreColor);
    coreColorVec.set(c.r, c.g, c.b);
    const a = new THREE.Color(cfg.color);
    arcColorVec.set(a.r, a.g, a.b);
  };
  setColors();

  // ── main ribbon: rebuilt glow (§15/§18) = 3 stacked analytic ribbons.
  // core half-span comes from `thickness` only; `glow.width` drives the
  // halo layer span alone (separate bloom-like outer glow). ────────────────
  const coreHalf = Math.max(cfg.thickness, 0.0005) * 0.5;
  const mainGeo = ribbonGeometry(0, mainCount, 0, 1);
  const prof = cfg.glow.profile === 'triangle' ? 1 : 0;
  const ribbonSpecs = [
    {
      layers: 'core' as const,
      halfWidth: coreHalf,
      intensity: cfg.intensity,
      glowIntensity: cfg.glow.intensity,
      haloIntensity: cfg.glow.intensity * 0.4,
    },
    {
      layers: 'sheath' as const,
      halfWidth: coreHalf * 2,
      intensity: cfg.intensity,
      glowIntensity: cfg.glow.intensity * 0.8,
      haloIntensity: cfg.glow.intensity * 0.4,
    },
    {
      layers: 'halo' as const,
      halfWidth: coreHalf * Math.max(2, cfg.glow.width),
      intensity: cfg.intensity,
      glowIntensity: cfg.glow.intensity * 0.8,
      haloIntensity: cfg.glow.intensity * 0.4,
    },
  ];
  const mainMeshes: {
    mesh: THREE.Mesh;
    handles: ReturnType<typeof createElectricArcRibbonMaterial>;
    layer: 'core' | 'sheath' | 'halo';
  }[] = [];
  for (const spec of ribbonSpecs) {
    const handles = createElectricArcRibbonMaterial(
      arcBuffer,
      widthBuffer,
      totalSamples,
      {
        coreColor: coreColorVec,
        arcColor: arcColorVec,
        halfWidth: spec.halfWidth,
        intensity: spec.intensity,
        glowIntensity: spec.glowIntensity,
        haloIntensity: spec.haloIntensity,
        layers: spec.layers,
        profileMode: prof,
      }
    );
    const mesh = new THREE.Mesh(mainGeo, handles.material);
    mesh.frustumCulled = false;
    root.add(mesh);
    mainMeshes.push({ mesh, handles, layer: spec.layers });
  }

  // ── branches: one extra geometry + one extra draw, batched (§23) ─────────
  const branchNum =
    cfg.branches.enabled && cfg.branches.maxCount > 0
      ? Math.min(8, Math.round(cfg.branches.maxCount))
      : 0;
  const branchMeshes: {
    mesh: THREE.Mesh;
    handles: ReturnType<typeof createElectricArcRibbonMaterial>;
  }[] = [];
  const branchGeos: THREE.BufferGeometry[] = [];
  if (branchNum > 0) {
    const [ts0, ts1] = cfg.branches.thicknessScale;
    const wMid = (ts0 + ts1) * 0.5;
    const start0 = mainCount;
    const geo = ribbonGeometry(start0, branchNum * ARC_BRANCH_SAMPLES, 0, 1);
    const branchHandles = createElectricArcRibbonMaterial(
      arcBuffer,
      widthBuffer,
      totalSamples,
      {
        coreColor: coreColorVec,
        arcColor: arcColorVec,
        halfWidth: coreHalf * wMid * 2,
        intensity: cfg.intensity * 0.8,
        glowIntensity: cfg.glow.intensity * 0.7,
        haloIntensity: cfg.glow.intensity * 0.3,
        layers: 'core',
        profileMode: prof,
      }
    );
    const mesh = new THREE.Mesh(geo, branchHandles.material);
    mesh.frustumCulled = false;
    root.add(mesh);
    branchGeos.push(geo);
    branchMeshes.push({ mesh, handles: branchHandles });
  }

  // ── contacts: procedural TSL billboard quads (§20) ────────────────────────
  const contactA = new Vector3(cfg.start.x, cfg.start.y, cfg.start.z);
  const contactB = new Vector3(cfg.end.x, cfg.end.y, cfg.end.z);
  const contactMeshes: {
    mesh: THREE.Mesh;
    handles: ReturnType<typeof createElectricContactMaterial>;
    center: Vector3;
  }[] = [];
  const contactGeo = contactGeometry();
  if (cfg.contact.enabled) {
    for (const center of [contactA, contactB]) {
      const handles = createElectricContactMaterial({
        color: arcColorVec,
        coreColor: coreColorVec,
        center,
        halfSize: Math.max(cfg.contact.radius, 0.005),
        intensity: cfg.contact.intensity,
        flicker: uniforms.globalFlicker,
      });
      const mesh = new THREE.Mesh(contactGeo, handles.material);
      mesh.frustumCulled = false;
      root.add(mesh);
      contactMeshes.push({ mesh, handles, center });
    }
  }

  // ── per-frame update: uniform writes only (§41) ──────────────────────────
  const update: ElectricArcBackendInstance['update'] = (cycle, start, end) => {
    (uniforms.start as unknown as { value: Vector3 }).value.copy(start);
    (uniforms.end as unknown as { value: Vector3 }).value.copy(end);

    _dir.subVectors(end, start);
    const dist = Math.max(_dir.length(), 1e-4);
    _dir.multiplyScalar(1 / dist);
    if (_dir.y < 0.85 && _dir.y > -0.85) _helper.set(0, 1, 0);
    else _helper.set(1, 0, 0);
    nBasis(_ub, _dir, _helper);
    nBasis(_vb, _dir, _ub);
    (uniforms.basisU as unknown as { value: Vector3 }).value.copy(_ub);
    (uniforms.basisV as unknown as { value: Vector3 }).value.copy(_vb);

    const epoch = Math.floor(cycle.elapsed * cfg.flickerHz * cfg.speed);
    const flicker = globalFlicker(cfg.seed, epoch);

    setNum(uniforms.time, cycle.elapsed * cfg.speed);
    setNum(uniforms.epoch, epoch);
    setNum(uniforms.globalFlicker, flicker);
    setNum(uniforms.amp, cfg.amplitude);
    setNum(uniforms.knots, cfg.coarseKnots);
    setNum(uniforms.microF, cfg.microFrequency);
    setNum(uniforms.pin, cfg.endpointPinning);
    setNum(uniforms.brightnessVar, cfg.brightnessVariation);
    setNum(uniforms.intensity, cfg.intensity);
    setNum(
      uniforms.branchProb,
      cfg.branchProbability || cfg.branches.probability
    );

    if (contactMeshes.length === 2) {
      contactMeshes[0].center.copy(start);
      contactMeshes[1].center.copy(end);
    }
    return flicker;
  };

  const updateLive = (patch: Partial<ElectricArcConfig>): void => {
    void patch;
    setColors();
    const pMode: 0 | 1 = cfg.glow.profile === 'triangle' ? 1 : 0;
    for (const m of mainMeshes) {
      const h = m.handles;
      setNum(h.profileMode, pMode);
      if (m.layer === 'core') {
        setNum(h.halfWidth, coreHalfOf());
        setNum(h.intensity, cfg.intensity);
      } else if (m.layer === 'sheath') {
        setNum(h.halfWidth, coreHalfOf() * 2);
        setNum(h.glowIntensity, cfg.glow.intensity * 0.8);
        setNum(h.haloIntensity, cfg.glow.intensity * 0.4);
      } else {
        setNum(h.halfWidth, coreHalfOf() * Math.max(2, cfg.glow.width));
        setNum(h.glowIntensity, cfg.glow.intensity * 0.8);
        setNum(h.haloIntensity, cfg.glow.intensity * 0.4);
      }
    }
    for (const b of branchMeshes) {
      setNum(b.handles.halfWidth, coreHalfOf() * midThicknessScale() * 2);
      setNum(b.handles.intensity, cfg.intensity * 0.8);
      setNum(b.handles.profileMode, pMode);
    }
    for (const c of contactMeshes) {
      setNum(c.handles.intensity, cfg.contact.intensity);
      setNum(c.handles.halfSize, Math.max(cfg.contact.radius, 0.005));
    }
  };

  const coreHalfOf = (): number => Math.max(cfg.thickness, 0.0005) * 0.5;
  const midThicknessScale = (): number => {
    const [ts0, ts1] = cfg.branches.thicknessScale;
    return (ts0 + ts1) * 0.5;
  };

  let disposed = false;
  return {
    root,
    update,
    updateLive,
    backend: SimulationBackend.GPU,
    computeNode: pipeline.computeNode,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      for (const m of mainMeshes) m.handles.material.dispose();
      mainGeo.dispose();
      for (const b of branchMeshes) {
        b.handles.material.dispose();
      }
      for (const geo of branchGeos) geo.dispose();
      for (const c of contactMeshes) c.handles.material.dispose();
      contactGeo.dispose();
      pipeline.dispose();
    },
  };
}
