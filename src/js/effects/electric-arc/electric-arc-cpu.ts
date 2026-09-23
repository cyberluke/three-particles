/**
 * CPU implementation of the Electric Arc backend (§17, §18, §19).
 *
 * Owns the centerline generation in plain scalar JavaScript and renders it
 * as three layered camera-facing ribbons with built-in `MeshBasicMaterial`
 * + `AdditiveBlending` (portable across `WebGLRenderer` and WebGPU fallback).
 * The GPU path in `./webgpu/` is the optimized one-draw-call version; the
 * physical model is identical.
 *
 * @module
 */
import * as THREE from 'three';
import { SimulationBackend } from '../three-particles/three-particles-enums.js';
import {
  createContactTexture,
  createContactSprites,
  createProfileTexture,
  updateContactSprites,
  writeProfileTexture,
} from './electric-arc-contact.js';
import {
  buildRibbonGeometry,
  type RibbonGeometry,
} from './electric-arc-cpu-geometry.js';
import {
  clamp,
  coarseOffset,
  globalFlicker,
  mixSeedScalar,
  organicOffset,
  pulseOffset,
  widthFactor,
} from './electric-arc-math.js';
import { impulseNoise, microNoise } from './electric-arc-noise.js';
import type {
  ElectricArcBackendInstance,
  ElectricArcConfig,
  NormalizedElectricArcConfig,
} from './electric-arc-types.js';

const N_ARC = 0.68;
const N_MICRO = 0.24;
const N_IMPULSE = 0.08;

// module scratch (single-threaded hot loop, no per-frame allocation)
const _dir = new THREE.Vector3();
const _helper = new THREE.Vector3();
const _basisU = new THREE.Vector3();
const _basisV = new THREE.Vector3();
const _tangent = new THREE.Vector3();
const _prev = new THREE.Vector3();
const _next = new THREE.Vector3();
const _camRight = new THREE.Vector3();

type Layer = {
  geometry: RibbonGeometry;
  mesh: THREE.Mesh;
  material: THREE.MeshBasicMaterial;
  halfWidth: number;
};

export function createElectricArcCpu(
  config: NormalizedElectricArcConfig
): ElectricArcBackendInstance {
  const cfg = config;
  const root = new THREE.Group();
  root.name = 'electric-arc-cpu';

  const seg = cfg.segments;
  const maxKnots = 21; // coarseKnots <= 20 (+1 interpolation lookahead)
  const center = new Float32Array(seg * 4);
  const widths = new Float32Array(seg);
  const knotsU = new Float32Array(maxKnots);
  const knotsV = new Float32Array(maxKnots);
  let lastEpoch = -1;
  let lastKnots = -1;
  let lastFlicker = 1;

  // ── textures ──────────────────────────────────────────────────────────────
  const PROFILE_SIZE = 48;
  const coreMap = createProfileTexture(PROFILE_SIZE, 700, cfg.glow.profile);
  const innerMap = createProfileTexture(PROFILE_SIZE, 70, cfg.glow.profile);
  const haloMap = createProfileTexture(PROFILE_SIZE, 8, cfg.glow.profile);
  let lastProfile = cfg.glow.profile;

  const arcColor = new THREE.Color(cfg.color);
  const coreColor = new THREE.Color(cfg.coreColor);

  // ── layered ribbon (§18: three draws on the fallback path) ────────────────
  const makeLayer = (
    segments: number,
    halfWidth: number,
    color: THREE.Color,
    map: THREE.DataTexture
  ): Layer => {
    const geometry = buildRibbonGeometry(segments);
    const brightnessArr = new Float32Array(segments * 2 * 3);
    geometry.geometry.setAttribute(
      'color',
      new THREE.BufferAttribute(brightnessArr, 3)
    );
    const material = new THREE.MeshBasicMaterial({
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      toneMapped: false,
      vertexColors: true,
      side: THREE.DoubleSide,
      map,
    });
    material.color.copy(color);
    const mesh = new THREE.Mesh(geometry.geometry, material);
    mesh.frustumCulled = false;
    return { geometry, mesh, material, halfWidth };
  };

  // ── layered ribbon (§18, rebuilt glow): the core is owned by `thickness`
  // only; the glow layers form a separate bloom-like outer halo whose span
  // is driven by `glow.width` without touching the core thickness. ──────────
  const glowOuter = Math.max(2, cfg.glow.width);
  const layers: Layer[] = [
    makeLayer(
      seg,
      cfg.thickness * 0.5,
      coreColor.clone().multiplyScalar(cfg.intensity),
      coreMap
    ),
    makeLayer(
      seg,
      cfg.thickness * 0.5 * 2,
      arcColor.clone().multiplyScalar(cfg.glow.intensity * 0.8),
      innerMap
    ),
    makeLayer(
      seg,
      cfg.thickness * 0.5 * glowOuter,
      arcColor.clone().multiplyScalar(cfg.glow.intensity * 0.4),
      haloMap
    ),
  ];
  const layerAll: Layer[] = [...layers];
  for (const l of layers) root.add(l.mesh);

  // ── branches: lightweight quadratic-curve strips (§23), 8 samples (parity) ─
  const branchSegments = 8;
  type Branch = {
    core: Layer;
    glow: Layer;
    center: Float32Array;
    w: Float32Array;
  };
  const branches: Branch[] = [];
  if (cfg.branches.enabled && cfg.branches.maxCount > 0) {
    for (let b = 0; b < cfg.branches.maxCount; b++) {
      const [ts0, ts1] = cfg.branches.thicknessScale;
      const w = ts0 + (ts1 - ts0) * 0.5;
      const coreL = makeLayer(
        branchSegments,
        cfg.thickness * w,
        coreColor.clone().multiplyScalar(cfg.intensity * 0.8),
        coreMap
      );
      const glowL = makeLayer(
        branchSegments,
        cfg.thickness * w * 3,
        arcColor.clone().multiplyScalar(cfg.glow.intensity * 0.7),
        innerMap
      );
      root.add(coreL.mesh);
      root.add(glowL.mesh);
      const branch: Branch = {
        core: coreL,
        glow: glowL,
        center: new Float32Array(branchSegments * 4),
        w: new Float32Array(branchSegments),
      };
      branches.push(branch);
      layerAll.push(coreL, glowL);
    }
  }

  // ── contacts (sprites + generated radial texture, §20) ────────────────────
  const contactTex = createContactTexture(64);
  const contacts = cfg.contact.enabled
    ? createContactSprites(
        cfg.contact.radius,
        cfg.contact.intensity,
        cfg.color,
        contactTex
      )
    : null;
  if (contacts) root.add(contacts.group);

  // ── centerline (§5–§9) ────────────────────────────────────────────────────
  const coarseKind =
    cfg.chaosAlgorithm === 'pulse'
      ? 'pulse'
      : cfg.chaosAlgorithm === 'organic'
        ? 'organic'
        : 'linear';

  // knot levels (linear: lattice levels; pulse/organic evaluate per sample)
  const rebuildKnots = (epoch: number): void => {
    const kn = clamp(cfg.coarseKnots, 1, maxKnots - 1);
    for (let c = 0; c <= kn; c++) {
      knotsU[c] = coarseOffset(cfg.seed, epoch, c, kn, 0);
      knotsV[c] = coarseOffset(cfg.seed, epoch, c, kn, 1);
    }
    lastKnots = kn;
  };

  const brightness = (i: number, epoch: number, flicker: number): number =>
    clamp(
      flicker *
        (1 +
          cfg.brightnessVariation *
            (mixSeedScalar(i + 1, cfg.seed, epoch) * (1 / 4294967296) - 0.5) *
            2),
      0.15,
      1.6
    );

  const update: ElectricArcBackendInstance['update'] = (cycle, start, end) => {
    const sx = start.x,
      sy = start.y,
      sz = start.z;
    const ex = end.x,
      ey = end.y,
      ez = end.z;
    _dir.set(ex - sx, ey - sy, ez - sz);
    const dist = Math.max(_dir.length(), 1e-4);
    _dir.multiplyScalar(1 / dist);
    if (_dir.y < 0.85 && _dir.y > -0.85) _helper.set(0, 1, 0);
    else _helper.set(1, 0, 0);
    _basisU.crossVectors(_dir, _helper).normalize();
    _basisV.crossVectors(_dir, _basisU).normalize();
    const bUx = _basisU.x,
      bUy = _basisU.y,
      bUz = _basisU.z;
    const bVx = _basisV.x,
      bVy = _basisV.y,
      bVz = _basisV.z;

    const epoch = Math.floor(cycle.elapsed * cfg.flickerHz * cfg.speed);
    if (epoch !== lastEpoch || cfg.coarseKnots !== lastKnots) {
      rebuildKnots(epoch);
      lastEpoch = epoch;
    }
    const flicker = globalFlicker(cfg.seed, epoch);
    lastFlicker = flicker;

    const amp = cfg.amplitude;
    const pin = cfg.endpointPinning;
    const time = cycle.elapsed * cfg.speed;
    const mf = cfg.microFrequency;
    const kn = clamp(cfg.coarseKnots, 1, maxKnots - 1);
    const inv = 1 / (seg - 1);
    const ux0 = ex - sx,
      uy0 = ey - sy,
      uz0 = ez - sz;

    for (let i = 0; i < seg; i++) {
      const t = i * inv;
      const env =
        i === 0
          ? 0
          : i === seg - 1
            ? 0
            : Math.pow(Math.max(Math.sin(Math.PI * t), 0), pin);

      // coarse kinks: linear interpolates lattice knots (§7); pulse/organic
      // evaluate per-sample (their shapes are not linear between levels)
      let cu: number;
      let cvs: number;
      if (coarseKind === 'linear') {
        const cellF = t * kn;
        const cell0 = Math.floor(cellF);
        const fCell = cellF - cell0;
        const c1 = cell0 + 1 > kn ? kn : cell0 + 1;
        const ku0 = knotsU[cell0];
        const ku1 = knotsU[c1];
        const kv0 = knotsV[cell0];
        const kv1 = knotsV[c1];
        cu = ku0 + (ku1 - ku0) * fCell;
        cvs = kv0 + (kv1 - kv0) * fCell;
      } else if (coarseKind === 'pulse') {
        cu = pulseOffset(cfg.seed, epoch, t, kn, 0);
        cvs = pulseOffset(cfg.seed, epoch, t, kn, 1);
      } else {
        cu = organicOffset(cfg.seed, epoch, t, kn, 0);
        cvs = organicOffset(cfg.seed, epoch, t, kn, 1);
      }

      const ou =
        N_ARC * cu +
        N_MICRO * microNoise(t, time, mf, 0.0) +
        N_IMPULSE * impulseNoise(t, time, mf, 0.0);
      const ov =
        N_ARC * cvs +
        N_MICRO * microNoise(t, time, mf, 1.0) +
        N_IMPULSE * impulseNoise(t, time, mf, 1.0);

      let x = sx + ux0 * t;
      let y = sy + uy0 * t;
      let z = sz + uz0 * t;
      x += (bUx * ou + bVx * ov) * amp * env;
      y += (bUy * ou + bVy * ov) * amp * env;
      z += (bUz * ou + bVz * ov) * amp * env;

      // variable width: full core when straight, thinned where rotated
      widths[i] = widthFactor(ou * env, ov * env);

      // explicit exact endpoints (§6)
      if (i === 0) {
        x = sx;
        y = sy;
        z = sz;
      } else if (i === seg - 1) {
        x = ex;
        y = ey;
        z = ez;
      }

      const o = i * 4;
      center[o] = x;
      center[o + 1] = y;
      center[o + 2] = z;
      center[o + 3] = brightness(i, epoch, flicker);
    }

    // branches
    if (branches.length > 0) {
      const prob = cfg.branchProbability || cfg.branches.probability || 0.04;
      const invB = 1 / (branchSegments - 1);
      const [l0, l1] = cfg.branches.length;
      const [ts0, ts1] = cfg.branches.thicknessScale;
      for (let b = 0; b < branches.length; b++) {
        const bc = branches[b];
        const h = mixSeedScalar(b + 1, cfg.seed, epoch) * (1 / 4294967296);
        const active = h < prob ? 1 : 0;
        const oT = mixSeedScalar(b, cfg.seed, 11) * (1 / 4294967296);
        const originIdx = Math.min(Math.round(oT * (seg - 1)), seg - 1);
        const oo0 = originIdx * 4;
        const ox = center[oo0],
          oy = center[oo0 + 1],
          oz = center[oo0 + 2];
        const dh = (ch: number): number =>
          mixSeedScalar(b, cfg.seed, ch) * (2 / 4294967296) - 1;
        let dxn = dh(13),
          dyn = dh(14),
          dzn = dh(15);
        const dl = Math.sqrt(dxn * dxn + dyn * dyn + dzn * dzn) || 1;
        dxn /= dl;
        dyn /= dl;
        dzn /= dl;
        const len =
          l0 + (l1 - l0) * (mixSeedScalar(b, cfg.seed, 16) * (1 / 4294967296));
        const scaleA = active ? len : 0;
        const w =
          ts0 +
          (ts1 - ts0) * (mixSeedScalar(b, cfg.seed, 17) * (1 / 4294967296));
        // inherit the origin sample's variable-width factor on the branch
        const wf = widths[originIdx];
        for (let i = 0; i < branchSegments; i++) bc.w[i] = wf;
        // quadratic curve: origin -> control(d*0.5 + bend*U) -> end(d)
        const bend = dh(18) * 0.28;
        const cX = ox + (dxn * 0.5 + bUx * bend) * scaleA;
        const cY = oy + (dyn * 0.5 + bUy * bend) * scaleA;
        const cZ = oz + (dzn * 0.5 + bUz * bend) * scaleA;
        const eX = ox + dxn * scaleA;
        const eY = oy + dyn * scaleA;
        const eZ = oz + dzn * scaleA;
        for (let i = 0; i < branchSegments; i++) {
          const s = i * invB;
          const w0 = (1 - s) * (1 - s);
          const w1 = 2 * (1 - s) * s;
          const w2 = s * s;
          const o = i * 4;
          bc.center[o] = w0 * ox + w1 * cX + w2 * eX;
          bc.center[o + 1] = w0 * oy + w1 * cY + w2 * eY;
          bc.center[o + 2] = w0 * oz + w1 * cZ + w2 * eZ;
          bc.center[o + 3] = active ? 1 : 0;
        }
        // width for this epoch
        ((bc.core.halfWidth = cfg.thickness * w),
          (bc.glow.halfWidth = cfg.thickness * w * 3));
        // brightness attribute
        for (const l of [bc.core, bc.glow] as Layer[]) {
          const arr = l.geometry.geometry.getAttribute(
            'color'
          ) as THREE.BufferAttribute;
          const fa = arr.array as Float32Array;
          for (let i = 0; i < branchSegments; i++) {
            const bv = active ? 1 : 0;
            const vi = i * 6;
            fa[vi] = bv;
            fa[vi + 1] = bv;
            fa[vi + 2] = bv;
            fa[vi + 3] = bv;
            fa[vi + 4] = bv;
            fa[vi + 5] = bv;
          }
          arr.needsUpdate = true;
        }
      }
    }

    // bake brightness into the main ribbon layer color attributes (2 verts/sample)
    for (const l of layers) {
      const arr = l.geometry.geometry.getAttribute(
        'color'
      ) as THREE.BufferAttribute;
      const fa = arr.array as Float32Array;
      for (let i = 0; i < seg; i++) {
        const bv = center[i * 4 + 3];
        const vi = i * 6;
        fa[vi] = bv;
        fa[vi + 1] = bv;
        fa[vi + 2] = bv;
        fa[vi + 3] = bv;
        fa[vi + 4] = bv;
        fa[vi + 5] = bv;
      }
      arr.needsUpdate = true;
    }

    // contacts with the shared discharge flicker (§20)
    if (contacts) {
      updateContactSprites(
        contacts.start,
        contacts.end,
        sx,
        sy,
        sz,
        ex,
        ey,
        ez,
        flicker,
        contacts.baseScale
      );
    }

    return flicker;
  };

  // ── camera-facing expansion in onBeforeRender (§19) ──────────────────────
  const hook = (
    layer: Layer,
    src: Float32Array,
    n: number,
    w?: Float32Array
  ): void => {
    const posAttr = layer.geometry.geometry.getAttribute(
      'position'
    ) as THREE.BufferAttribute;
    const out = layer.geometry.positionArray;
    layer.mesh.onBeforeRender = (_r, _s, camera) => {
      const e = camera.matrixWorld.elements;
      _camRight.set(e[0], e[1], e[2]);
      const camX = e[12],
        camY = e[13],
        camZ = e[14];
      for (let i = 0; i < n; i++) {
        const o = i * 4;
        const cx = src[o],
          cy = src[o + 1],
          cz = src[o + 2];
        const pi = i > 0 ? (i - 1) * 4 : o;
        const ni = i < n - 1 ? (i + 1) * 4 : o;
        _prev.set(src[pi], src[pi + 1], src[pi + 2]);
        _next.set(src[ni], src[ni + 1], src[ni + 2]);
        _tangent.subVectors(_next, _prev);
        const tl = _tangent.length();
        if (tl < 0.0001) {
          _tangent.set(0, 1, 0);
        } else {
          _tangent.multiplyScalar(1 / tl);
        }
        const vx = camX - cx,
          vy = camY - cy,
          vz = camZ - cz;
        const vl = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1;
        const ux = vx / vl,
          uy = vy / vl,
          uz = vz / vl;
        const tx = _tangent.x,
          ty = _tangent.y,
          tz = _tangent.z;
        let px = ty * uz - tz * uy;
        let py = tz * ux - tx * uz;
        let pz = tx * uy - ty * ux;
        const pl = Math.sqrt(px * px + py * py + pz * pz);
        if (pl < 0.0001) {
          const cr = _camRight;
          const d = cr.x * tx + cr.y * ty + cr.z * tz;
          px = cr.x - tx * d;
          py = cr.y - ty * d;
          pz = cr.z - tz * d;
          const fl = Math.sqrt(px * px + py * py + pz * pz) || 1;
          px /= fl;
          py /= fl;
          pz /= fl;
        } else if (pl < 0.7) {
          // smooth blend toward camera-right fallback over [0, 0.7] (§14/§19)
          const w = pl / 0.7;
          const nx = px / pl,
            ny = py / pl,
            nz = pz / pl;
          const cr = _camRight;
          const d = cr.x * tx + cr.y * ty + cr.z * tz;
          let fx = cr.x - tx * d;
          let fy = cr.y - ty * d;
          let fz = cr.z - tz * d;
          const fl = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1;
          fx /= fl;
          fy /= fl;
          fz /= fl;
          px = nx + (fx - nx) * (1 - w);
          py = ny + (fy - ny) * (1 - w);
          pz = nz + (fz - nz) * (1 - w);
          const nl = Math.sqrt(px * px + py * py + pz * pz) || 1;
          px /= nl;
          py /= nl;
          pz /= nl;
        } else {
          px /= pl;
          py /= pl;
          pz /= pl;
        }
        const hw = w ? layer.halfWidth * w[i] : layer.halfWidth;
        const oL = i * 6;
        const oR = oL + 3;
        out[oL] = cx - px * hw;
        out[oL + 1] = cy - py * hw;
        out[oL + 2] = cz - pz * hw;
        out[oR] = cx + px * hw;
        out[oR + 1] = cy + py * hw;
        out[oR + 2] = cz + pz * hw;
      }
      posAttr.needsUpdate = true;
    };
  };

  const refreshHooks = (): void => {
    layers.forEach((l) => hook(l, center, seg, widths));
    for (const b of branches) {
      hook(b.core, b.center, branchSegments, b.w);
      hook(b.glow, b.center, branchSegments, b.w);
    }
  };
  refreshHooks();

  const updateLive = (patch: Partial<ElectricArcConfig>): void => {
    void patch;
    // colors / widths are numeric and read fresh from `cfg` on every frame;
    // re-sync the built-in material colors now.
    const cc = new THREE.Color(cfg.coreColor).multiplyScalar(cfg.intensity);
    layers[0].material.color.copy(cc);
    layers[1].material.color
      .copy(new THREE.Color(cfg.color))
      .multiplyScalar(cfg.glow.intensity * 0.8);
    layers[2].material.color
      .copy(new THREE.Color(cfg.color))
      .multiplyScalar(cfg.glow.intensity * 0.4);
    // rebuilt glow: core = `thickness` only; glow.halo span = `glow.width`
    layers[0].halfWidth = cfg.thickness * 0.5;
    layers[1].halfWidth = cfg.thickness * 0.5 * 2;
    layers[2].halfWidth = cfg.thickness * 0.5 * Math.max(2, cfg.glow.width);
    // cross-section shape switch re-encodes the 1D profile textures in place
    if (
      cfg.glow.profile !== lastProfile &&
      patch?.glow?.profile !== undefined
    ) {
      lastProfile = cfg.glow.profile;
      writeProfileTexture(coreMap, PROFILE_SIZE, 700, lastProfile);
      writeProfileTexture(innerMap, PROFILE_SIZE, 70, lastProfile);
      writeProfileTexture(haloMap, PROFILE_SIZE, 8, lastProfile);
    }
    if (contacts) {
      const col = new THREE.Color(cfg.color);
      for (const m of contacts.materials) {
        m.color.setRGB(
          col.r * cfg.contact.intensity,
          col.g * cfg.contact.intensity,
          col.b * cfg.contact.intensity
        );
      }
      (contacts as { baseScale: number }).baseScale = Math.max(
        cfg.contact.radius * 2,
        0.01
      );
    }
  };

  void lastFlicker;
  void layerAll;
  void refreshHooks;

  let disposed = false;
  return {
    root,
    update: (cycle, start, end) => update(cycle, start, end),
    updateLive,
    backend: SimulationBackend.CPU,
    computeNode: null,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      for (const l of layerAll) {
        l.mesh.onBeforeRender =
          undefined as unknown as THREE.Object3D['onBeforeRender'];
        l.material.dispose();
        l.geometry.geometry.dispose();
      }
      coreMap.dispose();
      innerMap.dispose();
      haloMap.dispose();
      if (contacts) {
        contacts.materials.forEach((m) => m.dispose());
      }
      contactTex.dispose();
    },
  };
}
