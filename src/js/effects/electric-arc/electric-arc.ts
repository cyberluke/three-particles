/**
 * Public `createElectricArc` factory + lifecycle facade (§1, §24–§26).
 *
 * One subsystem, two optimal execution paths, identical public API:
 *
 *   - GPU (WebGPU compute-capable renderer, `/webgpu` registered):
 *     compute storage buffer + TSL analytic one-draw filament.
 *   - CPU (everything else): scalar centerline in JS + layered
 *     `MeshBasicMaterial` ribbons with `AdditiveBlending`.
 *
 * Backend decision preserves the engine's `SimulationBackend` semantics:
 * AUTO -> GPU where possible, CPU otherwise; CPU -> always CPU;
 * GPU -> compute when available, CPU fallback otherwise.
 *
 * @module
 */
import * as THREE from 'three';
import { SimulationBackend } from '../three-particles/three-particles-enums.js';
import { resolveSimulationBackend } from '../three-particles/three-particles-renderer-detect.js';
import {
  mergeLiveConfig,
  normalizeElectricArcConfig,
  touchesStructuralField,
} from './electric-arc-config.js';
import { createElectricArcCpu } from './electric-arc-cpu.js';
import {
  getElectricArcGPUFactory,
  getElectricArcGPURenderer,
} from './electric-arc-gpu-registry.js';
import {
  createArcLighting,
  type ArcLighting,
} from './electric-arc-lighting.js';
import { rotateZ2 } from './electric-arc-math.js';
import {
  createArcSparks,
  pushSparkLiveConfig,
  type ArcSparks,
} from './electric-arc-sparks.js';
import type {
  ElectricArc,
  ElectricArcBackendInstance,
  ElectricArcBinding,
  ElectricArcConfig,
  ElectricArcEndpointRef,
  ElectricArcFrame,
  ElectricArcRuntimeEndpoints,
  NormalizedElectricArcConfig,
} from './electric-arc-types.js';

const _zero = new THREE.Vector3();
const _tmp = new THREE.Vector3();

// module scratch for endpoint composition (allocation-free per-frame hot path)
const _helper = new THREE.Vector3();
const _local = new THREE.Vector3();
const _axis = new THREE.Vector3();
const _euler = new THREE.Euler(0, 0, 0, 'XYZ');
const _rotM = new THREE.Matrix4();
const DEG = Math.PI / 180;

const toLocal = (
  p: { x?: number; y?: number; z?: number } | THREE.Vector3
): THREE.Vector3 => {
  _tmp.set(p.x ?? 0, p.y ?? 0, p.z ?? 0);
  return _tmp;
};

const resolveEndpoint = (
  ref:
    ElectricArcEndpointRef | { x?: number; y?: number; z?: number } | undefined,
  out: THREE.Vector3
): void => {
  if (!ref) return;
  if (ref instanceof THREE.Object3D) {
    ref.updateWorldMatrix(true, false);
    out.setFromMatrixPosition(ref.matrixWorld);
    return;
  }
  const maybe = ref as { object?: THREE.Object3D; offset?: THREE.Vector3 };
  if (maybe.object instanceof THREE.Object3D) {
    maybe.object.updateWorldMatrix(true, false);
    const { x = 0, y = 0, z = 0 } = maybe.offset ?? _zero;
    _tmp.set(x, y, z);
    out.copy(
      maybe.object.matrixWorld
        ? _tmp.applyMatrix4(maybe.object.matrixWorld)
        : out
    );
    return;
  }
  const p = ref as { x?: number; y?: number; z?: number };
  if (
    typeof p.x === 'number' ||
    typeof p.y === 'number' ||
    typeof p.z === 'number'
  ) {
    out.set(p.x ?? 0, p.y ?? 0, p.z ?? 0);
  }
};

/**
 * Create an electric arc effect between two endpoints.
 *
 * @example
 * ```typescript
 * const arc = createElectricArc({
 *   start: new THREE.Vector3(-1.1, 0, 0),
 *   end: new THREE.Vector3(1.1, 0, 0),
 *   color: '#baff63',
 *   coreColor: '#fffde0',
 *   thickness: 0.04,
 *   chaos: 0.19,
 *   seed: 271,
 *   intensity: 12,
 *   glow: { enabled: true, width: 7.5, intensity: 1.4 },
 *   contact: { enabled: true, radius: 0.075, intensity: 15 },
 *   lighting: { enabled: true },
 * });
 *
 * scene.add(arc.instance);
 *
 * // each frame
 * arc.update({ now: performance.now(), delta, elapsed });
 * if (arc.computeNode) renderer.compute(arc.computeNode);
 * renderer.render(scene, camera); // or your RenderPipeline
 * ```
 */
export function createElectricArc(config: ElectricArcConfig): ElectricArc {
  const normalized: NormalizedElectricArcConfig =
    normalizeElectricArcConfig(config);

  // ── backend resolution (§3) ───────────────────────────────────────────────
  const gpuFactory = getElectricArcGPUFactory();
  const gpuRenderer = gpuFactory ? getElectricArcGPURenderer() : null;
  const resolved = resolveSimulationBackend(
    gpuRenderer ?? undefined,
    normalized.simulationBackend
  );
  const useGPU = resolved === SimulationBackend.GPU && !!gpuFactory;

  const instance = new THREE.Group();
  instance.name = 'electric-arc';

  const createBackend = (
    cfg: NormalizedElectricArcConfig
  ): ElectricArcBackendInstance => {
    if (useGPU && gpuFactory) return gpuFactory.create(cfg);
    return createElectricArcCpu(cfg);
  };

  let backend: ElectricArcBackendInstance = createBackend(normalized);
  instance.add(backend.root);

  let lighting: ArcLighting | null = createArcLighting(normalized);
  let sparks: ArcSparks | null = createArcSparks(normalized);
  if (lighting) instance.add(lighting.group);
  if (sparks) instance.add(sparks.group);

  // ── endpoints (§24, modes: standalone / bound) ────────────────────────────
  // BASE runtime anchors: `bound` mode takes them from the live binding
  // (e.g. VIVERRA ribbon terminal frames) every frame; `standalone` mode uses
  // the persistent config start/end (or setEndpoints direct values).
  // Persistent user transforms live ONLY in the normalized config
  // (`startOffset`/`endOffset`/`startRotation`/`endRotation`) and are never
  // overwritten by anchor updates.
  let startDirect = normalized.start.clone();
  let endDirect = normalized.end.clone();
  let binding: ElectricArcBinding | null = null;

  const scratchStart = new THREE.Vector3();
  const scratchEnd = new THREE.Vector3();
  // effective endpoints for this frame's composition
  const _effStart = new THREE.Vector3();
  const _effEnd = new THREE.Vector3();
  // shared chord frame (fallback basis): N along base start->end, U/V helpers
  const _chordN = new THREE.Vector3();
  const _chordU = new THREE.Vector3();
  const _chordV = new THREE.Vector3();
  // per-endpoint frames: N = tangent, U = normal, V = binormal
  const _startN = new THREE.Vector3();
  const _startU = new THREE.Vector3();
  const _startV = new THREE.Vector3();
  const _endN = new THREE.Vector3();
  const _endU = new THREE.Vector3();
  const _endV = new THREE.Vector3();
  let _startFrameSource: 'object' | 'chord' = 'chord';
  let _endFrameSource: 'object' | 'chord' = 'chord';

  const resolveEndpoints = (): void => {
    if (binding) {
      resolveEndpoint(binding.start, scratchStart);
      resolveEndpoint(binding.end, scratchEnd);
    } else {
      scratchStart.copy(startDirect);
      scratchEnd.copy(endDirect);
    }
  };
  resolveEndpoints();

  /** Shared chord frame: N along base start->end, U/V orthonormal helpers. */
  const chordFrame = (): void => {
    _chordN.subVectors(scratchEnd, scratchStart);
    const len = _chordN.length();
    if (len < 1e-4 || !Number.isFinite(len)) {
      // degenerate zero-length chord: stable fallback basis
      _chordN.set(0, 0, 1);
      _chordU.set(1, 0, 0);
      _chordV.set(0, 1, 0);
      return;
    }
    _chordN.multiplyScalar(1 / len);
    if (_chordN.y < 0.85 && _chordN.y > -0.85) _helper.set(0, 1, 0);
    else _helper.set(1, 0, 0);
    // U = N x helper, V = N x U: right-handed frame (for a +X chord with
    // helper (0,1,0): U = (0,0,1), V = (0,-1,0)).
    _chordU.crossVectors(_chordN, _helper).normalize();
    _chordV.crossVectors(_chordN, _chordU).normalize();
  };

  /**
   * Object-axis frame for an endpoint ref: tangent = world +Z, normal =
   * world +X, binormal = world +Y of the bound Object3D. Returns false when
   * the ref carries no object or a degenerate matrix (chord frame is used).
   */
  const objectFrame = (
    ref:
      | ElectricArcEndpointRef
      | { x?: number; y?: number; z?: number }
      | undefined,
    n: THREE.Vector3,
    u: THREE.Vector3,
    v: THREE.Vector3
  ): boolean => {
    const obj =
      ref instanceof THREE.Object3D
        ? ref
        : (ref as { object?: THREE.Object3D } | undefined)?.object;
    if (!(obj instanceof THREE.Object3D)) return false;
    obj.updateWorldMatrix(true, false);
    const m = obj.matrixWorld.elements;
    u.set(m[0], m[1], m[2]);
    v.set(m[4], m[5], m[6]);
    n.set(m[8], m[9], m[10]);
    if (u.lengthSq() < 1e-8 || v.lengthSq() < 1e-8 || n.lengthSq() < 1e-8)
      return false; // degenerate scale: chord fallback is deterministic
    u.normalize();
    v.normalize();
    n.normalize();
    return true;
  };

  const copyChordInto = (
    n: THREE.Vector3,
    u: THREE.Vector3,
    v: THREE.Vector3
  ): void => {
    n.copy(_chordN);
    u.copy(_chordU);
    v.copy(_chordV);
  };

  /**
   * Effective anchor = base + frame * (R(endpointRotation) * offset).
   * ENDPOINT_FRAME_LOCAL axes: x along the frame normal (U), y along the
   * binormal (V), z along the tangent (N). Allocation-free.
   */
  const applyEndpointTransform = (
    base: THREE.Vector3,
    offset: THREE.Vector3,
    rot: { pitch: number; yaw: number; roll: number },
    u: THREE.Vector3,
    v: THREE.Vector3,
    n: THREE.Vector3,
    out: THREE.Vector3
  ): void => {
    out.copy(base);
    if (offset.lengthSq() === 0) return; // zero offset: rotation unobservable
    _euler.set(rot.pitch * DEG, rot.yaw * DEG, rot.roll * DEG, 'XYZ');
    _rotM.makeRotationFromEuler(_euler);
    _local.copy(offset).applyMatrix4(_rotM);
    out.addScaledVector(u, _local.x);
    out.addScaledVector(v, _local.y);
    out.addScaledVector(n, _local.z);
  };

  /**
   * Compose effective endpoints from base anchors + persistent transforms.
   * Frame priority per endpoint: bound Object3D own world axes, otherwise
   * the deterministic chord frame built once from the BASE anchors.
   */
  const composeEndpoints = (): void => {
    chordFrame();
    _startFrameSource = objectFrame(
      binding ? binding.start : startDirect,
      _startN,
      _startU,
      _startV
    )
      ? 'object'
      : (copyChordInto(_startN, _startU, _startV), 'chord');
    _endFrameSource = objectFrame(
      binding ? binding.end : endDirect,
      _endN,
      _endU,
      _endV
    )
      ? 'object'
      : (copyChordInto(_endN, _endU, _endV), 'chord');
    applyEndpointTransform(
      scratchStart,
      normalized.startOffset,
      normalized.startRotation,
      _startU,
      _startV,
      _startN,
      _effStart
    );
    applyEndpointTransform(
      scratchEnd,
      normalized.endOffset,
      normalized.endRotation,
      _endU,
      _endV,
      _endN,
      _effEnd
    );
    scratchStart.copy(_effStart);
    scratchEnd.copy(_effEnd);
  };

  // ── facade ────────────────────────────────────────────────────────────────
  let disposedFlag = false;

  const rebuild = (): void => {
    instance.remove(backend.root);
    backend.dispose();
    backend = createBackend(normalized);
    instance.add(backend.root);
  };

  const update = (cycle: {
    now: number;
    delta: number;
    elapsed: number;
  }): void => {
    if (disposedFlag) return;
    resolveEndpoints();
    // persistent per-endpoint transforms first (local frame), then the
    // whole-arc roll `rotationZ` (global), exactly as documented.
    composeEndpoints();
    // whole-arc rotation around Z (degrees): front and side read differently
    if (normalized.rotationZ) {
      rotateZ2(scratchStart, normalized.rotationZ);
      rotateZ2(scratchEnd, normalized.rotationZ);
    }
    const flicker = backend.update(cycle, scratchStart, scratchEnd);
    if (lighting) lighting.update(scratchStart, scratchEnd, flicker);
    if (sparks) sparks.update(cycle, scratchStart, scratchEnd);
  };

  const updateConfig = (patch: Partial<ElectricArcConfig>): void => {
    if (disposedFlag) return;
    const structural = touchesStructuralField(patch);
    mergeLiveConfig(normalized, patch);
    if (structural) {
      // recreate internal backend resources, keep the public identity stable
      if (lighting) {
        instance.remove(lighting.group);
        lighting.dispose();
        lighting = createArcLighting(normalized);
        if (lighting) instance.add(lighting.group);
      }
      if (sparks) {
        instance.remove(sparks.group);
        sparks.dispose();
        sparks = createArcSparks(normalized);
        if (sparks) instance.add(sparks.group);
      }
      rebuild();
    } else {
      backend.updateLive(patch);
      // spark sections are owned by child particle systems: forward the
      // merged numeric fields so lifetime/speed/size react immediately.
      if (sparks && patch.sparks !== undefined && sparks.systems.length > 0) {
        pushSparkLiveConfig(sparks.systems, normalized);
      }
    }
  };

  return {
    instance,
    update,
    updateConfig,
    setEndpoints(start, end) {
      // runtime base anchors only; persistent transforms are untouched
      startDirect = toPoint(start);
      endDirect = toPoint(end);
    },
    bindEndpoints(next) {
      binding = next;
    },
    clearEndpointBinding() {
      binding = null;
    },
    getRuntimeEndpoints(): ElectricArcRuntimeEndpoints {
      // read-only diagnostics of the last resolved state; small object
      // literals are fine here (editor diagnostics, not the render hot path)
      resolveEndpoints();
      const baseStart = {
        x: scratchStart.x,
        y: scratchStart.y,
        z: scratchStart.z,
      };
      const baseEnd = {
        x: scratchEnd.x,
        y: scratchEnd.y,
        z: scratchEnd.z,
      };
      composeEndpoints(); // also resolves per-endpoint frames + sources
      const startFrame = {
        tangent: { x: _startN.x, y: _startN.y, z: _startN.z },
        normal: { x: _startU.x, y: _startU.y, z: _startU.z },
        binormal: { x: _startV.x, y: _startV.y, z: _startV.z },
        frameSource: _startFrameSource,
      };
      const endFrame = {
        tangent: { x: _endN.x, y: _endN.y, z: _endN.z },
        normal: { x: _endU.x, y: _endU.y, z: _endU.z },
        binormal: { x: _endV.x, y: _endV.y, z: _endV.z },
        frameSource: _endFrameSource,
      };
      if (normalized.rotationZ) {
        rotateZ2(scratchStart, normalized.rotationZ);
        rotateZ2(scratchEnd, normalized.rotationZ);
      }
      const sourceId =
        binding &&
        (binding.start instanceof THREE.Object3D
          ? binding.start.name
          : (binding.start as { object?: THREE.Object3D }).object?.name) ||
          undefined;
      return {
        mode: binding ? 'bound' : 'standalone',
        ...(sourceId ? { sourceId } : {}),
        baseStart,
        baseEnd,
        effectiveStart: {
          x: scratchStart.x,
          y: scratchStart.y,
          z: scratchStart.z,
        },
        effectiveEnd: { x: scratchEnd.x, y: scratchEnd.y, z: scratchEnd.z },
        startFrame,
        endFrame,
      };
    },
    get backend() {
      return backend.backend;
    },
    get computeNode() {
      const arcNode = backend.computeNode;
      if (sparks && sparks.systems.length > 0) {
        const nodes: unknown[] = [];
        if (Array.isArray(arcNode)) nodes.push(...arcNode);
        else if (arcNode) nodes.push(arcNode);
        for (const s of sparks.systems) {
          const sn = s.computeNode;
          if (Array.isArray(sn)) nodes.push(...sn);
          else if (sn) nodes.push(sn);
        }
        return nodes.length > 0 ? nodes : null;
      }
      return arcNode;
    },
    dispose() {
      if (disposedFlag) return;
      disposedFlag = true;
      backend.dispose();
      lighting?.dispose();
      sparks?.dispose();
      for (const child of [...instance.children]) instance.remove(child);
    },
  };
}

/** normalize Point3D | Vector3 into a new Vector3 (no aliasing). */
const toPoint = (
  p: { x?: number; y?: number; z?: number } | THREE.Vector3
): THREE.Vector3 => new THREE.Vector3(p.x ?? 0, p.y ?? 0, p.z ?? 0);

void toLocal;
