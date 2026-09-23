/**
 * Procedural TSL billboard material for endpoint contact plasma (§20).
 *
 * Each contact is a 4-vertex quad; the material expands corners along the
 * camera right/up basis extracted from `cameraViewMatrix` (same approach as
 * the trail ribbon's camera-right usage) around a center uniform, and
 * shades a radial HDR profile: hot white core + colored corona, driven by
 * the shared discharge flicker so all elements blink in sync.
 *
 * @module
 */
import { AdditiveBlending, DoubleSide, Vector3 } from 'three';
import {
  Fn,
  attribute,
  cameraViewMatrix,
  vec3,
  vec4,
  float,
  exp,
  varyingProperty,
  uniform,
  Discard,
  type ShaderNodeObject,
  type Node,
} from 'three/tsl';
import { MeshBasicNodeMaterial } from 'three/webgpu';

export type ElectricContactParams = {
  color: Vector3;
  coreColor: Vector3;
  /** shared reference, copied per frame by the owner. */
  center: Vector3;
  halfSize: number;
  intensity: number;
  /** shared global flicker node (same value as the arc uses). */
  flicker: ShaderNodeObject<Node>;
};

export type ElectricContactMaterialHandles = {
  material: MeshBasicNodeMaterial;
  halfSize: ShaderNodeObject<Node>;
  intensity: ShaderNodeObject<Node>;
};

export function createElectricContactMaterial(
  params: ElectricContactParams
): ElectricContactMaterialHandles {
  const uColor = uniform(params.color);
  const uCore = uniform(params.coreColor);
  const uCenter = uniform(params.center);
  const uHalf = uniform(float(params.halfSize));
  const uIntensity = uniform(float(params.intensity));

  const vUv = varyingProperty('vec2', 'vContactUV');

  const positionNode = Fn((): ShaderNodeObject<Node> => {
    // packed corner attribute: position.xy = (-1..1, -1..1)
    const corner = attribute('position', 'vec2');
    vUv.assign(attribute('uv', 'vec2'));

    const camRight = vec3(
      cameraViewMatrix.element(0).element(0),
      cameraViewMatrix.element(1).element(0),
      cameraViewMatrix.element(2).element(0)
    );
    const camUp = vec3(
      cameraViewMatrix.element(0).element(1),
      cameraViewMatrix.element(1).element(1),
      cameraViewMatrix.element(2).element(1)
    );

    return uCenter
      .add(camRight.mul(corner.x.mul(uHalf)))
      .add(camUp.mul(corner.y.mul(uHalf)));
  })();

  const colorNode = Fn((): ShaderNodeObject<Node> => {
    const x = vUv.x.sub(float(0.5)).mul(float(2));
    const y = vUv.y.sub(float(0.5)).mul(float(2));
    const d2 = x.mul(x).add(y.mul(y));

    const core = exp(d2.mul(float(-9)));
    const corona = exp(d2.mul(float(-2.2)));

    const kc = core.mul(uIntensity).mul(params.flicker);
    const kw = corona.mul(params.flicker);

    const r = uCore.x.mul(kc).add(uColor.x.mul(kw));
    const g = uCore.y.mul(kc).add(uColor.y.mul(kw));
    const b = uCore.z.mul(kc).add(uColor.z.mul(kw));

    const alpha = core.add(corona.mul(float(0.6)));
    Discard(alpha.lessThan(float(0.001)));
    return vec4(r, g, b, alpha);
  })();

  const material = new MeshBasicNodeMaterial();
  material.transparent = true;
  material.blending = AdditiveBlending;
  material.depthTest = true;
  material.depthWrite = false;
  material.toneMapped = false;
  material.fog = false;
  material.side = DoubleSide;
  material.positionNode = positionNode;
  material.colorNode = colorNode;

  return { material, halfSize: uHalf, intensity: uIntensity };
}
