/**
 * Minimal type shim for `@types/three@0.183.x` / `@types/three@0.186.x` subpath
 * exports. Both tarballs only list `Node` on the `three/webgpu` subpath (via
 * `three/webgpu → nodes/Nodes.js → nodes/core/Node.js`) and neither publishes
 * the `ShaderNodeObject<T>` alias older three typings did. The engine files all
 * use `import { type Node, type ShaderNodeObject } from 'three/tsl'`, which
 * produced TS2305 against `@types/three@0.183.1` under Jest's `bundler`-style
 * module resolution.
 *
 * The TSL node objects are proxy-shaped (`Fn(...)`, `.mul(...)`, `.element(i)`
 * etc.), so an `any`-flavoured declaration is enough for the whole chain and
 * matches three's actual runtime surface.
 *
 * The `export {}` at the end turns this file into a *module*, so the
 * `declare module` blocks are augmentations (they merge with `@types/three`)
 * instead of full replacements.
 */

declare module 'three/tsl' {
  // TSL `Node` proxy: method-chained scalar / vector / texture expression
  // object. `any` covers every `mul/sub/element/add/…` helper without having
  // to mirror three's ~120 node classes here.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export type Node = any;
  // `ShaderNodeObject<T>` used to be `T extends { type: infer U } ? Node<U> : T`
  // in older typings; with the `any` above the identity alias preserves every
  // call-site value.
  export type ShaderNodeObject<T = unknown> = T;
}

declare module 'three/webgpu' {
  export type ShaderNodeObject<T = unknown> = T;
}

export {};
