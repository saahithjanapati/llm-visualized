import * as THREE from 'three';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { animatePrismMultiplyTransition } from '../src/engine/layers/gpt2MultiplyTransition.js';

class ImmediateTween {
    constructor(state) {
        this.state = state;
        this._target = null;
        this._onUpdate = null;
        this._onComplete = null;
    }

    to(target) {
        this._target = target;
        return this;
    }

    easing() {
        return this;
    }

    onUpdate(fn) {
        this._onUpdate = fn;
        return this;
    }

    onComplete(fn) {
        this._onComplete = fn;
        return this;
    }

    start() {
        if (this._target && this.state && typeof this.state === 'object') {
            Object.assign(this.state, this._target);
        }
        if (typeof this._onUpdate === 'function') {
            this._onUpdate(this.state);
        }
        if (typeof this._onComplete === 'function') {
            this._onComplete(this.state);
        }
        return this;
    }

    stop() {
        return this;
    }
}

let previousTWEEN;

beforeEach(() => {
    previousTWEEN = globalThis.TWEEN;
    globalThis.TWEEN = {
        Tween: ImmediateTween,
        Easing: {
            Back: { Out: (t) => t },
            Quadratic: {
                InOut: (t) => t,
                Out: (t) => t
            }
        }
    };
});

afterEach(() => {
    vi.restoreAllMocks();
    if (previousTWEEN === undefined) {
        delete globalThis.TWEEN;
    } else {
        globalThis.TWEEN = previousTWEEN;
    }
});

describe('animatePrismMultiplyTransition', () => {
    it('reuses the pulse handoff to reveal the multiply result', () => {
        const root = new THREE.Group();
        const sourceGroup = new THREE.Group();
        const resultGroup = new THREE.Group();
        root.add(sourceGroup);
        root.add(resultGroup);

        sourceGroup.scale.set(1, 1, 1);
        resultGroup.visible = false;
        resultGroup.scale.set(1, 1, 1);

        const sourceVec = {
            group: sourceGroup,
            mesh: {
                material: new THREE.MeshBasicMaterial({
                    opacity: 1,
                    transparent: false,
                    depthWrite: true
                })
            }
        };
        const multResult = {
            group: resultGroup,
            mesh: {
                material: new THREE.MeshBasicMaterial({
                    opacity: 0.35,
                    transparent: true,
                    depthWrite: false
                })
            }
        };
        const scaleParam = { group: new THREE.Group() };
        scaleParam.group.visible = true;
        const onComplete = vi.fn();
        const emitProgress = vi.fn();

        const originalSet = multResult.group.scale.set.bind(multResult.group.scale);
        const scaleCalls = [];
        multResult.group.scale.set = (x, y, z) => {
            scaleCalls.push([x, y, z]);
            return originalSet(x, y, z);
        };

        animatePrismMultiplyTransition({
            sourceVec,
            multResult,
            scaleParam,
            instant: true,
            emitProgress,
            onComplete
        });

        expect(scaleCalls.some(([x]) => x > 1)).toBe(true);
        expect(sourceGroup.visible).toBe(false);
        expect(sourceGroup.parent).toBeNull();
        expect(scaleParam.group.visible).toBe(false);
        expect(multResult.group.visible).toBe(true);
        expect(multResult.group.scale.x).toBeCloseTo(1);
        expect(multResult.group.scale.y).toBeCloseTo(1);
        expect(multResult.group.scale.z).toBeCloseTo(1);
        expect(multResult.mesh.material.opacity).toBe(1);
        expect(multResult.mesh.material.transparent).toBe(false);
        expect(emitProgress).toHaveBeenCalled();
        expect(onComplete).toHaveBeenCalledTimes(1);
    });
});
