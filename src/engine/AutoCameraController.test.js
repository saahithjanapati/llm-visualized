import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';
import { AutoCameraController } from './AutoCameraController.js';

function createController() {
    const controller = Object.create(AutoCameraController.prototype);
    controller._autoCameraFollow = false;
    controller._autoCameraSmoothValid = true;
    controller._autoCameraViewKey = 'ln';
    controller._autoCameraViewPendingKey = 'default';
    controller._autoCameraViewPendingSinceMs = 123;
    controller._autoCameraViewContext = { holdViewUntilLn2Inside: true };
    controller._autoCameraViewBlendAlphaTransition = 0.12;
    controller._autoCameraPostAddLockActive = false;
    controller._autoCameraPostAddLockUntilMs = 0;
    controller._autoCameraForceEmbedVocabStartLock = false;
    controller._autoCameraEmbedVocabOffsetsEnabled = false;
    controller._startupCameraIntroPlayed = true;
    controller._startupCameraOverviewPending = false;
    controller._overviewCameraPosition = new THREE.Vector3(0, 0, 0);
    controller._overviewCameraTarget = new THREE.Vector3(0, 0, 0);
    controller._autoCameraCenter = new THREE.Vector3(10, 20, 30);
    controller._autoCameraCurrentCameraOffset = new THREE.Vector3(1, 2, 3);
    controller._autoCameraCurrentTargetOffset = new THREE.Vector3(4, 5, 6);
    controller._autoCameraInspectorRef = new THREE.Vector3();
    controller._autoCameraViewFromCameraOffset = new THREE.Vector3();
    controller._autoCameraViewFromTargetOffset = new THREE.Vector3();
    controller._autoCameraViewToCameraOffset = new THREE.Vector3();
    controller._autoCameraViewToTargetOffset = new THREE.Vector3();
    controller._autoCameraDefaultCameraOffset = new THREE.Vector3(100, 0, 0);
    controller._autoCameraDefaultTargetOffset = new THREE.Vector3(0, 100, 0);
    controller._autoCameraLnCameraOffset = new THREE.Vector3(7, 8, 9);
    controller._autoCameraLnTargetOffset = new THREE.Vector3(10, 11, 12);
    controller._autoCameraTravelCameraOffset = new THREE.Vector3(13, 14, 15);
    controller._autoCameraTravelTargetOffset = new THREE.Vector3(16, 17, 18);

    controller._syncAutoCameraSemanticView = vi.fn();
    controller._resetAutoCameraReferenceMotion = vi.fn();
    controller._updateCameraOffsetOverlay = vi.fn();
    controller._updateAutoCameraFollow = vi.fn();
    controller._updateAutoCameraScaledOffsets = vi.fn();
    controller._resolveActiveLanePosition = vi.fn(() => ({ laneIndex: -1, laneCount: 0 }));
    controller._captureAutoCameraOffsets = vi.fn(() => false);
    controller._resolveAutoCameraViewKey = vi.fn(() => 'default');
    controller._resolveStableAutoCameraViewKey = vi.fn(() => 'ln');
    controller._resolveAutoCameraOffsetsForViewKey = vi.fn((key) => {
        if (key === 'ln') {
            return {
                cameraOffset: controller._autoCameraLnCameraOffset,
                targetOffset: controller._autoCameraLnTargetOffset
            };
        }
        if (key === 'travel') {
            return {
                cameraOffset: controller._autoCameraTravelCameraOffset,
                targetOffset: controller._autoCameraTravelTargetOffset
            };
        }
        return {
            cameraOffset: controller._autoCameraDefaultCameraOffset,
            targetOffset: controller._autoCameraDefaultTargetOffset
        };
    });
    controller._setAutoCameraOffsets = vi.fn();

    return controller;
}

describe('AutoCameraController follow re-enable', () => {
    it('preserves any pending semantic view transition when follow is turned back on', () => {
        const controller = createController();

        controller.setEnabled(true);

        expect(controller._autoCameraViewPendingKey).toBe('default');
        expect(controller._autoCameraViewPendingSinceMs).toBe(123);
    });

    it('uses the stabilized semantic view for smooth follow resets', () => {
        const controller = createController();

        controller._applyFollowReset({ smoothReset: true });

        expect(controller._resolveStableAutoCameraViewKey).toHaveBeenCalledWith(
            'default',
            controller._autoCameraViewContext
        );
        expect(controller._resolveAutoCameraOffsetsForViewKey).toHaveBeenCalledWith('ln', {});
        expect(controller._autoCameraViewKey).toBe('ln');
        expect(controller._autoCameraViewPendingKey).toBe('default');
        expect(controller._autoCameraViewPendingSinceMs).toBe(123);
    });

    it('jumps straight to the live semantic view when reset follow is re-enabled', () => {
        const controller = createController();

        controller._resolveAutoCameraViewKey = vi.fn(() => 'travel');

        controller._applyFollowReset({
            smoothReset: true,
            preferLiveSemanticView: true
        });

        expect(controller._resolveStableAutoCameraViewKey).not.toHaveBeenCalled();
        expect(controller._resolveAutoCameraOffsetsForViewKey).toHaveBeenCalledWith('travel', {});
        expect(controller._autoCameraViewKey).toBe('travel');
        expect(controller._autoCameraViewPendingKey).toBe('travel');
        expect(controller._autoCameraViewPendingSinceMs).toBe(0);
    });

    it('does not re-arm the startup overview when a live follow target already exists', () => {
        const controller = createController();

        controller._startupCameraIntroPlayed = false;
        controller._applyFollowReset = vi.fn();
        controller._resolveActiveLanePosition = vi.fn((ref) => {
            ref.set(10, 20, 30);
            return { laneIndex: 0, laneCount: 1 };
        });

        controller.setEnabled(true, { resetView: true, smoothReset: true });

        expect(controller._startupCameraOverviewPending).toBe(false);
        expect(controller._applyFollowReset).toHaveBeenCalledWith({
            smoothReset: true,
            preferLiveSemanticView: true
        });
    });
});

describe('AutoCameraController vector center sampling', () => {
    it('keeps the follow anchor stable across temporary vector scale pulses', () => {
        const controller = Object.create(AutoCameraController.prototype)
        const group = new THREE.Group()
        group.position.set(50, 100, -20)

        const mesh = {
            getMatrixAt(index, matrix) {
                if (index === 0) {
                    matrix.makeTranslation(-4, 10, 0)
                    return
                }
                matrix.makeTranslation(8, 14, 0)
            }
        }

        const vec = {
            group,
            mesh,
            rawData: [1, 2],
            instanceCount: 2
        }

        group.scale.setScalar(1)
        group.updateMatrixWorld(true)
        const baseCenter = new THREE.Vector3()
        expect(controller._getVectorWorldCenter(vec, baseCenter)).toBe(true)
        expect(baseCenter.x).toBeCloseTo(52, 6)
        expect(baseCenter.y).toBeCloseTo(112, 6)
        expect(baseCenter.z).toBeCloseTo(-20, 6)

        group.scale.setScalar(1.16)
        group.updateMatrixWorld(true)
        const pulsedCenter = new THREE.Vector3()
        expect(controller._getVectorWorldCenter(vec, pulsedCenter)).toBe(true)
        expect(pulsedCenter.x).toBeCloseTo(baseCenter.x, 6)
        expect(pulsedCenter.y).toBeCloseTo(baseCenter.y, 6)
        expect(pulsedCenter.z).toBeCloseTo(baseCenter.z, 6)
    })
})

describe('AutoCameraController lane reference anchoring', () => {
    it('anchors even lane counts to the midpoint between the two center residual lanes', () => {
        const controller = Object.create(AutoCameraController.prototype);
        const lanes = [-30, -10, 10, 30].map((zPos) => {
            const group = new THREE.Group();
            group.position.set(0, 100, zPos);
            group.updateMatrixWorld(true);
            return {
                originalVec: { group }
            };
        });

        controller._pipeline = {
            _layers: [{ lanes }],
            _currentLayerIdx: 0,
            _kvCacheDecodeActive: false
        };

        const target = new THREE.Vector3();
        const info = controller._resolveActiveLanePosition(target);

        expect(info.laneIndex).toBe(1);
        expect(info.laneLabel).toBe(2.5);
        expect(target.x).toBeCloseTo(0, 6);
        expect(target.y).toBeCloseTo(100, 6);
        expect(target.z).toBeCloseTo(0, 6);
    });

    it('returns the fractional lane label in the follow inspector reference', () => {
        const controller = Object.create(AutoCameraController.prototype);
        controller._autoCameraInspectorRef = new THREE.Vector3();
        controller._resolveActiveLanePosition = vi.fn((ref) => {
            ref.set(1, 2, 3);
            return { laneIndex: 1, laneLabel: 2.5, laneCount: 4 };
        });

        expect(controller.getReference()).toEqual({
            laneIndex: 1,
            laneLabel: 2.5,
            position: { x: 1, y: 2, z: 3 }
        });
    });
});
