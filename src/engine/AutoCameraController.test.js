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
    controller._autoCameraViewFromCameraOffset = new THREE.Vector3();
    controller._autoCameraViewFromTargetOffset = new THREE.Vector3();
    controller._autoCameraViewToCameraOffset = new THREE.Vector3();
    controller._autoCameraViewToTargetOffset = new THREE.Vector3();
    controller._autoCameraDefaultCameraOffset = new THREE.Vector3(100, 0, 0);
    controller._autoCameraDefaultTargetOffset = new THREE.Vector3(0, 100, 0);
    controller._autoCameraLnCameraOffset = new THREE.Vector3(7, 8, 9);
    controller._autoCameraLnTargetOffset = new THREE.Vector3(10, 11, 12);

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
