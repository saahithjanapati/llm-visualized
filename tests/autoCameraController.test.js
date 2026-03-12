import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { AutoCameraController } from '../src/engine/AutoCameraController.js';
import { HORIZ_PHASE, LN2_PHASE } from '../src/engine/layers/gpt2LanePhases.js';

function createController({ followEnabled = true } = {}) {
    const group = new THREE.Object3D();
    group.position.set(10, 20, 30);

    const lane = {
        horizPhase: HORIZ_PHASE.WAITING,
        ln2Phase: LN2_PHASE.NOT_STARTED,
        posAddComplete: true,
        originalVec: {
            group
        }
    };

    const layer = {
        index: 0,
        isActive: true,
        lanes: [lane],
        mhsaAnimation: {
            mhaPassThroughPhase: 'parallel_pass_through_active',
            rowMergePhase: 'not_started',
            outputProjMatrixAnimationPhase: 'waiting',
            outputProjMatrixReturnComplete: false
        }
    };

    const pipeline = {
        _layers: [layer],
        _currentLayerIdx: 0,
        addEventListener() {},
        removeEventListener() {},
        isForwardPassComplete() {
            return false;
        }
    };

    const controls = {
        target: new THREE.Vector3(),
        addEventListener() {},
        removeEventListener() {},
        update() {}
    };

    const camera = new THREE.PerspectiveCamera();
    camera.position.set(0, 0, 100);

    const engine = {
        camera,
        controls,
        notifyCameraUpdated() {}
    };

    const controller = new AutoCameraController({
        pipeline,
        engine,
        opts: {
            autoCameraFollow: followEnabled,
            autoCameraViewSwitchHoldMs: 0,
            autoCameraSmoothAlpha: 1,
            autoCameraOffsetLerpAlpha: 1,
            autoCameraViewBlendAlpha: 1,
            autoCameraViewBlendAlphaTransition: 1,
            autoCameraDefaultCameraOffset: new THREE.Vector3(0, 0, 100),
            autoCameraDefaultTargetOffset: new THREE.Vector3(0, 0, 0),
            autoCameraMhsaCameraOffset: new THREE.Vector3(100, 0, 0),
            autoCameraMhsaTargetOffset: new THREE.Vector3(10, 0, 0),
            autoCameraLnCameraOffset: new THREE.Vector3(0, 100, 0),
            autoCameraLnTargetOffset: new THREE.Vector3(0, 10, 0)
        }
    });

    return { controller, layer, lane };
}

function settleView(controller) {
    controller.update();
    if (Number.isFinite(controller._autoCameraViewPendingSinceMs) && controller._autoCameraViewPendingSinceMs > 0) {
        controller._autoCameraViewPendingSinceMs -= 1000;
    }
    controller.update();
}

describe('AutoCameraController', () => {
    it('keeps the semantic follow view key in sync while follow mode is disabled', () => {
        const { controller, layer, lane } = createController({ followEnabled: true });

        settleView(controller);
        expect(controller.getViewKey()).toBe('mhsa');

        controller.setEnabled(false);

        lane.horizPhase = HORIZ_PHASE.WAITING_FOR_LN2;
        lane.ln2Phase = LN2_PHASE.RIGHT;
        layer.mhsaAnimation.mhaPassThroughPhase = 'mha_pass_through_complete';
        layer.mhsaAnimation.rowMergePhase = 'merged';
        layer.mhsaAnimation.outputProjMatrixAnimationPhase = 'vectors_inside';

        settleView(controller);
        expect(controller.getViewKey()).toBe('ln');

        controller.setEnabled(true, { resetView: true, smoothReset: true });
        expect(controller.getViewKey()).toBe('ln');
    });
});
