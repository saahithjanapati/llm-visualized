import * as THREE from 'three';
import { HIDE_INSTANCE_Y_OFFSET } from '../utils/constants.js';
import {
    getAutoCameraViewSwitchHoldMs,
    resolveAutoCameraViewState,
    resolveStableAutoCameraViewKey
} from './autoCameraViewLogic.js';

const TMP_CENTER_A = new THREE.Vector3();
const TMP_CENTER_B = new THREE.Vector3();
const TMP_CENTER_MAT_A = new THREE.Matrix4();
const TMP_CENTER_MAT_B = new THREE.Matrix4();
const STOP_RISE_RELEASE_DURATION_MS = 420;
const STOP_RISE_RELEASE_FALLBACK_STEP = 0.05;
const AUTO_CAMERA_VIEW_SWITCH_HOLD_MS_DEFAULT = 90;
const AUTO_CAMERA_FRAME_DELTA_SEC_DEFAULT = 1 / 60;
const AUTO_CAMERA_FRAME_DELTA_SEC_MIN = 1 / 180;
const AUTO_CAMERA_FRAME_DELTA_SEC_MAX = 0.09;
const AUTO_CAMERA_ALPHA_BASE_HZ = 60;
const AUTO_CAMERA_REF_MOTION_RAMP_UP_SEC_DEFAULT = 0.24;
const AUTO_CAMERA_REF_MOTION_RAMP_DOWN_SEC_DEFAULT = 0.16;
const AUTO_CAMERA_REF_MOTION_START_SPEED_DEFAULT = 26;
const AUTO_CAMERA_REF_MOTION_STOP_SPEED_DEFAULT = 10;
const AUTO_CAMERA_REF_MOTION_MIN_SCALE_DEFAULT = 0.36;
const AUTO_CAMERA_CONTROLS_CAPTURE_EPSILON_SQ = 1;
const STARTUP_CAMERA_INTRO_HOLD_MS_DEFAULT = 1000;
const STARTUP_CAMERA_INTRO_TRANSITION_MS_DEFAULT = 1400;

const coerceVector3 = (value, fallback = null) => {
    if (value instanceof THREE.Vector3) return value.clone();
    if (value && typeof value === 'object' && 'x' in value && 'y' in value && 'z' in value) {
        return new THREE.Vector3(value.x, value.y, value.z);
    }
    return fallback ? fallback.clone() : null;
};

const isFiniteVector3 = (value) => !!(
    value
    && Number.isFinite(value.x)
    && Number.isFinite(value.y)
    && Number.isFinite(value.z)
);

export class AutoCameraController {
    constructor({ pipeline, engine, opts = {} }) {
        this._pipeline = pipeline;
        this._engine = engine;
        this._initState(opts);
        this._initOffsets(opts);
        this._progressHandler = () => { this.maybeFocus(); };
        this._pipeline?.addEventListener?.('progress', this._progressHandler);
    }

    dispose() {
        if (this._progressHandler) {
            this._pipeline?.removeEventListener?.('progress', this._progressHandler);
            this._progressHandler = null;
        }
        if (this._engine?.controls && this._controlsChangeHandler) {
            this._engine.controls.removeEventListener('change', this._controlsChangeHandler);
        }
        this._controlsChangeHandler = null;
        this._completeStartupCameraIntro(false);
        if (this._cameraOffsetDiv) {
            this._cameraOffsetDiv.style.display = 'none';
        }
        if (this._panelShiftTween && typeof this._panelShiftTween.stop === 'function') {
            try { this._panelShiftTween.stop(); } catch (_) { /* no-op */ }
        }
        this._panelShiftTween = null;
    }

    update() {
        const hasPanelShift = this._hasPanelShift();
        if (this._startupCameraIntroActive) {
            const introHandled = this._updateStartupCameraIntro();
            if (hasPanelShift) {
                this._applyPanelShift();
            }
            this._updateCameraOffsetOverlay();
            if (introHandled) return;
        }
        if (!this._autoCameraFollow && !this._devMode && !hasPanelShift) return;
        if (this._autoCameraFollow) {
            this._updateAutoCameraFollow();
        }
        if (hasPanelShift) {
            this._applyPanelShift();
        }
        this._updateCameraOffsetOverlay();
    }

    setEnabled(enabled, { immediate = false, resetView = false, smoothReset = false } = {}) {
        const nextValue = !!enabled;
        if (nextValue === this._autoCameraFollow) {
            if (nextValue) {
                if (resetView) {
                    this._applyFollowReset({ smoothReset });
                }
                if (immediate) {
                    this._updateAutoCameraFollow();
                    this._updateCameraOffsetOverlay();
                }
            } else {
                this._updateCameraOffsetOverlay();
            }
            return;
        }
        this._autoCameraFollow = nextValue;
        if (this._autoCameraFollow) {
            this._autoCameraSmoothValid = false;
            this._resetAutoCameraReferenceMotion();
            this._autoCameraViewPendingKey = this._autoCameraViewKey;
            this._autoCameraViewPendingSinceMs = 0;
            this._autoCameraPostAddLockActive = false;
            this._autoCameraPostAddLockUntilMs = 0;
            if (resetView) {
                this._applyFollowReset({ smoothReset });
            }
        }
        if (this._autoCameraFollow) {
            if (immediate) {
                this._updateAutoCameraFollow();
            }
        } else {
            this._completeStartupCameraIntro(false);
            this._autoCameraPostAddLockActive = false;
            this._autoCameraPostAddLockUntilMs = 0;
            this._autoCameraForceEmbedVocabStartLock = false;
            this._clearAutoCameraOffsets();
            this._resetAutoCameraReferenceMotion();
        }
        this._updateCameraOffsetOverlay();
    }

    isEnabled() {
        return !!this._autoCameraFollow;
    }

    getViewKey() {
        return this._autoCameraViewKey || 'default';
    }

    _applyFollowReset({ smoothReset = false } = {}) {
        this._updateAutoCameraScaledOffsets();
        this._autoCameraSmoothValid = false;
        this._resetAutoCameraReferenceMotion();
        const useEmbedVocabStart = this._autoCameraEmbedVocabOffsetsEnabled;
        const fallbackViewKey = useEmbedVocabStart ? 'embed-vocab' : 'default';

        if (smoothReset) {
            const reference = this._autoCameraCenter;
            const laneInfo = this._resolveActiveLanePosition(reference);
            const hasReference = laneInfo.laneIndex >= 0
                && Number.isFinite(reference.x)
                && Number.isFinite(reference.y)
                && Number.isFinite(reference.z);
            const capturedFromCurrentView = hasReference
                ? this._captureAutoCameraOffsets(reference)
                : false;

            // Start follow-mode from whichever semantic view should be active
            // right now, then blend from the live camera pose to that target.
            this._autoCameraForceEmbedVocabStartLock = false;
            const resetViewKey = this._resolveAutoCameraViewKey() || fallbackViewKey;
            const {
                cameraOffset: resetCameraOffset,
                targetOffset: resetTargetOffset
            } = this._resolveAutoCameraOffsetsForViewKey(resetViewKey, {});

            this._autoCameraViewKey = resetViewKey;
            this._autoCameraViewPendingKey = resetViewKey;
            this._autoCameraViewPendingSinceMs = 0;
            this._autoCameraViewBlendAlphaActive = this._autoCameraViewBlendAlphaTransition;

            if (capturedFromCurrentView && this._hasAutoCameraOffsets) {
                this._autoCameraViewBlendT = 0;
                this._autoCameraViewFromCameraOffset.copy(this._autoCameraCurrentCameraOffset);
                this._autoCameraViewFromTargetOffset.copy(this._autoCameraCurrentTargetOffset);
                this._autoCameraViewToCameraOffset.copy(resetCameraOffset);
                this._autoCameraViewToTargetOffset.copy(resetTargetOffset);
            } else {
                this._autoCameraViewBlendT = 1;
                this._autoCameraViewFromCameraOffset.copy(resetCameraOffset);
                this._autoCameraViewFromTargetOffset.copy(resetTargetOffset);
                this._autoCameraViewToCameraOffset.copy(resetCameraOffset);
                this._autoCameraViewToTargetOffset.copy(resetTargetOffset);
                this._setAutoCameraOffsets(
                    resetCameraOffset,
                    resetTargetOffset,
                    { snap: true }
                );
            }
            return;
        }

        const resetViewKey = fallbackViewKey;
        const {
            cameraOffset: resetCameraOffset,
            targetOffset: resetTargetOffset
        } = this._resolveAutoCameraOffsetsForViewKey(resetViewKey, {});
        this._autoCameraViewKey = resetViewKey;
        this._autoCameraViewPendingKey = resetViewKey;
        this._autoCameraViewPendingSinceMs = 0;
        this._autoCameraViewBlendT = 1;
        this._autoCameraViewFromCameraOffset.copy(resetCameraOffset);
        this._autoCameraViewFromTargetOffset.copy(resetTargetOffset);
        this._autoCameraViewToCameraOffset.copy(resetCameraOffset);
        this._autoCameraViewToTargetOffset.copy(resetTargetOffset);
        this._autoCameraForceEmbedVocabStartLock = useEmbedVocabStart;
        this._setAutoCameraOffsets(
            resetCameraOffset,
            resetTargetOffset,
            { snap: true }
        );
    }

    setScreenShiftPixels(pixels, { immediate = false, durationMs = 520 } = {}) {
        const next = Number.isFinite(pixels) ? pixels : 0;
        const delta = Math.abs(next - this._panelShiftPxTarget);
        const hasTween = this._panelShiftTween && typeof this._panelShiftTween.stop === 'function';

        if (delta < 0.5 && !immediate && !hasTween) {
            this._panelShiftPxTarget = next;
            return;
        }

        if (hasTween) {
            try { this._panelShiftTween.stop(); } catch (_) { /* no-op */ }
            this._panelShiftTween = null;
        }

        this._panelShiftPxTarget = next;

        const tweenAvailable = typeof TWEEN !== 'undefined' && TWEEN?.Tween;
        if (immediate || !tweenAvailable) {
            this._panelShiftPxCurrent = next;
            return;
        }

        const duration = Math.max(200, Number.isFinite(durationMs) ? durationMs : 520);
        const easing = TWEEN?.Easing?.Cubic?.InOut || TWEEN?.Easing?.Quadratic?.InOut;
        const state = { value: this._panelShiftPxCurrent };
        this._panelShiftTween = new TWEEN.Tween(state)
            .to({ value: next }, duration)
            .easing(typeof easing === 'function' ? easing : TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                this._panelShiftPxCurrent = state.value;
            })
            .onComplete(() => {
                this._panelShiftPxCurrent = next;
                this._panelShiftTween = null;
            })
            .start();
    }

    focusOverview({ immediate = true, durationMs = 1400 } = {}) {
        const engine = this._engine;
        const camera = engine?.camera;
        const controls = engine?.controls;
        if (!camera || !controls || !controls.target) return;
        if (!this._overviewCameraPosition || !this._overviewCameraTarget) return;
        if (immediate || typeof TWEEN === 'undefined' || !TWEEN?.Tween) {
            camera.position.copy(this._overviewCameraPosition);
            controls.target.copy(this._overviewCameraTarget);
            if (typeof controls.update === 'function') controls.update();
            engine?.notifyCameraUpdated?.();
            return;
        }

        const duration = Math.max(200, Number.isFinite(durationMs) ? durationMs : 1400);
        const easing = TWEEN?.Easing?.Quadratic?.InOut;

        if (this._overviewCameraTween && typeof this._overviewCameraTween.stop === 'function') {
            this._overviewCameraTween.stop();
        }
        if (this._overviewTargetTween && typeof this._overviewTargetTween.stop === 'function') {
            this._overviewTargetTween.stop();
        }

        this._overviewCameraTween = new TWEEN.Tween(camera.position)
            .to({
                x: this._overviewCameraPosition.x,
                y: this._overviewCameraPosition.y,
                z: this._overviewCameraPosition.z
            }, duration)
            .easing(typeof easing === 'function' ? easing : TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => engine?.notifyCameraUpdated?.())
            .start();

        this._overviewTargetTween = new TWEEN.Tween(controls.target)
            .to({
                x: this._overviewCameraTarget.x,
                y: this._overviewCameraTarget.y,
                z: this._overviewCameraTarget.z
            }, duration)
            .easing(typeof easing === 'function' ? easing : TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                if (typeof controls.update === 'function') controls.update();
                engine?.notifyCameraUpdated?.();
            })
            .start();
    }

    playStartupCameraIntro({
        holdMs = STARTUP_CAMERA_INTRO_HOLD_MS_DEFAULT,
        transitionMs = STARTUP_CAMERA_INTRO_TRANSITION_MS_DEFAULT
    } = {}) {
        if (this._startupCameraIntroActive && this._startupCameraIntroPromise) {
            return this._startupCameraIntroPromise;
        }
        if (this._startupCameraIntroPlayed || !this._autoCameraFollow) {
            return Promise.resolve(false);
        }

        const engine = this._engine;
        const camera = engine?.camera;
        const controls = engine?.controls;
        if (!camera || !controls || !controls.target) {
            return Promise.resolve(false);
        }
        if (!isFiniteVector3(this._overviewCameraPosition) || !isFiniteVector3(this._overviewCameraTarget)) {
            return Promise.resolve(false);
        }

        const reference = this._startupCameraIntroTargetReference;
        const { laneIndex } = this._resolveActiveLanePosition(reference);
        if (laneIndex < 0 || !isFiniteVector3(reference)) {
            return Promise.resolve(false);
        }

        const targetViewKey = this._autoCameraEmbedVocabOffsetsEnabled
            ? 'embed-vocab'
            : (this._resolveAutoCameraViewKey() || 'default');
        const resolvedOffsets = this._resolveAutoCameraOffsetsForViewKey(targetViewKey, {});

        this._startupCameraIntroPlayed = true;
        this._startupCameraIntroActive = true;
        this._startupCameraIntroStage = 'hold';
        this._startupCameraIntroStageStartedAtMs = this._getNowMs();
        this._startupCameraIntroHoldMs = Math.max(
            0,
            Number.isFinite(holdMs) ? holdMs : STARTUP_CAMERA_INTRO_HOLD_MS_DEFAULT
        );
        this._startupCameraIntroTransitionMs = Math.max(
            0,
            Number.isFinite(transitionMs) ? transitionMs : STARTUP_CAMERA_INTRO_TRANSITION_MS_DEFAULT
        );
        this._startupCameraIntroTargetViewKey = targetViewKey;
        this._startupCameraIntroFromCamera.copy(this._overviewCameraPosition);
        this._startupCameraIntroFromTarget.copy(this._overviewCameraTarget);
        this._startupCameraIntroToCamera.copy(reference).add(resolvedOffsets.cameraOffset);
        this._startupCameraIntroToTarget.copy(reference).add(resolvedOffsets.targetOffset);
        this._applyAbsoluteCameraPose(this._overviewCameraPosition, this._overviewCameraTarget);

        this._startupCameraIntroPromise = new Promise((resolve) => {
            this._startupCameraIntroResolve = resolve;
        });

        return this._startupCameraIntroPromise;
    }

    getReference() {
        const ref = this._autoCameraInspectorRef;
        const info = this._resolveActiveLanePosition(ref);
        if (!info || info.laneIndex < 0) return null;
        return {
            laneIndex: info.laneIndex,
            position: { x: ref.x, y: ref.y, z: ref.z }
        };
    }

    maybeFocus({ immediate = false } = {}) {
        if (this._startupCameraIntroActive) {
            if (immediate) {
                const nowMs = this._updateAutoCameraFrameTiming();
                const introHandled = this._updateStartupCameraIntro(nowMs);
                if (!introHandled && this._autoCameraFollow) {
                    this._updateAutoCameraFollow();
                }
            }
            this._updateCameraOffsetOverlay();
            return;
        }
        if (!this._autoCameraFollow && !immediate) {
            this._updateCameraOffsetOverlay();
            return;
        }
        this._updateAutoCameraFollow();
        this._updateCameraOffsetOverlay();
    }

    setDevMode(enabled) {
        this._devMode = !!enabled;
        if (!this._devMode && this._cameraOffsetDiv) {
            this._cameraOffsetDiv.style.display = 'none';
        } else {
            this._updateCameraOffsetOverlay();
        }
    }

    _initState(opts) {
        this._autoCameraFollow = opts.autoCameraFollow !== false;
        const smoothAlpha = typeof opts.autoCameraSmoothAlpha === 'number' ? opts.autoCameraSmoothAlpha : 0.14;
        this._autoCameraSmoothAlpha = Math.min(1, Math.max(0, smoothAlpha));
        const offsetAlpha = typeof opts.autoCameraOffsetLerpAlpha === 'number' ? opts.autoCameraOffsetLerpAlpha : 0.14;
        this._autoCameraOffsetLerpAlpha = Math.min(1, Math.max(0, offsetAlpha));
        const viewBlendAlpha = typeof opts.autoCameraViewBlendAlpha === 'number' ? opts.autoCameraViewBlendAlpha : 0.12;
        this._autoCameraViewBlendAlpha = Math.min(1, Math.max(0, viewBlendAlpha));
        const viewSwitchHoldMs = Number.isFinite(opts.autoCameraViewSwitchHoldMs)
            ? opts.autoCameraViewSwitchHoldMs
            : AUTO_CAMERA_VIEW_SWITCH_HOLD_MS_DEFAULT;
        this._autoCameraViewSwitchHoldMs = Math.max(0, Math.min(600, viewSwitchHoldMs));
        const mlpBlendAlpha = typeof opts.autoCameraViewBlendAlphaMlpReturn === 'number'
            ? opts.autoCameraViewBlendAlphaMlpReturn
            : (this._autoCameraViewBlendAlpha * 0.35);
        this._autoCameraViewBlendAlphaMlpReturn = Math.min(1, Math.max(0, mlpBlendAlpha));
        const transitionBlendAlpha = typeof opts.autoCameraViewBlendAlphaTransition === 'number'
            ? opts.autoCameraViewBlendAlphaTransition
            : this._autoCameraViewBlendAlphaMlpReturn;
        this._autoCameraViewBlendAlphaTransition = Math.min(1, Math.max(0, transitionBlendAlpha));
        this._autoCameraViewBlendAlphaActive = this._autoCameraViewBlendAlpha;
        const mobileScale = typeof opts.autoCameraMobileScale === 'number' ? opts.autoCameraMobileScale : 1.0;
        this._autoCameraScaleMax = Math.max(1.0, mobileScale);
        const mobileShiftX = typeof opts.autoCameraMobileShiftX === 'number' ? opts.autoCameraMobileShiftX : 0;
        this._autoCameraShiftMaxX = mobileShiftX;
        const travelShiftX = typeof opts.autoCameraTravelMobileShiftX === 'number' ? opts.autoCameraTravelMobileShiftX : 0;
        this._autoCameraShiftMaxTravelX = travelShiftX;
        const mhsaShiftX = typeof opts.autoCameraMhsaMobileShiftX === 'number' ? opts.autoCameraMhsaMobileShiftX : 0;
        this._autoCameraShiftMaxMhsaX = mhsaShiftX;
        this._autoCameraMhsaMobileOverrideCameraOffset = null;
        this._autoCameraMhsaMobileOverrideTargetOffset = null;
        this._autoCameraMhsaMobileOverrideEnabled = false;
        this._autoCameraConcatMobileOverrideCameraOffset = null;
        this._autoCameraConcatMobileOverrideTargetOffset = null;
        this._autoCameraConcatMobileOverrideEnabled = false;
        this._autoCameraTravelMobileOverrideCameraOffset = null;
        this._autoCameraTravelMobileOverrideTargetOffset = null;
        this._autoCameraTravelMobileOverrideEnabled = false;
        this._autoCameraFinalMobileOverrideCameraOffset = null;
        this._autoCameraFinalMobileOverrideTargetOffset = null;
        this._autoCameraFinalMobileOverrideEnabled = false;
        this._autoCameraScaleMinWidth = Number.isFinite(opts.autoCameraScaleMinWidth)
            ? Math.max(200, opts.autoCameraScaleMinWidth)
            : 360;
        this._autoCameraScaleMaxWidth = Number.isFinite(opts.autoCameraScaleMaxWidth)
            ? Math.max(this._autoCameraScaleMinWidth + 10, opts.autoCameraScaleMaxWidth)
            : 880;
        this._autoCameraScaleLast = 1.0;
        this._autoCameraMobileFactorLast = 0;
        this._autoCameraShiftLastX = 0;
        this._autoCameraCenter = new THREE.Vector3();
        this._autoCameraOffsetScratch = new THREE.Vector3();
        this._autoCameraDesiredCameraOffset = new THREE.Vector3();
        this._autoCameraDesiredTargetOffset = new THREE.Vector3();
        this._autoCameraCurrentCameraOffset = new THREE.Vector3();
        this._autoCameraCurrentTargetOffset = new THREE.Vector3();
        this._autoCameraLastAppliedCameraPosition = new THREE.Vector3();
        this._autoCameraLastAppliedTarget = new THREE.Vector3();
        this._autoCameraSmoothedRef = new THREE.Vector3();
        this._autoCameraPrevReference = new THREE.Vector3();
        this._autoCameraPrevReferenceValid = false;
        this._autoCameraReferenceMotionRampT = 0;
        this._autoCameraReferenceMotionRampUpSec = Number.isFinite(opts.autoCameraReferenceMotionRampUpSec)
            ? Math.max(0.02, opts.autoCameraReferenceMotionRampUpSec)
            : AUTO_CAMERA_REF_MOTION_RAMP_UP_SEC_DEFAULT;
        this._autoCameraReferenceMotionRampDownSec = Number.isFinite(opts.autoCameraReferenceMotionRampDownSec)
            ? Math.max(0.02, opts.autoCameraReferenceMotionRampDownSec)
            : AUTO_CAMERA_REF_MOTION_RAMP_DOWN_SEC_DEFAULT;
        this._autoCameraReferenceMotionStartSpeed = Number.isFinite(opts.autoCameraReferenceMotionStartSpeed)
            ? Math.max(0, opts.autoCameraReferenceMotionStartSpeed)
            : AUTO_CAMERA_REF_MOTION_START_SPEED_DEFAULT;
        this._autoCameraReferenceMotionStopSpeed = Number.isFinite(opts.autoCameraReferenceMotionStopSpeed)
            ? Math.max(0, opts.autoCameraReferenceMotionStopSpeed)
            : AUTO_CAMERA_REF_MOTION_STOP_SPEED_DEFAULT;
        this._autoCameraReferenceMotionMinScale = Number.isFinite(opts.autoCameraReferenceMotionMinScale)
            ? Math.min(1, Math.max(0, opts.autoCameraReferenceMotionMinScale))
            : AUTO_CAMERA_REF_MOTION_MIN_SCALE_DEFAULT;
        this._autoCameraSmoothValid = false;
        this._autoCameraPostAddLockRef = new THREE.Vector3();
        this._autoCameraPostAddLockUntilMs = 0;
        this._autoCameraPostAddLockActive = false;
        this._autoCameraViewKey = 'default';
        this._autoCameraViewPendingKey = 'default';
        this._autoCameraViewPendingSinceMs = 0;
        this._autoCameraViewBlendT = 1;
        this._autoCameraViewFromCameraOffset = new THREE.Vector3();
        this._autoCameraViewFromTargetOffset = new THREE.Vector3();
        this._autoCameraViewToCameraOffset = new THREE.Vector3();
        this._autoCameraViewToTargetOffset = new THREE.Vector3();
        this._autoCameraViewBlendCameraOffset = new THREE.Vector3();
        this._autoCameraViewBlendTargetOffset = new THREE.Vector3();
        this._autoCameraViewContext = null;
        this._autoCameraForceEmbedVocabStartLock = false;
        this._autoCameraDefaultCameraOffset = new THREE.Vector3();
        this._autoCameraDefaultTargetOffset = new THREE.Vector3();
        this._autoCameraEmbedVocabCameraOffset = new THREE.Vector3();
        this._autoCameraEmbedVocabTargetOffset = new THREE.Vector3();
        this._autoCameraEmbedPositionCameraOffset = new THREE.Vector3();
        this._autoCameraEmbedPositionTargetOffset = new THREE.Vector3();
        this._autoCameraEmbedAddCameraOffset = new THREE.Vector3();
        this._autoCameraEmbedAddTargetOffset = new THREE.Vector3();
        this._autoCameraMhsaCameraOffset = new THREE.Vector3();
        this._autoCameraMhsaTargetOffset = new THREE.Vector3();
        this._autoCameraConcatCameraOffset = new THREE.Vector3();
        this._autoCameraConcatTargetOffset = new THREE.Vector3();
        this._autoCameraResolvedViewOffsets = {
            cameraOffset: this._autoCameraDefaultCameraOffset,
            targetOffset: this._autoCameraDefaultTargetOffset
        };
        this._autoCameraEmbedVocabOffsetsEnabled = false;
        this._autoCameraEmbedPositionOffsetsEnabled = false;
        this._autoCameraEmbedAddOffsetsEnabled = false;
        this._autoCameraMhsaOffsetsEnabled = false;
        this._autoCameraConcatOffsetsEnabled = false;
        this._autoCameraLnCameraOffset = new THREE.Vector3();
        this._autoCameraLnTargetOffset = new THREE.Vector3();
        this._autoCameraTopLnCameraOffset = new THREE.Vector3();
        this._autoCameraTopLnTargetOffset = new THREE.Vector3();
        this._autoCameraTravelCameraOffset = new THREE.Vector3();
        this._autoCameraTravelTargetOffset = new THREE.Vector3();
        this._autoCameraFinalCameraOffset = new THREE.Vector3();
        this._autoCameraFinalTargetOffset = new THREE.Vector3();
        this._autoCameraLayerEndDesktopCameraOffset = new THREE.Vector3();
        this._autoCameraLayerEndDesktopTargetOffset = new THREE.Vector3();
        this._autoCameraLnOffsetsEnabled = false;
        this._autoCameraTopLnOffsetsEnabled = false;
        this._autoCameraTravelOffsetsEnabled = false;
        this._autoCameraFinalOffsetsEnabled = false;
        this._autoCameraLayerEndDesktopOffsetsEnabled = false;
        this._autoCameraDefaultCameraOffsetBase = new THREE.Vector3();
        this._autoCameraDefaultTargetOffsetBase = new THREE.Vector3();
        this._autoCameraEmbedVocabCameraOffsetBase = new THREE.Vector3();
        this._autoCameraEmbedVocabTargetOffsetBase = new THREE.Vector3();
        this._autoCameraEmbedPositionCameraOffsetBase = new THREE.Vector3();
        this._autoCameraEmbedPositionTargetOffsetBase = new THREE.Vector3();
        this._autoCameraEmbedAddCameraOffsetBase = new THREE.Vector3();
        this._autoCameraEmbedAddTargetOffsetBase = new THREE.Vector3();
        this._autoCameraMhsaCameraOffsetBase = new THREE.Vector3();
        this._autoCameraMhsaTargetOffsetBase = new THREE.Vector3();
        this._autoCameraConcatCameraOffsetBase = new THREE.Vector3();
        this._autoCameraConcatTargetOffsetBase = new THREE.Vector3();
        this._autoCameraLnCameraOffsetBase = new THREE.Vector3();
        this._autoCameraLnTargetOffsetBase = new THREE.Vector3();
        this._autoCameraTopLnCameraOffsetBase = new THREE.Vector3();
        this._autoCameraTopLnTargetOffsetBase = new THREE.Vector3();
        this._autoCameraTravelCameraOffsetBase = new THREE.Vector3();
        this._autoCameraTravelTargetOffsetBase = new THREE.Vector3();
        this._autoCameraFinalCameraOffsetBase = new THREE.Vector3();
        this._autoCameraFinalTargetOffsetBase = new THREE.Vector3();
        this._autoCameraLayerEndDesktopCameraOffsetBase = new THREE.Vector3();
        this._autoCameraLayerEndDesktopTargetOffsetBase = new THREE.Vector3();
        this._autoCameraInspectorRef = new THREE.Vector3();
        this._hasAutoCameraOffsets = false;
        this._hasAutoCameraAppliedPose = false;
        this._suppressControlsChange = false;
        this._devMode = !!opts.devMode;
        this._cameraOffsetDiv = (typeof document !== 'undefined')
            ? document.getElementById('cameraOffsetOverlay')
            : null;
        this._controlsChangeHandler = null;
        this._overviewCameraTween = null;
        this._overviewTargetTween = null;
        this._panelShiftPxTarget = 0;
        this._panelShiftPxCurrent = 0;
        this._panelShiftTween = null;
        this._panelShiftPxApplied = 0;
        this._panelShiftViewWidth = 0;
        this._panelShiftViewHeight = 0;
        this._panelShiftViewActive = false;
        this._autoCameraFrameDeltaSec = AUTO_CAMERA_FRAME_DELTA_SEC_DEFAULT;
        this._autoCameraLastUpdateMs = 0;
        this._startupCameraIntroActive = false;
        this._startupCameraIntroPlayed = false;
        this._startupCameraIntroStage = 'idle';
        this._startupCameraIntroStageStartedAtMs = 0;
        this._startupCameraIntroHoldMs = STARTUP_CAMERA_INTRO_HOLD_MS_DEFAULT;
        this._startupCameraIntroTransitionMs = STARTUP_CAMERA_INTRO_TRANSITION_MS_DEFAULT;
        this._startupCameraIntroTargetViewKey = 'embed-vocab';
        this._startupCameraIntroFromCamera = new THREE.Vector3();
        this._startupCameraIntroFromTarget = new THREE.Vector3();
        this._startupCameraIntroToCamera = new THREE.Vector3();
        this._startupCameraIntroToTarget = new THREE.Vector3();
        this._startupCameraIntroCurrentCamera = new THREE.Vector3();
        this._startupCameraIntroCurrentTarget = new THREE.Vector3();
        this._startupCameraIntroTargetReference = new THREE.Vector3();
        this._startupCameraIntroPromise = null;
        this._startupCameraIntroResolve = null;
    }

    _initOffsets(opts) {
        this._overviewCameraPosition = coerceVector3(
            (opts.skipToEndCameraPosition ?? opts.cameraPosition) ?? null,
            null
        );
        this._overviewCameraTarget = coerceVector3(
            (opts.skipToEndCameraTarget ?? opts.cameraTarget) ?? null,
            null
        );

        const defaultTargetOffset = coerceVector3(opts.autoCameraDefaultTargetOffset, new THREE.Vector3(0, 0, 0));
        const cameraTarget = this._engine?.controls?.target || new THREE.Vector3();
        const defaultCameraOffset = coerceVector3(
            opts.autoCameraDefaultCameraOffset,
            this._engine?.camera?.position ? this._engine.camera.position.clone().sub(cameraTarget) : new THREE.Vector3(0, 2000, 16000)
        );
        if (defaultTargetOffset) this._autoCameraDefaultTargetOffsetBase.copy(defaultTargetOffset);
        if (defaultCameraOffset) this._autoCameraDefaultCameraOffsetBase.copy(defaultCameraOffset);

        const embedVocabCamOffset = coerceVector3(opts.autoCameraEmbedVocabCameraOffset, null);
        const embedVocabTargetOffset = coerceVector3(opts.autoCameraEmbedVocabTargetOffset, null);
        if (embedVocabCamOffset && embedVocabTargetOffset) {
            this._autoCameraEmbedVocabCameraOffsetBase.copy(embedVocabCamOffset);
            this._autoCameraEmbedVocabTargetOffsetBase.copy(embedVocabTargetOffset);
            this._autoCameraEmbedVocabOffsetsEnabled = true;
        }

        const embedPositionCamOffset = coerceVector3(opts.autoCameraEmbedPositionCameraOffset, null);
        const embedPositionTargetOffset = coerceVector3(opts.autoCameraEmbedPositionTargetOffset, null);
        if (embedPositionCamOffset && embedPositionTargetOffset) {
            this._autoCameraEmbedPositionCameraOffsetBase.copy(embedPositionCamOffset);
            this._autoCameraEmbedPositionTargetOffsetBase.copy(embedPositionTargetOffset);
            this._autoCameraEmbedPositionOffsetsEnabled = true;
        }

        const embedAddCamOffset = coerceVector3(opts.autoCameraEmbedAddCameraOffset, null);
        const embedAddTargetOffset = coerceVector3(opts.autoCameraEmbedAddTargetOffset, null);
        if (embedAddCamOffset && embedAddTargetOffset) {
            this._autoCameraEmbedAddCameraOffsetBase.copy(embedAddCamOffset);
            this._autoCameraEmbedAddTargetOffsetBase.copy(embedAddTargetOffset);
            this._autoCameraEmbedAddOffsetsEnabled = true;
        }

        const mhsaCamOffset = coerceVector3(opts.autoCameraMhsaCameraOffset, null);
        const mhsaTargetOffset = coerceVector3(opts.autoCameraMhsaTargetOffset, null);
        if (mhsaCamOffset && mhsaTargetOffset) {
            this._autoCameraMhsaCameraOffsetBase.copy(mhsaCamOffset);
            this._autoCameraMhsaTargetOffsetBase.copy(mhsaTargetOffset);
            this._autoCameraMhsaOffsetsEnabled = true;
        }
        const mhsaMobileCamOffset = coerceVector3(opts.autoCameraMhsaMobileCameraOffset, null);
        const mhsaMobileTargetOffset = coerceVector3(opts.autoCameraMhsaMobileTargetOffset, null);
        if (mhsaMobileCamOffset && mhsaMobileTargetOffset) {
            this._autoCameraMhsaMobileOverrideCameraOffset = mhsaMobileCamOffset;
            this._autoCameraMhsaMobileOverrideTargetOffset = mhsaMobileTargetOffset;
            this._autoCameraMhsaMobileOverrideEnabled = true;
        }

        const concatCamOffset = coerceVector3(opts.autoCameraConcatCameraOffset, null);
        const concatTargetOffset = coerceVector3(opts.autoCameraConcatTargetOffset, null);
        if (concatCamOffset && concatTargetOffset) {
            this._autoCameraConcatCameraOffsetBase.copy(concatCamOffset);
            this._autoCameraConcatTargetOffsetBase.copy(concatTargetOffset);
            this._autoCameraConcatOffsetsEnabled = true;
        }
        const concatMobileCamOffset = coerceVector3(opts.autoCameraConcatMobileCameraOffset, null);
        const concatMobileTargetOffset = coerceVector3(opts.autoCameraConcatMobileTargetOffset, null);
        if (concatMobileCamOffset && concatMobileTargetOffset) {
            this._autoCameraConcatMobileOverrideCameraOffset = concatMobileCamOffset;
            this._autoCameraConcatMobileOverrideTargetOffset = concatMobileTargetOffset;
            this._autoCameraConcatMobileOverrideEnabled = true;
        }

        const lnCamOffset = coerceVector3(opts.autoCameraLnCameraOffset, null);
        const lnTargetOffset = coerceVector3(opts.autoCameraLnTargetOffset, null);
        if (lnCamOffset && lnTargetOffset) {
            this._autoCameraLnCameraOffsetBase.copy(lnCamOffset);
            this._autoCameraLnTargetOffsetBase.copy(lnTargetOffset);
            this._autoCameraLnOffsetsEnabled = true;
        }

        const topLnCamOffset = coerceVector3(opts.autoCameraTopLnCameraOffset, null);
        const topLnTargetOffset = coerceVector3(opts.autoCameraTopLnTargetOffset, null);
        if (topLnCamOffset && topLnTargetOffset) {
            this._autoCameraTopLnCameraOffsetBase.copy(topLnCamOffset);
            this._autoCameraTopLnTargetOffsetBase.copy(topLnTargetOffset);
            this._autoCameraTopLnOffsetsEnabled = true;
        }

        const travelCamOffset = coerceVector3(opts.autoCameraTravelCameraOffset, null);
        const travelTargetOffset = coerceVector3(opts.autoCameraTravelTargetOffset, null);
        if (travelCamOffset && travelTargetOffset) {
            this._autoCameraTravelCameraOffsetBase.copy(travelCamOffset);
            this._autoCameraTravelTargetOffsetBase.copy(travelTargetOffset);
            this._autoCameraTravelOffsetsEnabled = true;
        }
        const travelMobileCamOffset = coerceVector3(opts.autoCameraTravelMobileCameraOffset, null);
        const travelMobileTargetOffset = coerceVector3(opts.autoCameraTravelMobileTargetOffset, null);
        if (travelMobileCamOffset && travelMobileTargetOffset) {
            this._autoCameraTravelMobileOverrideCameraOffset = travelMobileCamOffset;
            this._autoCameraTravelMobileOverrideTargetOffset = travelMobileTargetOffset;
            this._autoCameraTravelMobileOverrideEnabled = true;
        }

        const finalCamOffset = coerceVector3(opts.autoCameraFinalCameraOffset, null);
        const finalTargetOffset = coerceVector3(opts.autoCameraFinalTargetOffset, null);
        if (finalCamOffset && finalTargetOffset) {
            this._autoCameraFinalCameraOffsetBase.copy(finalCamOffset);
            this._autoCameraFinalTargetOffsetBase.copy(finalTargetOffset);
            this._autoCameraFinalOffsetsEnabled = true;
        }
        const finalMobileCamOffset = coerceVector3(opts.autoCameraFinalMobileCameraOffset, null);
        const finalMobileTargetOffset = coerceVector3(opts.autoCameraFinalMobileTargetOffset, null);
        if (finalMobileCamOffset && finalMobileTargetOffset) {
            this._autoCameraFinalMobileOverrideCameraOffset = finalMobileCamOffset;
            this._autoCameraFinalMobileOverrideTargetOffset = finalMobileTargetOffset;
            this._autoCameraFinalMobileOverrideEnabled = true;
        }

        const layerEndDesktopCamOffset = coerceVector3(opts.autoCameraLayerEndDesktopCameraOffset, null);
        const layerEndDesktopTargetOffset = coerceVector3(opts.autoCameraLayerEndDesktopTargetOffset, null);
        if (layerEndDesktopCamOffset && layerEndDesktopTargetOffset) {
            this._autoCameraLayerEndDesktopCameraOffsetBase.copy(layerEndDesktopCamOffset);
            this._autoCameraLayerEndDesktopTargetOffsetBase.copy(layerEndDesktopTargetOffset);
            this._autoCameraLayerEndDesktopOffsetsEnabled = true;
        }

        this._updateAutoCameraScaledOffsets(true);
        if (this._autoCameraFollow && this._autoCameraEmbedVocabOffsetsEnabled) {
            this._autoCameraViewKey = 'embed-vocab';
            this._autoCameraViewPendingKey = 'embed-vocab';
            this._autoCameraViewPendingSinceMs = 0;
            this._autoCameraViewBlendT = 1;
            this._autoCameraViewFromCameraOffset.copy(this._autoCameraEmbedVocabCameraOffset);
            this._autoCameraViewFromTargetOffset.copy(this._autoCameraEmbedVocabTargetOffset);
            this._autoCameraViewToCameraOffset.copy(this._autoCameraEmbedVocabCameraOffset);
            this._autoCameraViewToTargetOffset.copy(this._autoCameraEmbedVocabTargetOffset);
            this._autoCameraForceEmbedVocabStartLock = true;
            this._setAutoCameraOffsets(
                this._autoCameraEmbedVocabCameraOffset,
                this._autoCameraEmbedVocabTargetOffset,
                { snap: true }
            );
        }

        if (this._engine?.controls) {
            this._controlsChangeHandler = () => {
                if (!this._autoCameraFollow || this._suppressControlsChange) {
                    return;
                }
                if (!this._shouldCaptureOffsetsFromControlsChange()) {
                    this._updateCameraOffsetOverlay();
                    return;
                }
                this._captureAutoCameraOffsets();
                this._updateCameraOffsetOverlay();
            };
            this._engine.controls.addEventListener('change', this._controlsChangeHandler);
        }
    }

    _hasPanelShift() {
        if (this._panelShiftTween) return true;
        if (Math.abs(this._panelShiftPxCurrent) > 0.5) return true;
        return this._panelShiftViewActive;
    }

    _applyAbsoluteCameraPose(cameraPosition, cameraTarget) {
        const engine = this._engine;
        const camera = engine?.camera;
        const controls = engine?.controls;
        if (!camera || !controls || !controls.target) return false;
        if (!isFiniteVector3(cameraPosition) || !isFiniteVector3(cameraTarget)) return false;

        this._suppressControlsChange = true;
        try {
            camera.position.copy(cameraPosition);
            controls.target.copy(cameraTarget);
            if (typeof controls.update === 'function') controls.update();
            engine?.notifyCameraUpdated?.();
            this._recordAutoCameraAppliedPose();
        } finally {
            this._suppressControlsChange = false;
        }
        return true;
    }

    _completeStartupCameraIntro(result = false) {
        const resolve = this._startupCameraIntroResolve;
        this._startupCameraIntroActive = false;
        this._startupCameraIntroStage = 'idle';
        this._startupCameraIntroStageStartedAtMs = 0;
        this._startupCameraIntroPromise = null;
        this._startupCameraIntroResolve = null;
        if (typeof resolve === 'function') {
            resolve(result);
        }
        return result;
    }

    _finishStartupCameraIntro() {
        const targetViewKey = this._startupCameraIntroTargetViewKey || 'default';
        const resolvedOffsets = this._resolveAutoCameraOffsetsForViewKey(targetViewKey, {});

        this._applyAbsoluteCameraPose(this._startupCameraIntroToCamera, this._startupCameraIntroToTarget);
        this._autoCameraViewKey = targetViewKey;
        this._autoCameraViewPendingKey = targetViewKey;
        this._autoCameraViewPendingSinceMs = 0;
        this._autoCameraViewBlendT = 1;
        this._autoCameraViewBlendAlphaActive = this._autoCameraViewBlendAlphaTransition;
        this._autoCameraViewFromCameraOffset.copy(resolvedOffsets.cameraOffset);
        this._autoCameraViewFromTargetOffset.copy(resolvedOffsets.targetOffset);
        this._autoCameraViewToCameraOffset.copy(resolvedOffsets.cameraOffset);
        this._autoCameraViewToTargetOffset.copy(resolvedOffsets.targetOffset);
        this._autoCameraForceEmbedVocabStartLock = targetViewKey === 'embed-vocab';
        if (!this._captureAutoCameraOffsets(this._startupCameraIntroTargetReference)) {
            this._setAutoCameraOffsets(
                resolvedOffsets.cameraOffset,
                resolvedOffsets.targetOffset,
                { snap: true }
            );
        }
        if (isFiniteVector3(this._startupCameraIntroTargetReference)) {
            this._autoCameraSmoothedRef.copy(this._startupCameraIntroTargetReference);
            this._autoCameraSmoothValid = true;
            this._resetAutoCameraReferenceMotion(this._startupCameraIntroTargetReference);
        } else {
            this._autoCameraSmoothValid = false;
            this._resetAutoCameraReferenceMotion();
        }

        return this._completeStartupCameraIntro(true);
    }

    _updateStartupCameraIntro(nowMs = null) {
        if (!this._startupCameraIntroActive) return false;

        const currentNowMs = Number.isFinite(nowMs) ? nowMs : this._updateAutoCameraFrameTiming();
        if (this._startupCameraIntroStage === 'hold') {
            this._applyAbsoluteCameraPose(this._overviewCameraPosition, this._overviewCameraTarget);
            const elapsedMs = Math.max(0, currentNowMs - this._startupCameraIntroStageStartedAtMs);
            if (elapsedMs < this._startupCameraIntroHoldMs) {
                return true;
            }
            if (this._startupCameraIntroTransitionMs <= 0) {
                this._finishStartupCameraIntro();
                return false;
            }
            this._startupCameraIntroStage = 'transition';
            this._startupCameraIntroStageStartedAtMs = currentNowMs;
            return true;
        }

        if (this._startupCameraIntroStage === 'transition') {
            const durationMs = Math.max(1, this._startupCameraIntroTransitionMs);
            const rawT = Math.min(1, Math.max(0, (currentNowMs - this._startupCameraIntroStageStartedAtMs) / durationMs));
            const easedT = rawT * rawT * (3 - 2 * rawT);
            this._startupCameraIntroCurrentCamera
                .copy(this._startupCameraIntroFromCamera)
                .lerp(this._startupCameraIntroToCamera, easedT);
            this._startupCameraIntroCurrentTarget
                .copy(this._startupCameraIntroFromTarget)
                .lerp(this._startupCameraIntroToTarget, easedT);
            this._applyAbsoluteCameraPose(
                this._startupCameraIntroCurrentCamera,
                this._startupCameraIntroCurrentTarget
            );
            if (rawT >= 1) {
                this._finishStartupCameraIntro();
                return false;
            }
            return true;
        }

        return false;
    }

    _applyPanelShift() {
        const engine = this._engine;
        const camera = engine?.camera;
        if (!engine || !camera) return;

        const renderer = engine?.renderer;
        const viewportWidth = renderer?.domElement?.clientWidth
            || (typeof window !== 'undefined' ? window.innerWidth : 0);
        const viewportHeight = renderer?.domElement?.clientHeight
            || (typeof window !== 'undefined' ? window.innerHeight : 0);
        if (!Number.isFinite(viewportWidth) || viewportWidth <= 0) return;
        if (!Number.isFinite(viewportHeight) || viewportHeight <= 0) return;

        const shiftPxRaw = Number.isFinite(this._panelShiftPxCurrent) ? this._panelShiftPxCurrent : 0;
        const shiftPx = Math.abs(shiftPxRaw) < 0.5 ? 0 : shiftPxRaw;

        const dimsChanged = viewportWidth !== this._panelShiftViewWidth
            || viewportHeight !== this._panelShiftViewHeight;
        const shiftChanged = Math.abs(shiftPx - this._panelShiftPxApplied) >= 0.25;
        if (!dimsChanged && !shiftChanged) return;

        this._panelShiftViewWidth = viewportWidth;
        this._panelShiftViewHeight = viewportHeight;
        this._panelShiftPxApplied = shiftPx;

        if (shiftPx === 0) {
            if (camera.view && camera.view.enabled && typeof camera.clearViewOffset === 'function') {
                camera.clearViewOffset();
            }
            this._panelShiftViewActive = false;
        } else if (typeof camera.setViewOffset === 'function') {
            camera.setViewOffset(viewportWidth, viewportHeight, shiftPx, 0, viewportWidth, viewportHeight);
            this._panelShiftViewActive = true;
        }

        engine?.notifyCameraUpdated?.();
    }

    _resolveActiveLanePosition(targetVec = null) {
        const layers = this._pipeline?._layers;
        if (!Array.isArray(layers) || layers.length === 0) {
            return { laneIndex: -1, laneCount: 0 };
        }

        let layerIndex = Math.min(this._pipeline._currentLayerIdx, layers.length - 1);
        let layer = layers[layerIndex];
        const layerIsDormant = !!(layer
            && layer.isActive === false
            && layer._transitionPhase !== 'positioning');
        // During layer handoff there is a brief window where _currentLayerIdx
        // points at the next dormant layer before residual lanes are attached.
        // Keep following the previously active layer to avoid camera snaps.
        if (layerIsDormant && layerIndex > 0) {
            layerIndex -= 1;
            layer = layers[layerIndex];
        }
        const lanes = Array.isArray(layer?.lanes) ? layer.lanes : [];
        const laneCount = lanes.length;
        if (!laneCount) {
            return { laneIndex: -1, laneCount };
        }

        const laneIndex = Math.min(laneCount - 1, Math.floor(laneCount / 2));
        const lane = lanes[laneIndex];
        const vec = lane?.originalVec;
        const vecGroup = vec?.group;
        if (!vecGroup || typeof vecGroup.getWorldPosition !== 'function') {
            return { laneIndex: -1, laneCount };
        }

        if (targetVec) {
            if (lane?.stopRise && lane?.stopRiseTarget) {
                if (!this._resolveStopRiseFollowReference(lane, vec, targetVec)) {
                    vecGroup.getWorldPosition(targetVec);
                }
            } else if (this._resolveStopRiseReleaseReference(lane, vec, targetVec)) {
                // Keep blending to the settled residual vector for a short
                // release window after stopRise clears to avoid a final jerk.
            } else if (!this._getVectorWorldCenter(vec, targetVec)) {
                this._clearStopRiseFollowState(lane);
                vecGroup.getWorldPosition(targetVec);
            } else {
                this._clearStopRiseFollowState(lane);
            }
            this._applyKvDecodeVirtualCenterZ(targetVec, lane, layerIndex, laneCount);
        }

        return { laneIndex, laneCount };
    }

    _shouldUseKvDecodeVirtualCenter(layerIndex, laneCount) {
        if (!Number.isFinite(layerIndex) || layerIndex <= 0) return false;
        if (laneCount !== 1) return false;
        const pipeline = this._pipeline;
        if (!pipeline || !pipeline._kvCacheDecodeActive) return false;
        const layoutCount = Number.isFinite(pipeline._laneLayoutCount)
            ? Math.max(1, Math.floor(pipeline._laneLayoutCount))
            : laneCount;
        return layoutCount > laneCount;
    }

    _applyKvDecodeVirtualCenterZ(targetVec, lane, layerIndex, laneCount) {
        if (!targetVec || !Number.isFinite(targetVec.z)) return;
        if (!this._shouldUseKvDecodeVirtualCenter(layerIndex, laneCount)) return;
        if (!Number.isFinite(lane?.zPos)) return;

        // In decode passes above layer 0 we animate one active lane, but camera
        // offsets should still be applied relative to the virtual middle lane of
        // the full layout. Lane local z=0 is that imaginary middle position.
        targetVec.z -= lane.zPos;
    }

    _clearStopRiseFollowState(lane) {
        if (!lane) return;
        delete lane.__followStopRiseTarget;
        delete lane.__followStopRiseLastY;
        delete lane.__followStopRiseRef;
        delete lane.__followStopRiseReleaseFrom;
        delete lane.__followStopRiseReleaseProgress;
        delete lane.__followStopRiseReleaseStartedAt;
    }

    _resolveStopRiseFollowReference(lane, sourceVec, out) {
        if (!lane || !out || !lane.stopRiseTarget) return false;

        const sourceCenter = TMP_CENTER_A;
        const targetCenter = TMP_CENTER_B;
        const sourceOk = this._getGroupWorldPosition(sourceVec?.group, sourceCenter);
        const targetVecRef = this._findVectorByGroup(lane, lane.stopRiseTarget);
        const targetOk = this._getGroupWorldPosition(
            targetVecRef?.group || lane.stopRiseTarget,
            targetCenter
        );

        if (!sourceOk && !targetOk) return false;
        if (!targetOk) {
            this._clearStopRiseFollowState(lane);
            out.copy(sourceCenter);
            return true;
        }
        if (!sourceOk) {
            this._clearStopRiseFollowState(lane);
            out.copy(targetCenter);
            return true;
        }

        if (lane.__followStopRiseTarget !== lane.stopRiseTarget || !lane.__followStopRiseRef) {
            lane.__followStopRiseTarget = lane.stopRiseTarget;
            lane.__followStopRiseLastY = sourceCenter.y;
            lane.__followStopRiseRef = new THREE.Vector3().copy(sourceCenter);
        }
        delete lane.__followStopRiseReleaseFrom;
        delete lane.__followStopRiseReleaseProgress;
        delete lane.__followStopRiseReleaseStartedAt;

        const rawProgress = Number.isFinite(lane.mhsaResidualAddProgress)
            ? lane.mhsaResidualAddProgress
            : 0;
        const clampedProgress = Math.min(1, Math.max(0, rawProgress));
        const easedProgress = clampedProgress * clampedProgress * (3 - 2 * clampedProgress);

        const desired = this._autoCameraOffsetScratch;
        desired.lerpVectors(sourceCenter, targetCenter, easedProgress);

        const ref = lane.__followStopRiseRef;
        const smoothAlpha = this._resolveAutoCameraFrameAlpha(0.18 + 0.42 * easedProgress);
        ref.lerp(desired, smoothAlpha);

        const isUpwardHandoff = targetCenter.y >= sourceCenter.y - 0.5;
        if (isUpwardHandoff && Number.isFinite(lane.__followStopRiseLastY) && ref.y < lane.__followStopRiseLastY) {
            ref.y = lane.__followStopRiseLastY;
        }
        lane.__followStopRiseLastY = ref.y;
        out.copy(ref);
        return true;
    }

    _resolveStopRiseReleaseReference(lane, sourceVec, out) {
        if (!lane || !out || !lane.__followStopRiseRef) return false;

        const settledCenter = TMP_CENTER_A;
        const settledOk = this._getVectorWorldCenter(sourceVec, settledCenter)
            || this._getGroupWorldPosition(sourceVec?.group, settledCenter);
        if (!settledOk) {
            this._clearStopRiseFollowState(lane);
            return false;
        }

        if (!lane.__followStopRiseReleaseFrom) {
            lane.__followStopRiseReleaseFrom = lane.__followStopRiseRef.clone();
            lane.__followStopRiseReleaseProgress = 0;
            lane.__followStopRiseReleaseStartedAt = (typeof performance !== 'undefined'
                && typeof performance.now === 'function')
                ? performance.now()
                : Date.now();
        }

        const baseProgress = Number.isFinite(lane.__followStopRiseReleaseProgress)
            ? lane.__followStopRiseReleaseProgress
            : 0;
        const nowMs = (typeof performance !== 'undefined' && typeof performance.now === 'function')
            ? performance.now()
            : Date.now();
        const startedAt = Number.isFinite(lane.__followStopRiseReleaseStartedAt)
            ? lane.__followStopRiseReleaseStartedAt
            : nowMs;
        const elapsedMs = Math.max(0, nowMs - startedAt);
        const timedProgress = STOP_RISE_RELEASE_DURATION_MS > 0
            ? Math.min(1, elapsedMs / STOP_RISE_RELEASE_DURATION_MS)
            : 1;
        const frameScale = (Number.isFinite(this._autoCameraFrameDeltaSec) && this._autoCameraFrameDeltaSec > 0)
            ? (this._autoCameraFrameDeltaSec * AUTO_CAMERA_ALPHA_BASE_HZ)
            : 1;
        const fallbackStep = STOP_RISE_RELEASE_FALLBACK_STEP * Math.min(2.5, Math.max(0.25, frameScale));
        const fallbackProgress = Math.min(1, baseProgress + fallbackStep);
        const nextProgress = Math.max(baseProgress, Math.max(timedProgress, fallbackProgress));
        lane.__followStopRiseReleaseProgress = nextProgress;

        const eased = nextProgress * nextProgress * nextProgress
            * (nextProgress * (nextProgress * 6 - 15) + 10);
        out.lerpVectors(lane.__followStopRiseReleaseFrom, settledCenter, eased);

        if (nextProgress >= 1 || out.distanceToSquared(settledCenter) <= 0.25) {
            out.copy(settledCenter);
            this._clearStopRiseFollowState(lane);
        }
        return true;
    }

    _getGroupWorldPosition(group, out) {
        if (!group || typeof group.getWorldPosition !== 'function') return false;
        group.getWorldPosition(out);
        return Number.isFinite(out.x) && Number.isFinite(out.y) && Number.isFinite(out.z);
    }

    _getVectorWorldCenter(vec, out) {
        const vecGroup = vec?.group;
        const mesh = vec?.mesh;
        const canSample = mesh && typeof mesh.getMatrixAt === 'function' && vecGroup?.matrixWorld;
        if (!canSample) {
            return this._getGroupWorldPosition(vecGroup, out);
        }
        const hideThreshold = HIDE_INSTANCE_Y_OFFSET * 0.5;
        const rawLen = Array.isArray(vec.rawData) ? vec.rawData.length : Infinity;
        const count = Number.isFinite(vec.instanceCount) ? vec.instanceCount : Infinity;
        const length = Math.max(1, Math.min(rawLen, count));
        const firstIndex = Math.max(0, Math.floor((length - 1) / 2));
        const secondIndex = length % 2 === 0 ? Math.min(length - 1, firstIndex + 1) : firstIndex;
        mesh.getMatrixAt(firstIndex, TMP_CENTER_MAT_A);
        out.setFromMatrixPosition(TMP_CENTER_MAT_A).applyMatrix4(vecGroup.matrixWorld);
        if (!Number.isFinite(out.y) || out.y <= hideThreshold) {
            return this._getGroupWorldPosition(vecGroup, out);
        }
        if (secondIndex !== firstIndex) {
            mesh.getMatrixAt(secondIndex, TMP_CENTER_MAT_B);
            TMP_CENTER_B.setFromMatrixPosition(TMP_CENTER_MAT_B).applyMatrix4(vecGroup.matrixWorld);
            if (!Number.isFinite(TMP_CENTER_B.y) || TMP_CENTER_B.y <= hideThreshold) {
                return this._getGroupWorldPosition(vecGroup, out);
            }
            out.add(TMP_CENTER_B).multiplyScalar(0.5);
        }
        return Number.isFinite(out.x) && Number.isFinite(out.y) && Number.isFinite(out.z);
    }

    _findVectorByGroup(lane, group) {
        if (!lane || !group) return null;
        const candidates = [
            lane.originalVec,
            lane.dupVec,
            lane.travellingVec,
            lane.resultVec,
            lane.postAdditionVec,
            lane.movingVecLN2,
            lane.resultVecLN2,
            lane.finalVecAfterMlp,
            lane.addTarget,
            lane.addTargetLN2,
            lane.multTarget,
            lane.multTargetLN2,
        ];
        for (let i = 0; i < candidates.length; i++) {
            const candidate = candidates[i];
            if (candidate && candidate.group === group) return candidate;
        }
        return null;
    }

    _clearAutoCameraOffsets() {
        this._hasAutoCameraOffsets = false;
        this._hasAutoCameraAppliedPose = false;
        this._autoCameraDesiredCameraOffset.set(0, 0, 0);
        this._autoCameraDesiredTargetOffset.set(0, 0, 0);
        this._autoCameraCurrentCameraOffset.set(0, 0, 0);
        this._autoCameraCurrentTargetOffset.set(0, 0, 0);
        this._autoCameraSmoothValid = false;
        this._resetAutoCameraReferenceMotion();
    }

    _shouldCaptureOffsetsFromControlsChange() {
        const engine = this._engine;
        const camera = engine?.camera;
        const target = engine?.controls?.target;
        if (!camera || !target) return false;
        if (!this._hasAutoCameraAppliedPose) return true;

        const cameraDriftSq = camera.position.distanceToSquared(this._autoCameraLastAppliedCameraPosition);
        const targetDriftSq = target.distanceToSquared(this._autoCameraLastAppliedTarget);
        return cameraDriftSq > AUTO_CAMERA_CONTROLS_CAPTURE_EPSILON_SQ
            || targetDriftSq > AUTO_CAMERA_CONTROLS_CAPTURE_EPSILON_SQ;
    }

    _recordAutoCameraAppliedPose() {
        const engine = this._engine;
        const camera = engine?.camera;
        const target = engine?.controls?.target;
        if (!camera || !target) {
            this._hasAutoCameraAppliedPose = false;
            return;
        }
        this._autoCameraLastAppliedCameraPosition.copy(camera.position);
        this._autoCameraLastAppliedTarget.copy(target);
        this._hasAutoCameraAppliedPose = true;
    }

    _setAutoCameraOffsets(cameraOffset, targetOffset, { snap = false } = {}) {
        if (cameraOffset) this._autoCameraDesiredCameraOffset.copy(cameraOffset);
        if (targetOffset) this._autoCameraDesiredTargetOffset.copy(targetOffset);
        if (snap || !this._hasAutoCameraOffsets) {
            this._autoCameraCurrentCameraOffset.copy(this._autoCameraDesiredCameraOffset);
            this._autoCameraCurrentTargetOffset.copy(this._autoCameraDesiredTargetOffset);
        }
        this._hasAutoCameraOffsets = true;
    }

    _captureAutoCameraOffsets(existingReference = null) {
        const engine = this._engine;
        const camera = engine?.camera;
        if (!camera) return false;

        const reference = existingReference || this._autoCameraCenter;
        if (!existingReference) {
            const laneInfo = this._resolveActiveLanePosition(reference);
            if (laneInfo.laneIndex < 0) {
                this._clearAutoCameraOffsets();
                return false;
            }
        }

        this._autoCameraDesiredCameraOffset.copy(camera.position).sub(reference);
        if (!Number.isFinite(this._autoCameraDesiredCameraOffset.x)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.y)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.z)) {
            this._clearAutoCameraOffsets();
            return false;
        }

        const controls = engine?.controls;
        if (controls && controls.target) {
            this._autoCameraDesiredTargetOffset.copy(controls.target).sub(reference);
        } else {
            this._autoCameraDesiredTargetOffset.set(0, 0, 0);
        }

        this._setAutoCameraOffsets(this._autoCameraDesiredCameraOffset, this._autoCameraDesiredTargetOffset, { snap: true });
        return true;
    }

    _applyAutoCamera(reference) {
        if (!this._autoCameraFollow || !this._hasAutoCameraOffsets) {
            return;
        }

        const engine = this._engine;
        const camera = engine?.camera;
        if (!engine || !camera) return;

        if (!Number.isFinite(reference?.x) || !Number.isFinite(reference?.y) || !Number.isFinite(reference?.z)) {
            return;
        }

        if (!Number.isFinite(this._autoCameraDesiredCameraOffset.x)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.y)
            || !Number.isFinite(this._autoCameraDesiredCameraOffset.z)) {
            return;
        }
        const offsetLerpAlpha = this._resolveAutoCameraFrameAlpha(this._autoCameraOffsetLerpAlpha);

        this._suppressControlsChange = true;
        try {
            const camOffset = this._autoCameraCurrentCameraOffset;
            if (!Number.isFinite(camOffset.x) || !Number.isFinite(camOffset.y) || !Number.isFinite(camOffset.z)) {
                camOffset.copy(this._autoCameraDesiredCameraOffset);
            } else if (offsetLerpAlpha > 0) {
                camOffset.lerp(this._autoCameraDesiredCameraOffset, offsetLerpAlpha);
            } else {
                camOffset.copy(this._autoCameraDesiredCameraOffset);
            }
            camera.position.copy(reference).add(camOffset);

            const controls = engine.controls;
            if (controls && controls.target) {
                const targetOffset = this._autoCameraCurrentTargetOffset;
                if (!Number.isFinite(targetOffset.x) || !Number.isFinite(targetOffset.y) || !Number.isFinite(targetOffset.z)) {
                    targetOffset.copy(this._autoCameraDesiredTargetOffset);
                } else if (offsetLerpAlpha > 0) {
                    targetOffset.lerp(this._autoCameraDesiredTargetOffset, offsetLerpAlpha);
                } else {
                    targetOffset.copy(this._autoCameraDesiredTargetOffset);
                }
                controls.target.copy(reference).add(targetOffset);
                if (typeof controls.update === 'function') {
                    controls.update();
                }
            }

            if (typeof engine.notifyCameraUpdated === 'function') {
                engine.notifyCameraUpdated();
            }
            this._recordAutoCameraAppliedPose();
        } finally {
            this._suppressControlsChange = false;
        }
    }

    _updateCameraOffsetOverlay() {
        const overlay = this._cameraOffsetDiv;
        if (!overlay || !this._devMode) {
            if (overlay) overlay.style.display = 'none';
            return;
        }

        if (!this._autoCameraFollow) {
            overlay.style.display = 'none';
            return;
        }

        const engine = this._engine;
        const camera = engine?.camera;
        if (!camera) {
            overlay.style.display = 'block';
            overlay.textContent = 'Offset vs Residual Lane —\nΔx: —\nΔy: —\nΔz: —';
            this._clearAutoCameraOffsets();
            return;
        }

        const reference = this._autoCameraCenter;
        const { laneIndex } = this._resolveActiveLanePosition(reference);
        if (laneIndex < 0 || !Number.isFinite(reference.x) || !Number.isFinite(reference.y) || !Number.isFinite(reference.z)) {
            overlay.style.display = 'block';
            overlay.textContent = 'Offset vs Residual Lane —\nΔx: —\nΔy: —\nΔz: —';
            this._clearAutoCameraOffsets();
            return;
        }

        const offset = this._autoCameraOffsetScratch;
        offset.copy(camera.position).sub(reference);
        const format = (value) => (Number.isFinite(value) ? value.toFixed(2) : '—');
        const laneLabel = Number.isInteger(laneIndex) && laneIndex >= 0 ? (laneIndex + 1) : '—';

        overlay.style.display = 'block';
        overlay.textContent = `Offset vs Residual Lane ${laneLabel}\nΔx: ${format(offset.x)}\nΔy: ${format(offset.y)}\nΔz: ${format(offset.z)}`;
    }

    _isTopLayerNormCameraPhase(layer, lanes) {
        if (!layer || !Array.isArray(lanes) || !lanes.length) return false;

        const topLnStopRiseActive = lanes.some((lane) => lane && lane.__topLnStopRise);
        const topLnActive = lanes.some((lane) => lane
            && (lane.__topLnStopRise
                || lane.__topLnEntered
                || lane.__topLnMultStarted
                || lane.__topLnShiftStarted
                || lane.__topLnShiftComplete));
        if (!topLnActive) return false;

        const forwardComplete = (typeof this._pipeline?.isForwardPassComplete === 'function')
            ? this._pipeline.isForwardPassComplete()
            : false;
        if (forwardComplete) return false;

        // Top-LN flow explicitly parks residual rise with __topLnStopRise while
        // vectors are in the final LayerNorm phase. Keep LN framing during this
        // control window regardless of top-embedding entry thresholds.
        if (topLnStopRiseActive) return true;

        const entryY = Number.isFinite(layer.__topEmbedEntryYLocal) ? layer.__topEmbedEntryYLocal : null;
        if (!Number.isFinite(entryY)) return true;

        const reachedProjectionZone = lanes.some((lane) => {
            const y = lane?.originalVec?.group?.position?.y;
            return Number.isFinite(y) && y >= entryY - 0.5;
        });
        return !reachedProjectionZone;
    }

    _resolveAutoCameraViewKey() {
        const layers = this._pipeline?._layers;
        const nowMs = this._getNowMs();
        const result = resolveAutoCameraViewState({
            pipeline: this._pipeline,
            layers,
            currentLayerIdx: this._pipeline?._currentLayerIdx ?? 0,
            priorViewKey: this._autoCameraViewKey || 'default',
            nowMs,
            isLargeDesktopViewport: this._autoCameraLayerEndDesktopOffsetsEnabled && this._isLargeDesktopViewport(),
            isTopLayerNormCameraPhase: (layer, lanes) => this._isTopLayerNormCameraPhase(layer, lanes),
        });
        this._autoCameraViewContext = result.viewContext;
        const rawKey = result.rawKey || 'default';
        if (this._autoCameraForceEmbedVocabStartLock && this._autoCameraEmbedVocabOffsetsEnabled) {
            const positionGate = this._pipeline?.__inputPositionChipGate;
            const positionGatePrimed = !!(positionGate
                && positionGate.enabled !== false
                && positionGate.pending === false);
            if (!positionGatePrimed) {
                return 'embed-vocab';
            }
            this._autoCameraForceEmbedVocabStartLock = false;
        }
        return rawKey;
    }

    _getNowMs() {
        return (typeof performance !== 'undefined' && typeof performance.now === 'function')
            ? performance.now()
            : Date.now();
    }

    _updateAutoCameraFrameTiming(nowMs = null) {
        const currentNowMs = Number.isFinite(nowMs) ? nowMs : this._getNowMs();
        const previousNowMs = this._autoCameraLastUpdateMs;

        let dtSec = AUTO_CAMERA_FRAME_DELTA_SEC_DEFAULT;
        if (Number.isFinite(previousNowMs) && previousNowMs > 0) {
            const rawDtSec = Math.max(0, (currentNowMs - previousNowMs) / 1000);
            if (Number.isFinite(rawDtSec) && rawDtSec > 0) {
                dtSec = rawDtSec;
            } else if (Number.isFinite(this._autoCameraFrameDeltaSec) && this._autoCameraFrameDeltaSec > 0) {
                dtSec = this._autoCameraFrameDeltaSec;
            }
        } else if (Number.isFinite(this._autoCameraFrameDeltaSec) && this._autoCameraFrameDeltaSec > 0) {
            dtSec = this._autoCameraFrameDeltaSec;
        }

        dtSec = Math.min(
            AUTO_CAMERA_FRAME_DELTA_SEC_MAX,
            Math.max(AUTO_CAMERA_FRAME_DELTA_SEC_MIN, dtSec)
        );
        this._autoCameraFrameDeltaSec = dtSec;
        this._autoCameraLastUpdateMs = currentNowMs;
        return currentNowMs;
    }

    _resolveAutoCameraFrameAlpha(baseAlpha) {
        const alpha = Number.isFinite(baseAlpha) ? baseAlpha : 0;
        if (alpha <= 0) return 0;
        if (alpha >= 1) return 1;

        const dtSec = Number.isFinite(this._autoCameraFrameDeltaSec) && this._autoCameraFrameDeltaSec > 0
            ? this._autoCameraFrameDeltaSec
            : AUTO_CAMERA_FRAME_DELTA_SEC_DEFAULT;
        const frameScale = dtSec * AUTO_CAMERA_ALPHA_BASE_HZ;
        const remainder = Math.max(0, 1 - alpha);
        const scaledAlpha = 1 - Math.pow(remainder, frameScale);
        return Math.min(1, Math.max(0, scaledAlpha));
    }

    _resetAutoCameraReferenceMotion(reference = null) {
        if (reference && Number.isFinite(reference.x) && Number.isFinite(reference.y) && Number.isFinite(reference.z)) {
            this._autoCameraPrevReference.copy(reference);
            this._autoCameraPrevReferenceValid = true;
        } else {
            this._autoCameraPrevReference.set(0, 0, 0);
            this._autoCameraPrevReferenceValid = false;
        }
        this._autoCameraReferenceMotionRampT = 0;
    }

    _updateAutoCameraReferenceMotionRamp(reference, { suspend = false } = {}) {
        if (!reference || !Number.isFinite(reference.x) || !Number.isFinite(reference.y) || !Number.isFinite(reference.z)) {
            this._resetAutoCameraReferenceMotion();
            return 1;
        }

        const dtSec = Number.isFinite(this._autoCameraFrameDeltaSec) && this._autoCameraFrameDeltaSec > 0
            ? this._autoCameraFrameDeltaSec
            : AUTO_CAMERA_FRAME_DELTA_SEC_DEFAULT;

        if (suspend) {
            this._autoCameraPrevReference.copy(reference);
            this._autoCameraPrevReferenceValid = true;
            this._autoCameraReferenceMotionRampT = 0;
            return 1;
        }

        if (!this._autoCameraPrevReferenceValid) {
            this._autoCameraPrevReference.copy(reference);
            this._autoCameraPrevReferenceValid = true;
            this._autoCameraReferenceMotionRampT = 0;
            return 1;
        }

        const deltaDist = reference.distanceTo(this._autoCameraPrevReference);
        this._autoCameraPrevReference.copy(reference);
        const speedUnitsPerSec = deltaDist / Math.max(1e-4, dtSec);

        const startSpeed = Math.max(0, this._autoCameraReferenceMotionStartSpeed);
        const stopSpeed = Math.min(
            startSpeed,
            Math.max(0, this._autoCameraReferenceMotionStopSpeed)
        );
        const hasRamp = this._autoCameraReferenceMotionRampT > 0.0001;
        const movingNow = speedUnitsPerSec >= (hasRamp ? stopSpeed : startSpeed);

        const rampUpStep = dtSec / Math.max(0.02, this._autoCameraReferenceMotionRampUpSec);
        const rampDownStep = dtSec / Math.max(0.02, this._autoCameraReferenceMotionRampDownSec);
        if (movingNow) {
            this._autoCameraReferenceMotionRampT = Math.min(1, this._autoCameraReferenceMotionRampT + rampUpStep);
        } else {
            this._autoCameraReferenceMotionRampT = Math.max(0, this._autoCameraReferenceMotionRampT - rampDownStep);
        }

        if (!movingNow && this._autoCameraReferenceMotionRampT <= 0.0001) {
            return 1;
        }

        const t = this._autoCameraReferenceMotionRampT;
        const easedT = t * t * (3 - 2 * t);
        const minScale = Math.min(1, Math.max(0, this._autoCameraReferenceMotionMinScale));
        return THREE.MathUtils.lerp(minScale, 1, easedT);
    }

    _getAutoCameraViewSwitchHoldMs(fromKey, toKey, viewContext = null) {
        return getAutoCameraViewSwitchHoldMs({
            fromKey,
            toKey,
            viewContext,
            baseHoldMs: this._autoCameraViewSwitchHoldMs
        });
    }

    _resolveStableAutoCameraViewKey(rawKey, viewContext = null) {
        const result = resolveStableAutoCameraViewKey({
            rawKey,
            currentKey: this._autoCameraViewKey || 'default',
            pendingKey: this._autoCameraViewPendingKey,
            pendingSinceMs: this._autoCameraViewPendingSinceMs,
            nowMs: this._getNowMs(),
            baseHoldMs: this._autoCameraViewSwitchHoldMs,
            viewContext
        });
        this._autoCameraViewPendingKey = result.pendingKey;
        this._autoCameraViewPendingSinceMs = result.pendingSinceMs;
        return result.key;
    }

    _resolveAutoCameraOffsetsForViewKey(key, out = this._autoCameraResolvedViewOffsets) {
        out.cameraOffset = this._autoCameraDefaultCameraOffset;
        out.targetOffset = this._autoCameraDefaultTargetOffset;

        if (key === 'layer-end-desktop' && this._autoCameraLayerEndDesktopOffsetsEnabled) {
            out.cameraOffset = this._autoCameraLayerEndDesktopCameraOffset;
            out.targetOffset = this._autoCameraLayerEndDesktopTargetOffset;
        } else if (key === 'embed-vocab' && this._autoCameraEmbedVocabOffsetsEnabled) {
            out.cameraOffset = this._autoCameraEmbedVocabCameraOffset;
            out.targetOffset = this._autoCameraEmbedVocabTargetOffset;
        } else if (key === 'embed-position' && this._autoCameraEmbedPositionOffsetsEnabled) {
            out.cameraOffset = this._autoCameraEmbedPositionCameraOffset;
            out.targetOffset = this._autoCameraEmbedPositionTargetOffset;
        } else if (key === 'embed-add' && this._autoCameraEmbedAddOffsetsEnabled) {
            out.cameraOffset = this._autoCameraEmbedAddCameraOffset;
            out.targetOffset = this._autoCameraEmbedAddTargetOffset;
        } else if (key === 'top-ln' && this._autoCameraTopLnOffsetsEnabled) {
            out.cameraOffset = this._autoCameraTopLnCameraOffset;
            out.targetOffset = this._autoCameraTopLnTargetOffset;
        } else if (key === 'ln' && this._autoCameraLnOffsetsEnabled) {
            out.cameraOffset = this._autoCameraLnCameraOffset;
            out.targetOffset = this._autoCameraLnTargetOffset;
        } else if (key === 'travel' && this._autoCameraTravelOffsetsEnabled) {
            out.cameraOffset = this._autoCameraTravelCameraOffset;
            out.targetOffset = this._autoCameraTravelTargetOffset;
        } else if (key === 'mhsa' && this._autoCameraMhsaOffsetsEnabled) {
            out.cameraOffset = this._autoCameraMhsaCameraOffset;
            out.targetOffset = this._autoCameraMhsaTargetOffset;
        } else if (key === 'concat' && this._autoCameraConcatOffsetsEnabled) {
            out.cameraOffset = this._autoCameraConcatCameraOffset;
            out.targetOffset = this._autoCameraConcatTargetOffset;
        } else if (key === 'final' && this._autoCameraFinalOffsetsEnabled) {
            out.cameraOffset = this._autoCameraFinalCameraOffset;
            out.targetOffset = this._autoCameraFinalTargetOffset;
        }

        return out;
    }

    _applyAutoCameraViewOffsets() {
        this._updateAutoCameraScaledOffsets();
        const rawKey = this._resolveAutoCameraViewKey();
        const viewContext = this._autoCameraViewContext;
        const key = this._resolveStableAutoCameraViewKey(rawKey, viewContext);
        const resolvedOffsets = this._resolveAutoCameraOffsetsForViewKey(key);
        let camOffset = resolvedOffsets.cameraOffset;
        let targetOffset = resolvedOffsets.targetOffset;

        const previousKey = this._autoCameraViewKey;
        if (key !== previousKey) {
            let blendAlpha = this._autoCameraViewBlendAlphaTransition;
            if (previousKey === 'concat' && key === 'default') {
                blendAlpha = Math.min(blendAlpha, 0.026);
            }
            this._autoCameraViewBlendAlphaActive = blendAlpha;
            this._autoCameraViewKey = key;
            if (!this._autoCameraSmoothValid) {
                this._autoCameraViewBlendT = 1;
                this._autoCameraViewFromCameraOffset.copy(camOffset);
                this._autoCameraViewFromTargetOffset.copy(targetOffset);
            } else {
                this._autoCameraViewBlendT = 0;
                this._autoCameraViewFromCameraOffset.copy(this._autoCameraCurrentCameraOffset);
                this._autoCameraViewFromTargetOffset.copy(this._autoCameraCurrentTargetOffset);
            }
        }

        this._autoCameraViewToCameraOffset.copy(camOffset);
        this._autoCameraViewToTargetOffset.copy(targetOffset);

        if (this._autoCameraViewBlendT < 1) {
            const blendStep = this._resolveAutoCameraFrameAlpha(this._autoCameraViewBlendAlphaActive);
            this._autoCameraViewBlendT = Math.min(1, this._autoCameraViewBlendT + blendStep);
            const linearT = this._autoCameraViewBlendT;
            const t = linearT * linearT * (3 - 2 * linearT);
            this._autoCameraViewBlendCameraOffset.copy(this._autoCameraViewFromCameraOffset).lerp(this._autoCameraViewToCameraOffset, t);
            this._autoCameraViewBlendTargetOffset.copy(this._autoCameraViewFromTargetOffset).lerp(this._autoCameraViewToTargetOffset, t);
            camOffset = this._autoCameraViewBlendCameraOffset;
            targetOffset = this._autoCameraViewBlendTargetOffset;
        }

        this._setAutoCameraOffsets(camOffset, targetOffset, { snap: false });
    }

    _computeAutoCameraScale() {
        if (typeof window === 'undefined') return 1.0;
        const width = window.innerWidth || 0;
        if (!Number.isFinite(width) || width <= 0) return 1.0;
        if (width >= this._autoCameraScaleMaxWidth) return 1.0;
        if (width <= this._autoCameraScaleMinWidth) return this._autoCameraScaleMax;
        const t = (this._autoCameraScaleMaxWidth - width)
            / (this._autoCameraScaleMaxWidth - this._autoCameraScaleMinWidth);
        return 1.0 + t * (this._autoCameraScaleMax - 1.0);
    }

    _computeAutoCameraMobileFactor() {
        if (typeof window === 'undefined') return 0;
        if (typeof window.matchMedia === 'function') {
            const coarse = window.matchMedia('(pointer: coarse)').matches
                || window.matchMedia('(hover: none) and (pointer: coarse)').matches;
            if (coarse) return 1;
        }
        const touchPoints = Number.isFinite(window?.navigator?.maxTouchPoints)
            ? window.navigator.maxTouchPoints
            : 0;
        if (touchPoints > 0) return 1;
        return 0;
    }

    _isLargeDesktopViewport() {
        if (typeof window === 'undefined') return false;
        if (this._computeAutoCameraMobileFactor() > 0.0001) return false;
        const width = window.innerWidth || 0;
        if (!Number.isFinite(width) || width <= 0) return false;
        return width >= this._autoCameraScaleMaxWidth;
    }

    _updateAutoCameraScaledOffsets(force = false) {
        const scale = this._computeAutoCameraScale();
        const mobileFactor = this._computeAutoCameraMobileFactor();
        const shiftFactor = (this._autoCameraScaleMax > 1.0)
            ? (scale - 1.0) / Math.max(0.0001, this._autoCameraScaleMax - 1.0)
            : 0;
        const shiftX = Math.abs(this._autoCameraShiftMaxX) > 0.0001
            ? this._autoCameraShiftMaxX * shiftFactor
            : 0;
        const shiftTravelX = Math.abs(this._autoCameraShiftMaxTravelX) > 0.0001
            ? this._autoCameraShiftMaxTravelX * shiftFactor
            : 0;
        const shiftMhsaX = Math.abs(this._autoCameraShiftMaxMhsaX) > 0.0001
            ? this._autoCameraShiftMaxMhsaX * shiftFactor
            : 0;
        if (!force
            && Math.abs(scale - this._autoCameraScaleLast) < 0.001
            && Math.abs(shiftX - this._autoCameraShiftLastX) < 0.5
            && Math.abs(mobileFactor - this._autoCameraMobileFactorLast) < 0.001) {
            return;
        }
        this._autoCameraScaleLast = scale;
        this._autoCameraShiftLastX = shiftX;
        this._autoCameraMobileFactorLast = mobileFactor;

        const applyScaleShift = (dest, base, extraShiftX = 0) => {
            dest.copy(base).multiplyScalar(scale);
            if (shiftX || extraShiftX) dest.x += shiftX + extraShiftX;
        };

        applyScaleShift(this._autoCameraDefaultCameraOffset, this._autoCameraDefaultCameraOffsetBase);
        applyScaleShift(this._autoCameraDefaultTargetOffset, this._autoCameraDefaultTargetOffsetBase);
        if (this._autoCameraEmbedVocabOffsetsEnabled) {
            applyScaleShift(this._autoCameraEmbedVocabCameraOffset, this._autoCameraEmbedVocabCameraOffsetBase);
            applyScaleShift(this._autoCameraEmbedVocabTargetOffset, this._autoCameraEmbedVocabTargetOffsetBase);
        }
        if (this._autoCameraEmbedPositionOffsetsEnabled) {
            applyScaleShift(this._autoCameraEmbedPositionCameraOffset, this._autoCameraEmbedPositionCameraOffsetBase);
            applyScaleShift(this._autoCameraEmbedPositionTargetOffset, this._autoCameraEmbedPositionTargetOffsetBase);
        }
        if (this._autoCameraEmbedAddOffsetsEnabled) {
            applyScaleShift(this._autoCameraEmbedAddCameraOffset, this._autoCameraEmbedAddCameraOffsetBase);
            applyScaleShift(this._autoCameraEmbedAddTargetOffset, this._autoCameraEmbedAddTargetOffsetBase);
        }

        if (this._autoCameraMhsaOffsetsEnabled) {
            applyScaleShift(this._autoCameraMhsaCameraOffset, this._autoCameraMhsaCameraOffsetBase, shiftMhsaX);
            applyScaleShift(this._autoCameraMhsaTargetOffset, this._autoCameraMhsaTargetOffsetBase, shiftMhsaX);
            if (this._autoCameraMhsaMobileOverrideEnabled && mobileFactor > 0.0001) {
                this._autoCameraMhsaCameraOffset.lerp(
                    this._autoCameraMhsaMobileOverrideCameraOffset,
                    mobileFactor
                );
                this._autoCameraMhsaTargetOffset.lerp(
                    this._autoCameraMhsaMobileOverrideTargetOffset,
                    mobileFactor
                );
            }
        }
        if (this._autoCameraConcatOffsetsEnabled) {
            applyScaleShift(this._autoCameraConcatCameraOffset, this._autoCameraConcatCameraOffsetBase);
            applyScaleShift(this._autoCameraConcatTargetOffset, this._autoCameraConcatTargetOffsetBase);
            if (this._autoCameraConcatMobileOverrideEnabled && mobileFactor > 0.0001) {
                this._autoCameraConcatCameraOffset.lerp(
                    this._autoCameraConcatMobileOverrideCameraOffset,
                    mobileFactor
                );
                this._autoCameraConcatTargetOffset.lerp(
                    this._autoCameraConcatMobileOverrideTargetOffset,
                    mobileFactor
                );
            }
        }
        if (this._autoCameraLnOffsetsEnabled) {
            applyScaleShift(this._autoCameraLnCameraOffset, this._autoCameraLnCameraOffsetBase);
            applyScaleShift(this._autoCameraLnTargetOffset, this._autoCameraLnTargetOffsetBase);
        }
        if (this._autoCameraTopLnOffsetsEnabled) {
            applyScaleShift(this._autoCameraTopLnCameraOffset, this._autoCameraTopLnCameraOffsetBase);
            applyScaleShift(this._autoCameraTopLnTargetOffset, this._autoCameraTopLnTargetOffsetBase);
        }
        if (this._autoCameraTravelOffsetsEnabled) {
            applyScaleShift(this._autoCameraTravelCameraOffset, this._autoCameraTravelCameraOffsetBase, shiftTravelX);
            applyScaleShift(this._autoCameraTravelTargetOffset, this._autoCameraTravelTargetOffsetBase, shiftTravelX);
            if (this._autoCameraTravelMobileOverrideEnabled && mobileFactor > 0.0001) {
                this._autoCameraTravelCameraOffset.lerp(
                    this._autoCameraTravelMobileOverrideCameraOffset,
                    mobileFactor
                );
                this._autoCameraTravelTargetOffset.lerp(
                    this._autoCameraTravelMobileOverrideTargetOffset,
                    mobileFactor
                );
            }
        }
        if (this._autoCameraFinalOffsetsEnabled) {
            applyScaleShift(this._autoCameraFinalCameraOffset, this._autoCameraFinalCameraOffsetBase);
            applyScaleShift(this._autoCameraFinalTargetOffset, this._autoCameraFinalTargetOffsetBase);
            if (this._autoCameraFinalMobileOverrideEnabled && mobileFactor > 0.0001) {
                this._autoCameraFinalCameraOffset.lerp(
                    this._autoCameraFinalMobileOverrideCameraOffset,
                    mobileFactor
                );
                this._autoCameraFinalTargetOffset.lerp(
                    this._autoCameraFinalMobileOverrideTargetOffset,
                    mobileFactor
                );
            }
        }
        if (this._autoCameraLayerEndDesktopOffsetsEnabled) {
            applyScaleShift(this._autoCameraLayerEndDesktopCameraOffset, this._autoCameraLayerEndDesktopCameraOffsetBase);
            applyScaleShift(this._autoCameraLayerEndDesktopTargetOffset, this._autoCameraLayerEndDesktopTargetOffsetBase);
        }
    }

    _updateAutoCameraFollow() {
        if (!this._autoCameraFollow) return false;
        const engine = this._engine;
        const camera = engine?.camera;
        if (!camera) return false;
        const nowMs = this._updateAutoCameraFrameTiming();

        this._applyAutoCameraViewOffsets();

        const reference = this._autoCameraCenter;
        const { laneIndex } = this._resolveActiveLanePosition(reference);
        if (laneIndex < 0 || !Number.isFinite(reference.x) || !Number.isFinite(reference.y) || !Number.isFinite(reference.z)) {
            this._clearAutoCameraOffsets();
            return false;
        }

        const viewContext = this._autoCameraViewContext;
        const laneInView = viewContext?.lane || null;
        const residualHoldUntilMs = Number.isFinite(viewContext?.residualAddHoldUntilMs)
            ? viewContext.residualAddHoldUntilMs
            : 0;
        if (residualHoldUntilMs > nowMs) {
            if (!this._autoCameraPostAddLockActive || residualHoldUntilMs > this._autoCameraPostAddLockUntilMs) {
                const lockSource = this._autoCameraSmoothValid ? this._autoCameraSmoothedRef : reference;
                this._autoCameraPostAddLockRef.copy(lockSource);
                this._autoCameraPostAddLockUntilMs = residualHoldUntilMs;
                this._autoCameraPostAddLockActive = true;
            }
        } else if (this._autoCameraPostAddLockActive && nowMs >= this._autoCameraPostAddLockUntilMs) {
            this._autoCameraPostAddLockActive = false;
            this._autoCameraPostAddLockUntilMs = 0;
        }
        const inStopRiseTransition = !!(laneInView
            && (laneInView.stopRise || laneInView.__followStopRiseReleaseFrom));
        const inResidualAddTransition = !!(viewContext
            && (viewContext.anyResidualAddActive || viewContext.anyResidualAddReleaseHold));
        const inLn2PreStartTransition = !!(viewContext && viewContext.holdViewUntilLn2Inside);
        const inViewModeTransition = this._autoCameraViewBlendT < 1
            || (this._autoCameraViewPendingKey && this._autoCameraViewPendingKey !== this._autoCameraViewKey);
        let refSmoothAlpha = this._autoCameraSmoothAlpha;
        if (inViewModeTransition) {
            refSmoothAlpha = Math.min(refSmoothAlpha, 0.02);
        }
        if (viewContext && viewContext.inLayerHandoff) {
            refSmoothAlpha = Math.min(refSmoothAlpha, 0.02);
        }
        if (inStopRiseTransition) {
            refSmoothAlpha = Math.min(refSmoothAlpha, 0.018);
        }
        if (inResidualAddTransition) {
            refSmoothAlpha = Math.min(refSmoothAlpha, 0.014);
        }
        if (inLn2PreStartTransition) {
            refSmoothAlpha = Math.min(refSmoothAlpha, 0.014);
        }
        const postAddLockActiveNow = this._autoCameraPostAddLockActive
            && nowMs < this._autoCameraPostAddLockUntilMs;
        const motionRampScale = this._updateAutoCameraReferenceMotionRamp(reference, {
            suspend: postAddLockActiveNow
        });
        refSmoothAlpha *= motionRampScale;
        let followRef = reference;
        const resolvedRefSmoothAlpha = this._resolveAutoCameraFrameAlpha(refSmoothAlpha);
        if (resolvedRefSmoothAlpha > 0) {
            if (!this._autoCameraSmoothValid) {
                this._autoCameraSmoothedRef.copy(reference);
                this._autoCameraSmoothValid = true;
            } else {
                this._autoCameraSmoothedRef.lerp(reference, resolvedRefSmoothAlpha);
            }
            followRef = this._autoCameraSmoothedRef;
        }
        if (postAddLockActiveNow) {
            this._autoCameraSmoothedRef.copy(this._autoCameraPostAddLockRef);
            this._autoCameraSmoothValid = true;
            followRef = this._autoCameraPostAddLockRef;
        }

        if (!this._hasAutoCameraOffsets) {
            this._captureAutoCameraOffsets(followRef);
        }

        this._applyAutoCamera(followRef);
        return true;
    }
}
