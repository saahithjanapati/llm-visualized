import * as THREE from 'three';
import { HIDE_INSTANCE_Y_OFFSET } from '../utils/constants.js';

const TMP_CENTER_A = new THREE.Vector3();
const TMP_CENTER_B = new THREE.Vector3();
const TMP_CENTER_MAT_A = new THREE.Matrix4();
const TMP_CENTER_MAT_B = new THREE.Matrix4();

const coerceVector3 = (value, fallback = null) => {
    if (value instanceof THREE.Vector3) return value.clone();
    if (value && typeof value === 'object' && 'x' in value && 'y' in value && 'z' in value) {
        return new THREE.Vector3(value.x, value.y, value.z);
    }
    return fallback ? fallback.clone() : null;
};

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
            if (resetView) {
                const canSmoothReset = smoothReset && this._captureAutoCameraOffsets();
                this._setAutoCameraOffsets(
                    this._autoCameraDefaultCameraOffset,
                    this._autoCameraDefaultTargetOffset,
                    { snap: !canSmoothReset }
                );
            }
        }
        if (this._autoCameraFollow) {
            if (immediate) {
                this._updateAutoCameraFollow();
            }
        } else {
            this._clearAutoCameraOffsets();
        }
        this._updateCameraOffsetOverlay();
    }

    isEnabled() {
        return !!this._autoCameraFollow;
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
        const mlpBlendAlpha = typeof opts.autoCameraViewBlendAlphaMlpReturn === 'number'
            ? opts.autoCameraViewBlendAlphaMlpReturn
            : (this._autoCameraViewBlendAlpha * 0.35);
        this._autoCameraViewBlendAlphaMlpReturn = Math.min(1, Math.max(0, mlpBlendAlpha));
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
        this._autoCameraSmoothedRef = new THREE.Vector3();
        this._autoCameraSmoothValid = false;
        this._autoCameraViewKey = 'default';
        this._autoCameraViewBlendT = 1;
        this._autoCameraViewFromCameraOffset = new THREE.Vector3();
        this._autoCameraViewFromTargetOffset = new THREE.Vector3();
        this._autoCameraViewToCameraOffset = new THREE.Vector3();
        this._autoCameraViewToTargetOffset = new THREE.Vector3();
        this._autoCameraViewBlendCameraOffset = new THREE.Vector3();
        this._autoCameraViewBlendTargetOffset = new THREE.Vector3();
        this._autoCameraViewContext = null;
        this._autoCameraDefaultCameraOffset = new THREE.Vector3();
        this._autoCameraDefaultTargetOffset = new THREE.Vector3();
        this._autoCameraMhsaCameraOffset = new THREE.Vector3();
        this._autoCameraMhsaTargetOffset = new THREE.Vector3();
        this._autoCameraConcatCameraOffset = new THREE.Vector3();
        this._autoCameraConcatTargetOffset = new THREE.Vector3();
        this._autoCameraMhsaOffsetsEnabled = false;
        this._autoCameraConcatOffsetsEnabled = false;
        this._autoCameraLnCameraOffset = new THREE.Vector3();
        this._autoCameraLnTargetOffset = new THREE.Vector3();
        this._autoCameraTravelCameraOffset = new THREE.Vector3();
        this._autoCameraTravelTargetOffset = new THREE.Vector3();
        this._autoCameraLnOffsetsEnabled = false;
        this._autoCameraTravelOffsetsEnabled = false;
        this._autoCameraDefaultCameraOffsetBase = new THREE.Vector3();
        this._autoCameraDefaultTargetOffsetBase = new THREE.Vector3();
        this._autoCameraMhsaCameraOffsetBase = new THREE.Vector3();
        this._autoCameraMhsaTargetOffsetBase = new THREE.Vector3();
        this._autoCameraConcatCameraOffsetBase = new THREE.Vector3();
        this._autoCameraConcatTargetOffsetBase = new THREE.Vector3();
        this._autoCameraLnCameraOffsetBase = new THREE.Vector3();
        this._autoCameraLnTargetOffsetBase = new THREE.Vector3();
        this._autoCameraTravelCameraOffsetBase = new THREE.Vector3();
        this._autoCameraTravelTargetOffsetBase = new THREE.Vector3();
        this._autoCameraInspectorRef = new THREE.Vector3();
        this._hasAutoCameraOffsets = false;
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

        this._updateAutoCameraScaledOffsets(true);

        if (this._engine?.controls) {
            this._controlsChangeHandler = () => {
                if (!this._autoCameraFollow || this._suppressControlsChange) {
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

        const layerIndex = Math.min(this._pipeline._currentLayerIdx, layers.length - 1);
        const layer = layers[layerIndex];
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
            } else if (!this._getVectorWorldCenter(vec, targetVec)) {
                this._clearStopRiseFollowState(lane);
                vecGroup.getWorldPosition(targetVec);
            } else {
                this._clearStopRiseFollowState(lane);
            }
        }

        return { laneIndex, laneCount };
    }

    _clearStopRiseFollowState(lane) {
        if (!lane) return;
        delete lane.__followStopRiseTarget;
        delete lane.__followStopRiseLastY;
        delete lane.__followStopRiseRef;
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

        const rawProgress = Number.isFinite(lane.mhsaResidualAddProgress)
            ? lane.mhsaResidualAddProgress
            : 0;
        const clampedProgress = Math.min(1, Math.max(0, rawProgress));
        const easedProgress = clampedProgress * clampedProgress * (3 - 2 * clampedProgress);

        const desired = this._autoCameraOffsetScratch;
        desired.lerpVectors(sourceCenter, targetCenter, easedProgress);

        const ref = lane.__followStopRiseRef;
        const smoothAlpha = 0.18 + 0.42 * easedProgress;
        ref.lerp(desired, smoothAlpha);

        const isUpwardHandoff = targetCenter.y >= sourceCenter.y - 0.5;
        if (isUpwardHandoff && Number.isFinite(lane.__followStopRiseLastY) && ref.y < lane.__followStopRiseLastY) {
            ref.y = lane.__followStopRiseLastY;
        }
        lane.__followStopRiseLastY = ref.y;
        out.copy(ref);
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
        this._autoCameraDesiredCameraOffset.set(0, 0, 0);
        this._autoCameraDesiredTargetOffset.set(0, 0, 0);
        this._autoCameraCurrentCameraOffset.set(0, 0, 0);
        this._autoCameraCurrentTargetOffset.set(0, 0, 0);
        this._autoCameraSmoothValid = false;
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

        this._suppressControlsChange = true;
        try {
            const camOffset = this._autoCameraCurrentCameraOffset;
            if (!Number.isFinite(camOffset.x) || !Number.isFinite(camOffset.y) || !Number.isFinite(camOffset.z)) {
                camOffset.copy(this._autoCameraDesiredCameraOffset);
            } else if (this._autoCameraOffsetLerpAlpha > 0) {
                camOffset.lerp(this._autoCameraDesiredCameraOffset, this._autoCameraOffsetLerpAlpha);
            } else {
                camOffset.copy(this._autoCameraDesiredCameraOffset);
            }
            camera.position.copy(reference).add(camOffset);

            const controls = engine.controls;
            if (controls && controls.target) {
                const targetOffset = this._autoCameraCurrentTargetOffset;
                if (!Number.isFinite(targetOffset.x) || !Number.isFinite(targetOffset.y) || !Number.isFinite(targetOffset.z)) {
                    targetOffset.copy(this._autoCameraDesiredTargetOffset);
                } else if (this._autoCameraOffsetLerpAlpha > 0) {
                    targetOffset.lerp(this._autoCameraDesiredTargetOffset, this._autoCameraOffsetLerpAlpha);
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

        const topLnActive = lanes.some((lane) => lane
            && (lane.__topLnEntered || lane.__topLnMultStarted || lane.__topLnShiftStarted || lane.__topLnShiftComplete));
        if (!topLnActive) return false;

        const forwardComplete = (typeof this._pipeline?.isForwardPassComplete === 'function')
            ? this._pipeline.isForwardPassComplete()
            : false;
        if (forwardComplete) return false;

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
        if (!Array.isArray(layers) || !layers.length) return 'default';
        const layerIndex = Math.min(this._pipeline._currentLayerIdx, layers.length - 1);
        const layer = layers[layerIndex];
        const mhsa = layer?.mhsaAnimation;
        const lanes = Array.isArray(layer?.lanes) ? layer.lanes : [];
        const laneCount = lanes.length;
        const laneIndex = laneCount ? Math.min(laneCount - 1, Math.floor(laneCount / 2)) : -1;
        const lane = laneIndex >= 0 ? lanes[laneIndex] : null;
        const inLaneLn = !!(lane && (lane.horizPhase === 'insideLN' || lane.ln2Phase === 'insideLN'));
        const inTopLn = this._isTopLayerNormCameraPhase(layer, lanes);
        const inLn = inLaneLn || inTopLn;
        this._autoCameraViewContext = { lane, laneIndex, laneCount, inLn, inTopLn };
        const passPhase = mhsa?.mhaPassThroughPhase || 'positioning_mha_vectors';
        const inTravel = !!(lane && lane.horizPhase === 'travelMHSA'
            && passPhase === 'positioning_mha_vectors');
        const inCopyStage = !!(lane && lane.horizPhase === 'finishedHeads'
            && (passPhase === 'positioning_mha_vectors' || passPhase === 'ready_for_parallel_pass_through'));
        if (!mhsa) {
            if (inLn) return 'ln';
            return 'default';
        }

        if (inLn) {
            return 'ln';
        }
        if (inTravel || inCopyStage) {
            return 'travel';
        }

        const outputPhase = mhsa.outputProjMatrixAnimationPhase || 'waiting';
        const rowPhase = mhsa.rowMergePhase || 'not_started';
        const outputReturnComplete = mhsa.outputProjMatrixReturnComplete === true;
        const concatActive = (rowPhase === 'merging' || rowPhase === 'merged')
            && outputPhase === 'waiting'
            && !outputReturnComplete;
        if (concatActive) {
            return 'concat';
        }

        const mhsaGate = rowPhase === 'not_started' && outputPhase === 'waiting';
        const mhsaActive = mhsaGate && (passPhase === 'parallel_pass_through_active'
            || passPhase === 'mha_pass_through_complete'
            || (passPhase === 'ready_for_parallel_pass_through' && !inCopyStage));
        if (mhsaActive) {
            return 'mhsa';
        }
        return 'default';
    }

    _applyAutoCameraViewOffsets() {
        this._updateAutoCameraScaledOffsets();
        const key = this._resolveAutoCameraViewKey();
        const viewContext = this._autoCameraViewContext;
        let camOffset = this._autoCameraDefaultCameraOffset;
        let targetOffset = this._autoCameraDefaultTargetOffset;

        if (key === 'ln' && this._autoCameraLnOffsetsEnabled) {
            camOffset = this._autoCameraLnCameraOffset;
            targetOffset = this._autoCameraLnTargetOffset;
        } else if (key === 'travel' && this._autoCameraTravelOffsetsEnabled) {
            camOffset = this._autoCameraTravelCameraOffset;
            targetOffset = this._autoCameraTravelTargetOffset;
        } else if (key === 'mhsa' && this._autoCameraMhsaOffsetsEnabled) {
            camOffset = this._autoCameraMhsaCameraOffset;
            targetOffset = this._autoCameraMhsaTargetOffset;
        } else if (key === 'concat' && this._autoCameraConcatOffsetsEnabled) {
            camOffset = this._autoCameraConcatCameraOffset;
            targetOffset = this._autoCameraConcatTargetOffset;
        }

        if (key !== this._autoCameraViewKey) {
            const priorKey = this._autoCameraViewKey;
            const isMlpTransition = priorKey === 'ln'
                && key === 'default'
                && viewContext
                && !viewContext.inTopLn
                && viewContext.lane
                && viewContext.lane.ln2Phase
                && viewContext.lane.ln2Phase !== 'insideLN';
            this._autoCameraViewBlendAlphaActive = isMlpTransition
                ? this._autoCameraViewBlendAlphaMlpReturn
                : this._autoCameraViewBlendAlpha;
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
            this._autoCameraViewBlendT = Math.min(1, this._autoCameraViewBlendT + this._autoCameraViewBlendAlphaActive);
            const t = this._autoCameraViewBlendT;
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
    }

    _updateAutoCameraFollow() {
        if (!this._autoCameraFollow) return false;
        const engine = this._engine;
        const camera = engine?.camera;
        if (!camera) return false;

        this._applyAutoCameraViewOffsets();

        const reference = this._autoCameraCenter;
        const { laneIndex } = this._resolveActiveLanePosition(reference);
        if (laneIndex < 0 || !Number.isFinite(reference.x) || !Number.isFinite(reference.y) || !Number.isFinite(reference.z)) {
            this._clearAutoCameraOffsets();
            return false;
        }

        let followRef = reference;
        if (this._autoCameraSmoothAlpha > 0) {
            if (!this._autoCameraSmoothValid) {
                this._autoCameraSmoothedRef.copy(reference);
                this._autoCameraSmoothValid = true;
            } else {
                this._autoCameraSmoothedRef.lerp(reference, this._autoCameraSmoothAlpha);
            }
            followRef = this._autoCameraSmoothedRef;
        }

        if (!this._hasAutoCameraOffsets) {
            this._captureAutoCameraOffsets(followRef);
        }

        this._applyAutoCamera(followRef);
        return true;
    }
}
