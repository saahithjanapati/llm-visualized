import { appState } from '../state/appState.js';

const sceneInteractionLockStateByEngine = new WeakMap();

let modalUiLockState = null;

function snapshotControlsState(controls = null) {
    if (!controls) return null;
    return {
        enabled: controls.enabled,
        enableRotate: controls.enableRotate,
        enablePan: controls.enablePan,
        enableZoom: controls.enableZoom
    };
}

function applyControlsDisabledState(controls = null) {
    if (!controls) return;
    controls.enabled = false;
    controls.enableRotate = false;
    controls.enablePan = false;
    controls.enableZoom = false;
}

function restoreControlsState(controls = null, snapshot = null) {
    if (!controls || !snapshot) return;
    controls.enabled = snapshot.enabled;
    controls.enableRotate = snapshot.enableRotate;
    controls.enablePan = snapshot.enablePan;
    controls.enableZoom = snapshot.enableZoom;
}

function resolveCanvasPointerTarget(engine = null, fallbackCanvas = null) {
    return engine?.renderer?.domElement || fallbackCanvas || null;
}

export function acquireModalUiLock() {
    const body = typeof document !== 'undefined' ? document.body : null;

    if (!modalUiLockState || modalUiLockState.count === 0) {
        modalUiLockState = {
            count: 0,
            previousBodyOverflow: body?.style?.overflow || '',
            previousModalPaused: !!appState.modalPaused
        };
    }

    modalUiLockState.count += 1;
    if (body) {
        body.style.overflow = 'hidden';
    }
    appState.modalPaused = true;

    let released = false;
    return () => {
        if (released || !modalUiLockState) return false;
        released = true;
        modalUiLockState.count = Math.max(0, modalUiLockState.count - 1);
        if (modalUiLockState.count > 0) {
            return true;
        }

        const previousBodyOverflow = modalUiLockState.previousBodyOverflow;
        const previousModalPaused = modalUiLockState.previousModalPaused;
        modalUiLockState = null;

        if (body) {
            body.style.overflow = previousBodyOverflow;
        }
        appState.modalPaused = previousModalPaused;
        return true;
    };
}

export function acquireSceneBackgroundInteractionLock(engine = null) {
    if (!engine) {
        return () => false;
    }

    let state = sceneInteractionLockStateByEngine.get(engine);
    if (!state) {
        state = {
            count: 0,
            controlsSnapshot: null,
            canvas: null,
            previousCanvasPointerEvents: ''
        };
        sceneInteractionLockStateByEngine.set(engine, state);
    }

    if (state.count === 0) {
        const controls = engine.controls || null;
        const canvas = resolveCanvasPointerTarget(engine);
        state.controlsSnapshot = snapshotControlsState(controls);
        state.canvas = canvas;
        state.previousCanvasPointerEvents = canvas?.style?.pointerEvents || '';
    }

    state.count += 1;

    applyControlsDisabledState(engine.controls || null);
    const canvas = resolveCanvasPointerTarget(engine, state.canvas);
    if (canvas) {
        canvas.style.pointerEvents = 'none';
    }
    engine.resetInteractionState?.();

    let released = false;
    return () => {
        if (released) return false;
        released = true;

        const activeState = sceneInteractionLockStateByEngine.get(engine);
        if (!activeState) return false;

        activeState.count = Math.max(0, activeState.count - 1);
        if (activeState.count > 0) {
            engine.resetInteractionState?.();
            return true;
        }

        restoreControlsState(engine.controls || null, activeState.controlsSnapshot);
        const restoreCanvas = resolveCanvasPointerTarget(engine, activeState.canvas);
        if (restoreCanvas) {
            restoreCanvas.style.pointerEvents = activeState.previousCanvasPointerEvents;
        }
        activeState.controlsSnapshot = null;
        activeState.canvas = null;
        activeState.previousCanvasPointerEvents = '';
        engine.resetInteractionState?.();
        return true;
    };
}
