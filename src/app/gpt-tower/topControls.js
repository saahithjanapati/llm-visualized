export function initFollowModeControls({ pipeline, appState, followModeBtn, followSettingsToggle }) {
    const updateFollowButton = (enabled) => {
        if (!followModeBtn) return;
        const isOn = !!enabled;
        followModeBtn.dataset.state = isOn ? 'enabled' : 'disabled';
        followModeBtn.setAttribute('aria-pressed', String(isOn));
        followModeBtn.textContent = isOn ? 'Follow mode on' : 'Enable Follow Mode';
        followModeBtn.setAttribute('aria-label', isOn ? 'Follow mode enabled' : 'Enable follow mode');
        followModeBtn.setAttribute('title', isOn ? 'Follow mode enabled' : 'Enable follow mode');
        followModeBtn.disabled = isOn;
    };

    const setFollowMode = (enabled, { resetView = false } = {}) => {
        const next = !!enabled;
        if (appState.autoCameraFollow === next && pipeline?.isAutoCameraFollowEnabled?.() === next) {
            updateFollowButton(next);
            if (followSettingsToggle) followSettingsToggle.checked = next;
            return;
        }
        appState.autoCameraFollow = next;
        pipeline?.setAutoCameraFollow?.(next, { immediate: next, resetView: next && resetView, smoothReset: next && resetView });
        updateFollowButton(next);
        if (followSettingsToggle) followSettingsToggle.checked = next;
    };

    let pendingFollowDisableRequest = null;
    const suppressPendingFollowDisable = () => {
        if (pendingFollowDisableRequest) {
            pendingFollowDisableRequest.canceled = true;
        }
    };

    const queueFollowDisableFromInteraction = () => {
        const request = { canceled: false };
        pendingFollowDisableRequest = request;
        const run = () => {
            if (pendingFollowDisableRequest !== request) return;
            pendingFollowDisableRequest = null;
            if (request.canceled) return;
            if (pipeline?.isAutoCameraFollowEnabled?.()) {
                setFollowMode(false);
            }
        };
        if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(run);
        } else {
            setTimeout(run, 0);
        }
    };

    if (followModeBtn) {
        followModeBtn.addEventListener('click', (event) => {
            event.preventDefault();
            setFollowMode(true, { resetView: true });
        });
    }

    if (followSettingsToggle) {
        followSettingsToggle.addEventListener('change', () => {
            updateFollowButton(!!followSettingsToggle.checked);
        });
    }

    if (pipeline?.engine?.controls?.addEventListener) {
        const controls = pipeline.engine.controls;
        const camera = pipeline.engine.camera || null;
        const FOLLOW_DISABLE_MOVE_EPSILON_SQ = 1e-4;
        let activeControlInteraction = null;

        controls.addEventListener('start', () => {
            if (!pipeline?.isAutoCameraFollowEnabled?.()) {
                activeControlInteraction = null;
                return;
            }
            if (!camera || !controls?.target) {
                activeControlInteraction = { moved: true };
                return;
            }
            activeControlInteraction = {
                moved: false,
                cameraStart: camera.position.clone(),
                targetStart: controls.target.clone()
            };
        });

        controls.addEventListener('change', () => {
            if (!activeControlInteraction || activeControlInteraction.moved) return;
            if (!camera || !controls?.target) {
                activeControlInteraction.moved = true;
                return;
            }
            const cameraMoveSq = camera.position.distanceToSquared(activeControlInteraction.cameraStart);
            const targetMoveSq = controls.target.distanceToSquared(activeControlInteraction.targetStart);
            if (cameraMoveSq > FOLLOW_DISABLE_MOVE_EPSILON_SQ
                || targetMoveSq > FOLLOW_DISABLE_MOVE_EPSILON_SQ) {
                activeControlInteraction.moved = true;
            }
        });

        controls.addEventListener('end', () => {
            const interaction = activeControlInteraction;
            activeControlInteraction = null;
            if (!interaction?.moved) return;
            queueFollowDisableFromInteraction();
        });
    }

    if (typeof window !== 'undefined') {
        window.addEventListener('autoCameraFollowRequest', (event) => {
            const enabled = !!event?.detail?.enabled;
            if (!enabled) {
                setFollowMode(false);
            }
        });
    }

    updateFollowButton(pipeline?.isAutoCameraFollowEnabled?.());

    return { setFollowMode, updateFollowButton, suppressPendingFollowDisable };
}

export function initTopControlsAutohide({ topControls, settingsOverlay }) {
    const isSkinnyScreen = () => window.matchMedia('(max-aspect-ratio: 1/1), (max-width: 880px)').matches;
    const isTouchUi = () => window.matchMedia('(hover: none) and (pointer: coarse)').matches;
    let topControlsHideTimer = null;
    const autoHideDelayMs = () => (isTouchUi() ? 9000 : 5000);

    const isTopControlsVisible = () => {
        if (!topControls || typeof window === 'undefined') return false;
        const style = window.getComputedStyle(topControls);
        if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
        if (typeof document !== 'undefined' && document.body?.classList?.contains('detail-mobile-focus')) return false;
        const autoHidden = topControls.dataset.autoHidden === 'true';
        if (autoHidden) {
            const opacity = Number.parseFloat(style.opacity || '1');
            if (Number.isFinite(opacity) && opacity < 0.15) return false;
        }
        return true;
    };

    const findTopControlButtonAt = (x, y) => {
        if (!topControls) return null;
        const buttons = topControls.querySelectorAll('button');
        for (let i = buttons.length - 1; i >= 0; i -= 1) {
            const button = buttons[i];
            if (!button || button.disabled) continue;
            const rect = button.getBoundingClientRect();
            if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
                return button;
            }
        }
        return null;
    };

    const resolveTopControlButton = (event) => {
        if (!topControls || !event) return null;
        const targetButton = event.target instanceof Element ? event.target.closest('button') : null;
        if (targetButton && topControls.contains(targetButton)) return targetButton;
        if (!Number.isFinite(event.clientX) || !Number.isFinite(event.clientY)) return null;
        return findTopControlButtonAt(event.clientX, event.clientY);
    };

    const showTopControls = () => {
        if (!topControls) return;
        topControls.removeAttribute('data-auto-hidden');
        if (isSkinnyScreen()) {
            clearTimeout(topControlsHideTimer);
            topControlsHideTimer = setTimeout(() => {
                if (isSkinnyScreen()) {
                    topControls.setAttribute('data-auto-hidden', 'true');
                }
            }, autoHideDelayMs());
        }
    };

    const handleViewportChange = () => {
        if (!topControls) return;
        if (isSkinnyScreen()) {
            showTopControls();
        } else {
            clearTimeout(topControlsHideTimer);
            topControls.removeAttribute('data-auto-hidden');
        }
    };

    handleViewportChange();
    window.addEventListener('resize', handleViewportChange);

    const clearActiveTextSelection = () => {
        if (typeof window === 'undefined' || typeof window.getSelection !== 'function') return false;
        const selection = window.getSelection();
        if (!selection || selection.isCollapsed) return false;
        try {
            selection.removeAllRanges();
            return true;
        } catch (_) {
            return false;
        }
    };

    const isTouchPointerEvent = (event) => {
        if (!event) return false;
        if (event.pointerType) return event.pointerType === 'touch';
        return isTouchUi();
    };

    let pendingTopControlClick = null;
    let activeTopControlPointer = null;
    const TOP_CONTROL_TAP_SLOP_PX = 16;

    const registerPendingTopControlClick = (button) => {
        pendingTopControlClick = { button, until: Date.now() + 500 };
    };

    const triggerTopControlClick = (button, event) => {
        if (!button || button.disabled) return false;
        clearActiveTextSelection();
        registerPendingTopControlClick(button);
        if (event?.cancelable) event.preventDefault();
        if (event?.stopPropagation) event.stopPropagation();
        if (typeof button.click === 'function') {
            button.click();
            return true;
        }
        return false;
    };

    const onTopControlsPointerDown = (event) => {
        if (!topControls) return;
        if (!isTouchPointerEvent(event)) return;
        const target = resolveTopControlButton(event);
        if (!target || target.disabled) return;
        activeTopControlPointer = {
            id: Number.isFinite(event.pointerId) ? event.pointerId : null,
            button: target,
            startX: Number.isFinite(event.clientX) ? event.clientX : 0,
            startY: Number.isFinite(event.clientY) ? event.clientY : 0,
            moved: false
        };
        if (clearActiveTextSelection() && event.cancelable) {
            event.preventDefault();
        }
    };

    const onTopControlsPointerMove = (event) => {
        if (!activeTopControlPointer) return;
        if (activeTopControlPointer.id !== null && event.pointerId !== activeTopControlPointer.id) return;
        const dx = (Number.isFinite(event.clientX) ? event.clientX : 0) - activeTopControlPointer.startX;
        const dy = (Number.isFinite(event.clientY) ? event.clientY : 0) - activeTopControlPointer.startY;
        if (dx * dx + dy * dy > TOP_CONTROL_TAP_SLOP_PX * TOP_CONTROL_TAP_SLOP_PX) {
            activeTopControlPointer.moved = true;
        }
    };

    const onTopControlsPointerUp = (event) => {
        if (!activeTopControlPointer) return;
        if (activeTopControlPointer.id !== null && event.pointerId !== activeTopControlPointer.id) return;
        const { button, moved } = activeTopControlPointer;
        activeTopControlPointer = null;
        if (moved) return;
        const resolved = resolveTopControlButton(event) || button;
        if (!resolved || resolved.disabled) return;
        triggerTopControlClick(resolved, event);
    };

    const onTopControlsPointerCancel = (event) => {
        if (!activeTopControlPointer) return;
        if (activeTopControlPointer.id !== null && event.pointerId !== activeTopControlPointer.id) return;
        activeTopControlPointer = null;
    };

    const onTopControlsClick = (event) => {
        if (!pendingTopControlClick) return;
        if (!event.isTrusted) return;
        if (Date.now() > pendingTopControlClick.until) {
            pendingTopControlClick = null;
            return;
        }
        const target = event.target instanceof Element ? event.target.closest('button') : null;
        if (!target || target !== pendingTopControlClick.button) return;
        pendingTopControlClick = null;
        event.preventDefault();
        event.stopPropagation();
    };

    if (topControls) {
        topControls.addEventListener('pointerdown', onTopControlsPointerDown, { capture: true });
        topControls.addEventListener('pointermove', onTopControlsPointerMove, { capture: true });
        topControls.addEventListener('pointerup', onTopControlsPointerUp, { capture: true });
        topControls.addEventListener('pointercancel', onTopControlsPointerCancel, { capture: true });
        topControls.addEventListener('click', onTopControlsClick, { capture: true });
    }

    document.addEventListener('pointerdown', (event) => {
        if (isTouchPointerEvent(event)) {
            clearActiveTextSelection();
        }
        if (!topControls) return;
        if (topControls.contains(event.target)) return;
        if (settingsOverlay && settingsOverlay.getAttribute('aria-hidden') === 'false') return;
        if (!Number.isFinite(event.clientX) || !Number.isFinite(event.clientY)) return;
        if (!isTopControlsVisible()) return;
        if (event.pointerType !== 'touch' && event.button !== 0) return;
        const rect = topControls.getBoundingClientRect();
        if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) {
            return;
        }
        const button = findTopControlButtonAt(event.clientX, event.clientY);
        if (!button) return;
        showTopControls();
        triggerTopControlClick(button, event);
    }, { capture: true });

    window.addEventListener('pointerdown', (event) => {
        if (!topControls) return;
        // On touch devices, only reveal via the tap-selection path (void taps),
        // not immediately on gesture start (pan/zoom/orbit).
        if (isTouchPointerEvent(event)) return;
        const wasHidden = topControls.dataset.autoHidden === 'true';
        const hasPoint = wasHidden && typeof event.clientX === 'number' && typeof event.clientY === 'number';
        const tapPoint = hasPoint ? { x: event.clientX, y: event.clientY } : null;
        let preButton = null;
        let insideBefore = false;
        if (tapPoint) {
            try {
                const rect = topControls.getBoundingClientRect();
                insideBefore = tapPoint.x >= rect.left && tapPoint.x <= rect.right
                    && tapPoint.y >= rect.top && tapPoint.y <= rect.bottom;
                if (insideBefore) {
                    preButton = findTopControlButtonAt(tapPoint.x, tapPoint.y);
                }
            } catch (_) { /* no-op */ }
        }
        showTopControls();
        if (wasHidden && tapPoint) {
            requestAnimationFrame(() => {
                let button = preButton;
                if (!button) {
                    try {
                        const rect = topControls.getBoundingClientRect();
                        const insideAfter = tapPoint.x >= rect.left && tapPoint.x <= rect.right
                            && tapPoint.y >= rect.top && tapPoint.y <= rect.bottom;
                        if (insideAfter) {
                            button = findTopControlButtonAt(tapPoint.x, tapPoint.y);
                        }
                    } catch (_) { /* no-op */ }
                }
                if (button && typeof button.click === 'function') {
                    triggerTopControlClick(button, event);
                }
            });
        }
    }, { passive: true });

    return { showTopControls, isTopControlsVisible };
}
