const DEFAULT_TAP_SLOP_PX = 16;
const DEFAULT_PENDING_MS = 1200;
const DEFAULT_RETARGET_SUPPRESSION_MS = 420;
const DEFAULT_RETARGET_SLOP_PX = 28;
const MAX_PENDING_CLICKS = 24;

const isTouchLikeEvent = (event) => {
    if (!event) return false;
    if (event.pointerType) return event.pointerType === 'touch' || event.pointerType === 'pen';
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
        return window.matchMedia('(hover: none) and (pointer: coarse)').matches;
    }
    return false;
};

const resolveClosestTarget = (container, event, selector) => {
    if (!container || !event || !event.target) return null;
    const el = event.target instanceof Element ? event.target.closest(selector) : null;
    if (!el || !container.contains(el)) return null;
    return el;
};

const isElementDisabled = (el) => {
    if (!el) return true;
    if (el.getAttribute && el.getAttribute('aria-disabled') === 'true') return true;
    if (el.hasAttribute && el.hasAttribute('disabled')) return true;
    if ('disabled' in el && el.disabled) return true;
    const input = el.querySelector && el.querySelector('input');
    if (input && input.disabled) return true;
    return false;
};

const resolveActivationTarget = (target) => {
    if (!target) return null;
    const tag = target.tagName?.toUpperCase?.() || '';
    if (tag === 'LABEL') {
        const control = target.control || target.querySelector?.('input, button, select, textarea');
        if (control && typeof control.click === 'function' && !isElementDisabled(control)) {
            return control;
        }
    }
    return target;
};

export function initTouchClickFallback(
    container,
    {
        selector = 'button',
        tapSlopPx = DEFAULT_TAP_SLOP_PX,
        activateOnPointerDownSelector = null
    } = {}
) {
    if (!container) return () => {};

    let active = null;
    const pendingClicks = [];
    const shouldActivateOnPointerDown = (target) => (
        !!activateOnPointerDownSelector
        && !!target
        && typeof target.matches === 'function'
        && target.matches(activateOnPointerDownSelector)
    );

    const getPointerId = (event) => (Number.isFinite(event?.pointerId) ? event.pointerId : null);

    const matchesActivePointer = (event) => {
        if (!active) return false;
        if (active.id === null) return true;
        const pointerId = getPointerId(event);
        return pointerId === null || pointerId === active.id;
    };

    const prunePendingClicks = () => {
        const now = Date.now();
        for (let idx = pendingClicks.length - 1; idx >= 0; idx -= 1) {
            const pending = pendingClicks[idx];
            if (!pending || !pending.target?.isConnected || now > pending.until) {
                pendingClicks.splice(idx, 1);
            }
        }
    };

    const registerPendingClick = (target, {
        clientX = null,
        clientY = null
    } = {}) => {
        prunePendingClicks();
        pendingClicks.push({
            target,
            until: Date.now() + DEFAULT_PENDING_MS,
            activatedAt: Date.now(),
            clientX: Number.isFinite(clientX) ? clientX : null,
            clientY: Number.isFinite(clientY) ? clientY : null
        });
        if (pendingClicks.length > MAX_PENDING_CLICKS) {
            pendingClicks.splice(0, pendingClicks.length - MAX_PENDING_CLICKS);
        }
    };

    const findRetargetedPendingClickIndex = (target, event) => {
        const clickX = Number.isFinite(event?.clientX) ? event.clientX : null;
        const clickY = Number.isFinite(event?.clientY) ? event.clientY : null;
        if (!Number.isFinite(clickX) || !Number.isFinite(clickY)) return -1;
        const now = Date.now();
        for (let idx = pendingClicks.length - 1; idx >= 0; idx -= 1) {
            const pending = pendingClicks[idx];
            if (!pending || pending.target === target) continue;
            if (now - pending.activatedAt > DEFAULT_RETARGET_SUPPRESSION_MS) continue;
            if (!Number.isFinite(pending.clientX) || !Number.isFinite(pending.clientY)) continue;
            const dx = clickX - pending.clientX;
            const dy = clickY - pending.clientY;
            if (dx * dx + dy * dy <= DEFAULT_RETARGET_SLOP_PX * DEFAULT_RETARGET_SLOP_PX) {
                return idx;
            }
        }
        return -1;
    };

    const onPointerDown = (event) => {
        if (!isTouchLikeEvent(event)) return;
        const target = resolveClosestTarget(container, event, selector);
        if (!target || isElementDisabled(target)) return;
        if (shouldActivateOnPointerDown(target)) {
            active = null;
            registerPendingClick(target, {
                clientX: event.clientX,
                clientY: event.clientY
            });
            const activationTarget = resolveActivationTarget(target);
            if (event.cancelable) event.preventDefault();
            event.stopPropagation();
            if (activationTarget && typeof activationTarget.click === 'function') {
                activationTarget.click();
            }
            return;
        }
        const startX = Number.isFinite(event.clientX) ? event.clientX : 0;
        const startY = Number.isFinite(event.clientY) ? event.clientY : 0;
        active = {
            id: getPointerId(event),
            target,
            startX,
            startY,
            moved: false
        };
    };

    const onPointerMove = (event) => {
        if (!matchesActivePointer(event)) return;
        const dx = (Number.isFinite(event.clientX) ? event.clientX : 0) - active.startX;
        const dy = (Number.isFinite(event.clientY) ? event.clientY : 0) - active.startY;
        if (dx * dx + dy * dy > tapSlopPx * tapSlopPx) {
            active.moved = true;
        }
    };

    const onPointerUp = (event) => {
        if (!matchesActivePointer(event)) return;
        const { target, moved } = active;
        active = null;
        if (moved || !target || isElementDisabled(target)) return;
        registerPendingClick(target, {
            clientX: event.clientX,
            clientY: event.clientY
        });
        const activationTarget = resolveActivationTarget(target);
        if (activationTarget && typeof activationTarget.click === 'function') {
            activationTarget.click();
        }
    };

    const onPointerCancel = (event) => {
        if (!matchesActivePointer(event)) return;
        active = null;
    };

    const onClick = (event) => {
        if (!event.isTrusted) return;
        prunePendingClicks();
        if (!pendingClicks.length) return;
        const target = resolveClosestTarget(container, event, selector);
        if (!target) return;
        let pendingIndex = pendingClicks.findIndex((pending) => pending.target === target);
        if (pendingIndex < 0) {
            pendingIndex = findRetargetedPendingClickIndex(target, event);
        }
        if (pendingIndex < 0) return;
        pendingClicks.splice(pendingIndex, 1);
        event.preventDefault();
        event.stopPropagation();
    };

    container.addEventListener('pointerdown', onPointerDown, { capture: true });
    container.addEventListener('pointermove', onPointerMove, { capture: true });
    container.addEventListener('pointerup', onPointerUp, { capture: true });
    container.addEventListener('pointercancel', onPointerCancel, { capture: true });
    container.addEventListener('click', onClick, { capture: true });

    return () => {
        container.removeEventListener('pointerdown', onPointerDown, { capture: true });
        container.removeEventListener('pointermove', onPointerMove, { capture: true });
        container.removeEventListener('pointerup', onPointerUp, { capture: true });
        container.removeEventListener('pointercancel', onPointerCancel, { capture: true });
        container.removeEventListener('click', onClick, { capture: true });
    };
}
