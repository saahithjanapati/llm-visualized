const DEFAULT_TAP_SLOP_PX = 16;
const DEFAULT_PENDING_MS = 500;

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

export function initTouchClickFallback(container, { selector = 'button', tapSlopPx = DEFAULT_TAP_SLOP_PX } = {}) {
    if (!container) return () => {};

    let active = null;
    let pendingClick = null;

    const registerPendingClick = (target) => {
        pendingClick = { target, until: Date.now() + DEFAULT_PENDING_MS };
    };

    const onPointerDown = (event) => {
        if (!isTouchLikeEvent(event)) return;
        const target = resolveClosestTarget(container, event, selector);
        if (!target || isElementDisabled(target)) return;
        const startX = Number.isFinite(event.clientX) ? event.clientX : 0;
        const startY = Number.isFinite(event.clientY) ? event.clientY : 0;
        active = {
            id: Number.isFinite(event.pointerId) ? event.pointerId : null,
            target,
            startX,
            startY,
            moved: false
        };
        if (event.cancelable) event.preventDefault();
    };

    const onPointerMove = (event) => {
        if (!active) return;
        if (active.id !== null && event.pointerId !== active.id) return;
        const dx = (Number.isFinite(event.clientX) ? event.clientX : 0) - active.startX;
        const dy = (Number.isFinite(event.clientY) ? event.clientY : 0) - active.startY;
        if (dx * dx + dy * dy > tapSlopPx * tapSlopPx) {
            active.moved = true;
        }
    };

    const onPointerUp = (event) => {
        if (!active) return;
        if (active.id !== null && event.pointerId !== active.id) return;
        const { target, moved } = active;
        active = null;
        if (moved || !target || isElementDisabled(target)) return;
        registerPendingClick(target);
        if (event.cancelable) event.preventDefault();
        if (typeof target.click === 'function') {
            target.click();
        }
    };

    const onPointerCancel = (event) => {
        if (!active) return;
        if (active.id !== null && event.pointerId !== active.id) return;
        active = null;
    };

    const onClick = (event) => {
        if (!pendingClick || !event.isTrusted) return;
        if (Date.now() > pendingClick.until) {
            pendingClick = null;
            return;
        }
        const target = resolveClosestTarget(container, event, selector);
        if (!target || target !== pendingClick.target) return;
        pendingClick = null;
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
