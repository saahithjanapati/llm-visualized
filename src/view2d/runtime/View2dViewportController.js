function clampPositive(value, fallback = 0) {
    return Number.isFinite(value) && value > 0 ? value : fallback;
}

function clampScale(value, minScale, maxScale) {
    const safeMin = clampPositive(minScale, 0.01);
    const safeMax = Math.max(safeMin, clampPositive(maxScale, safeMin));
    const safeValue = clampPositive(value, safeMin);
    return Math.max(safeMin, Math.min(safeMax, safeValue));
}

function cloneBounds(bounds = null) {
    if (!bounds || typeof bounds !== 'object') return null;
    return {
        x: Number.isFinite(bounds.x) ? bounds.x : 0,
        y: Number.isFinite(bounds.y) ? bounds.y : 0,
        width: Number.isFinite(bounds.width) ? bounds.width : 0,
        height: Number.isFinite(bounds.height) ? bounds.height : 0
    };
}

function resolvePadding(padding = 24) {
    if (Number.isFinite(padding)) {
        const safe = Math.max(0, Math.floor(padding));
        return {
            top: safe,
            right: safe,
            bottom: safe,
            left: safe
        };
    }
    if (!padding || typeof padding !== 'object') {
        return resolvePadding(24);
    }
    return {
        top: Math.max(0, Math.floor(padding.top ?? padding.y ?? 24)),
        right: Math.max(0, Math.floor(padding.right ?? padding.x ?? 24)),
        bottom: Math.max(0, Math.floor(padding.bottom ?? padding.y ?? 24)),
        left: Math.max(0, Math.floor(padding.left ?? padding.x ?? 24))
    };
}

function easeInOutCubic(t) {
    const safe = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
    return safe < 0.5
        ? 4 * safe * safe * safe
        : 1 - (Math.pow(-2 * safe + 2, 3) / 2);
}

function interpolate(start, end, alpha) {
    return start + ((end - start) * alpha);
}

function resolveFitTransform(bounds, viewport, {
    minScale = 0.05,
    maxScale = 4,
    padding = 24
} = {}) {
    const safeBounds = cloneBounds(bounds);
    const width = clampPositive(viewport?.width, 0);
    const height = clampPositive(viewport?.height, 0);
    if (!safeBounds || !(safeBounds.width > 0) || !(safeBounds.height > 0) || !(width > 0) || !(height > 0)) {
        return null;
    }

    const resolvedPadding = resolvePadding(padding);
    const availableWidth = Math.max(1, width - resolvedPadding.left - resolvedPadding.right);
    const availableHeight = Math.max(1, height - resolvedPadding.top - resolvedPadding.bottom);
    const scale = clampScale(
        Math.min(
            availableWidth / Math.max(1, safeBounds.width),
            availableHeight / Math.max(1, safeBounds.height)
        ),
        minScale,
        maxScale
    );
    const fittedWidth = safeBounds.width * scale;
    const fittedHeight = safeBounds.height * scale;

    return {
        scale,
        panX: resolvedPadding.left + ((availableWidth - fittedWidth) * 0.5) - (safeBounds.x * scale),
        panY: resolvedPadding.top + ((availableHeight - fittedHeight) * 0.5) - (safeBounds.y * scale),
        padding: resolvedPadding
    };
}

export class View2dViewportController {
    constructor({
        minScale = 0.05,
        maxScale = 4,
        padding = 24
    } = {}) {
        this.minScale = clampPositive(minScale, 0.05);
        this.maxScale = Math.max(this.minScale, clampPositive(maxScale, this.minScale));
        this.padding = resolvePadding(padding);
        this.viewport = {
            width: 0,
            height: 0
        };
        this.sceneBounds = null;
        this.state = {
            scale: 1,
            panX: 0,
            panY: 0
        };
        this.animation = null;
    }

    setViewportSize(width = 0, height = 0) {
        this.viewport.width = clampPositive(width, 0);
        this.viewport.height = clampPositive(height, 0);
        return this.getState();
    }

    setSceneBounds(bounds = null) {
        this.sceneBounds = cloneBounds(bounds);
        return this.getState();
    }

    setState({
        scale = this.state.scale,
        panX = this.state.panX,
        panY = this.state.panY
    } = {}) {
        this.state = {
            scale: clampScale(scale, this.minScale, this.maxScale),
            panX: Number.isFinite(panX) ? panX : this.state.panX,
            panY: Number.isFinite(panY) ? panY : this.state.panY
        };
        return this.getState();
    }

    getState() {
        return {
            scale: this.state.scale,
            panX: this.state.panX,
            panY: this.state.panY,
            viewport: {
                ...this.viewport
            },
            sceneBounds: cloneBounds(this.sceneBounds)
        };
    }

    getViewportTransform(source = 'viewport-controller') {
        return {
            source,
            scale: this.state.scale,
            offsetX: this.state.panX,
            offsetY: this.state.panY
        };
    }

    fitToBounds(bounds = null, {
        padding = this.padding,
        minScale = this.minScale,
        maxScale = this.maxScale,
        source = 'fit-to-bounds'
    } = {}) {
        const transform = resolveFitTransform(bounds, this.viewport, {
            padding,
            minScale,
            maxScale
        });
        if (!transform) return this.getState();
        this.animation = null;
        return this.setState({
            scale: transform.scale,
            panX: transform.panX,
            panY: transform.panY
        });
    }

    fitScene(options = {}) {
        if (!this.sceneBounds) return this.getState();
        return this.fitToBounds(this.sceneBounds, {
            ...options,
            source: options?.source || 'fit-scene'
        });
    }

    flyToBounds(bounds = null, {
        animate = true,
        durationMs = 420,
        now = 0,
        padding = this.padding,
        minScale = this.minScale,
        maxScale = this.maxScale,
        source = 'fly-to-bounds'
    } = {}) {
        const target = resolveFitTransform(bounds, this.viewport, {
            padding,
            minScale,
            maxScale
        });
        if (!target) return this.getState();
        if (!animate || durationMs <= 0) {
            return this.fitToBounds(bounds, {
                padding,
                minScale,
                maxScale,
                source
            });
        }

        this.animation = {
            source,
            startTime: Number.isFinite(now) ? now : 0,
            durationMs: Math.max(1, Math.floor(durationMs)),
            startState: {
                ...this.state
            },
            endState: {
                scale: target.scale,
                panX: target.panX,
                panY: target.panY
            }
        };
        return this.getState();
    }

    step(now = 0) {
        if (!this.animation) return this.getState();
        const elapsed = Math.max(0, (Number.isFinite(now) ? now : 0) - this.animation.startTime);
        const alpha = easeInOutCubic(elapsed / Math.max(1, this.animation.durationMs));
        this.state = {
            scale: interpolate(this.animation.startState.scale, this.animation.endState.scale, alpha),
            panX: interpolate(this.animation.startState.panX, this.animation.endState.panX, alpha),
            panY: interpolate(this.animation.startState.panY, this.animation.endState.panY, alpha)
        };
        if (elapsed >= this.animation.durationMs) {
            this.state = { ...this.animation.endState };
            this.animation = null;
        }
        return this.getState();
    }

    panBy(deltaX = 0, deltaY = 0) {
        this.animation = null;
        return this.setState({
            panX: this.state.panX + (Number.isFinite(deltaX) ? deltaX : 0),
            panY: this.state.panY + (Number.isFinite(deltaY) ? deltaY : 0)
        });
    }

    zoomAt(multiplier = 1, anchorX = null, anchorY = null) {
        const currentScale = this.state.scale;
        const safeMultiplier = clampPositive(multiplier, 1);
        const nextScale = clampScale(currentScale * safeMultiplier, this.minScale, this.maxScale);
        const viewportWidth = clampPositive(this.viewport.width, 0);
        const viewportHeight = clampPositive(this.viewport.height, 0);
        const pointerX = Number.isFinite(anchorX) ? anchorX : (viewportWidth * 0.5);
        const pointerY = Number.isFinite(anchorY) ? anchorY : (viewportHeight * 0.5);
        const localX = (pointerX - this.state.panX) / Math.max(1e-6, currentScale);
        const localY = (pointerY - this.state.panY) / Math.max(1e-6, currentScale);

        this.animation = null;
        return this.setState({
            scale: nextScale,
            panX: pointerX - (localX * nextScale),
            panY: pointerY - (localY * nextScale)
        });
    }

    screenToWorld(x = 0, y = 0) {
        const scale = Math.max(1e-6, this.state.scale);
        return {
            x: (x - this.state.panX) / scale,
            y: (y - this.state.panY) / scale
        };
    }

    worldToScreen(x = 0, y = 0) {
        return {
            x: (x * this.state.scale) + this.state.panX,
            y: (y * this.state.scale) + this.state.panY
        };
    }
}

export function resolveViewportFitTransform(bounds, viewport, options = {}) {
    return resolveFitTransform(bounds, viewport, options);
}
