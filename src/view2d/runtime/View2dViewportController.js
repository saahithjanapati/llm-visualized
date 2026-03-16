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

function resolveViewportInsets(viewportInsets = null) {
    if (Number.isFinite(viewportInsets)) {
        const safe = Math.max(0, Math.floor(viewportInsets));
        return {
            top: safe,
            right: safe,
            bottom: safe,
            left: safe
        };
    }
    if (!viewportInsets || typeof viewportInsets !== 'object') {
        return {
            top: 0,
            right: 0,
            bottom: 0,
            left: 0
        };
    }
    return {
        top: Math.max(0, Math.floor(viewportInsets.top ?? viewportInsets.y ?? 0)),
        right: Math.max(0, Math.floor(viewportInsets.right ?? viewportInsets.x ?? 0)),
        bottom: Math.max(0, Math.floor(viewportInsets.bottom ?? viewportInsets.y ?? 0)),
        left: Math.max(0, Math.floor(viewportInsets.left ?? viewportInsets.x ?? 0))
    };
}

function resolveViewportGeometry(viewport = null, viewportInsets = null) {
    const width = clampPositive(viewport?.width, 0);
    const height = clampPositive(viewport?.height, 0);
    const requestedInsets = resolveViewportInsets(viewportInsets);
    const left = Math.min(requestedInsets.left, width);
    const right = Math.min(requestedInsets.right, Math.max(0, width - left));
    const top = Math.min(requestedInsets.top, height);
    const bottom = Math.min(requestedInsets.bottom, Math.max(0, height - top));
    const availableWidth = Math.max(0, width - left - right);
    const availableHeight = Math.max(0, height - top - bottom);

    return {
        width,
        height,
        x: left,
        y: top,
        availableWidth,
        availableHeight,
        centerX: left + (availableWidth * 0.5),
        centerY: top + (availableHeight * 0.5),
        insets: {
            top,
            right,
            bottom,
            left
        }
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
    padding = 24,
    viewportInsets = null
} = {}) {
    const safeBounds = cloneBounds(bounds);
    const geometry = resolveViewportGeometry(viewport, viewportInsets);
    const width = geometry.width;
    const height = geometry.height;
    if (!safeBounds || !(safeBounds.width > 0) || !(safeBounds.height > 0) || !(width > 0) || !(height > 0)) {
        return null;
    }

    const resolvedPadding = resolvePadding(padding);
    const availableWidth = Math.max(
        1,
        geometry.availableWidth - resolvedPadding.left - resolvedPadding.right
    );
    const availableHeight = Math.max(
        1,
        geometry.availableHeight - resolvedPadding.top - resolvedPadding.bottom
    );
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
        panX: geometry.x + resolvedPadding.left + ((availableWidth - fittedWidth) * 0.5) - (safeBounds.x * scale),
        panY: geometry.y + resolvedPadding.top + ((availableHeight - fittedHeight) * 0.5) - (safeBounds.y * scale),
        padding: resolvedPadding,
        viewportInsets: {
            ...geometry.insets
        }
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
        this.viewportInsets = resolveViewportInsets(0);
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
        this.viewportInsets = resolveViewportGeometry(this.viewport, this.viewportInsets).insets;
        return this.getState();
    }

    setViewportInsets(viewportInsets = null, {
        preserveVisibleCenter = false,
        animate = false,
        durationMs = 420,
        now = 0,
        source = 'viewport-insets'
    } = {}) {
        const previousGeometry = resolveViewportGeometry(this.viewport, this.viewportInsets);
        const nextGeometry = resolveViewportGeometry(this.viewport, viewportInsets);
        this.viewportInsets = nextGeometry.insets;
        if (!preserveVisibleCenter) {
            return this.getState();
        }

        const deltaX = nextGeometry.centerX - previousGeometry.centerX;
        const deltaY = nextGeometry.centerY - previousGeometry.centerY;
        if (Math.abs(deltaX) < 1e-6 && Math.abs(deltaY) < 1e-6) {
            return this.getState();
        }

        if (animate) {
            this.animation = {
                source,
                startTime: Number.isFinite(now) ? now : 0,
                durationMs: Math.max(1, Math.floor(durationMs)),
                startState: {
                    ...this.state
                },
                endState: {
                    ...this.state,
                    panX: this.state.panX + deltaX,
                    panY: this.state.panY + deltaY
                }
            };
            return this.getState();
        }

        this.state = {
            ...this.state,
            panX: this.state.panX + deltaX,
            panY: this.state.panY + deltaY
        };
        if (this.animation) {
            this.animation = {
                ...this.animation,
                startState: {
                    ...this.animation.startState,
                    panX: this.animation.startState.panX + deltaX,
                    panY: this.animation.startState.panY + deltaY
                },
                endState: {
                    ...this.animation.endState,
                    panX: this.animation.endState.panX + deltaX,
                    panY: this.animation.endState.panY + deltaY
                }
            };
        }
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
            viewportInsets: {
                ...this.viewportInsets
            },
            sceneBounds: cloneBounds(this.sceneBounds)
        };
    }

    getEffectiveViewportRect() {
        const geometry = resolveViewportGeometry(this.viewport, this.viewportInsets);
        return {
            x: geometry.x,
            y: geometry.y,
            width: geometry.availableWidth,
            height: geometry.availableHeight,
            viewportInsets: {
                ...geometry.insets
            }
        };
    }

    getViewportTransform(source = 'viewport-controller') {
        return {
            source,
            scale: this.state.scale,
            offsetX: this.state.panX,
            offsetY: this.state.panY,
            viewportInsets: {
                ...this.viewportInsets
            }
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
            maxScale,
            viewportInsets: this.viewportInsets
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
            maxScale,
            viewportInsets: this.viewportInsets
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
        const geometry = resolveViewportGeometry(this.viewport, this.viewportInsets);
        const pointerX = Number.isFinite(anchorX) ? anchorX : geometry.centerX;
        const pointerY = Number.isFinite(anchorY) ? anchorY : geometry.centerY;
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
