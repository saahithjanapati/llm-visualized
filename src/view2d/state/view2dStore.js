function cloneState(state = {}) {
    return {
        scene: state.scene || null,
        viewport: {
            zoom: Number.isFinite(state.viewport?.zoom) ? state.viewport.zoom : 1,
            panX: Number.isFinite(state.viewport?.panX) ? state.viewport.panX : 0,
            panY: Number.isFinite(state.viewport?.panY) ? state.viewport.panY : 0,
            width: Number.isFinite(state.viewport?.width) ? state.viewport.width : 0,
            height: Number.isFinite(state.viewport?.height) ? state.viewport.height : 0,
            devicePixelRatio: Number.isFinite(state.viewport?.devicePixelRatio)
                ? state.viewport.devicePixelRatio
                : 1
        },
        interaction: {
            hoverId: typeof state.interaction?.hoverId === 'string' ? state.interaction.hoverId : null,
            selectedId: typeof state.interaction?.selectedId === 'string' ? state.interaction.selectedId : null,
            focusPath: Array.isArray(state.interaction?.focusPath) ? [...state.interaction.focusPath] : []
        },
        flags: {
            dirty: !!state.flags?.dirty,
            isReady: !!state.flags?.isReady
        }
    };
}

export function createView2dStore(initialState = {}) {
    let state = cloneState(initialState);
    const subscribers = new Set();

    const emit = () => {
        subscribers.forEach((subscriber) => {
            try {
                subscriber(state);
            } catch (error) {
                console.warn('view2dStore subscriber failed:', error);
            }
        });
    };

    const setState = (updater) => {
        const nextState = typeof updater === 'function'
            ? updater(state)
            : updater;
        state = cloneState(nextState);
        emit();
        return state;
    };

    return {
        getState() {
            return state;
        },
        setState,
        setScene(scene) {
            return setState((current) => ({
                ...current,
                scene: scene || null,
                flags: {
                    ...current.flags,
                    dirty: true,
                    isReady: !!scene
                }
            }));
        },
        patchViewport(patch = {}) {
            return setState((current) => ({
                ...current,
                viewport: {
                    ...current.viewport,
                    ...patch
                }
            }));
        },
        setHoverTarget(nodeId = null) {
            return setState((current) => ({
                ...current,
                interaction: {
                    ...current.interaction,
                    hoverId: typeof nodeId === 'string' ? nodeId : null
                }
            }));
        },
        setSelection(nodeId = null, focusPath = null) {
            return setState((current) => ({
                ...current,
                interaction: {
                    ...current.interaction,
                    selectedId: typeof nodeId === 'string' ? nodeId : null,
                    focusPath: Array.isArray(focusPath) ? [...focusPath] : current.interaction.focusPath
                }
            }));
        },
        subscribe(listener) {
            if (typeof listener !== 'function') {
                return () => {};
            }
            subscribers.add(listener);
            return () => {
                subscribers.delete(listener);
            };
        }
    };
}
