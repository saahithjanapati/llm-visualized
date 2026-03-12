export function resolveSemanticTargetEntry(registry, target, options = {}) {
    if (!registry || typeof registry.resolveEntryForSemanticTarget !== 'function') return null;
    return registry.resolveEntryForSemanticTarget(target, options);
}

export function resolveSemanticTargetBounds(registry, target, options = {}) {
    if (!registry || typeof registry.resolveBoundsForSemanticTarget !== 'function') return null;
    return registry.resolveBoundsForSemanticTarget(target, options);
}

export function resolveSemanticTargetAnchors(registry, target, options = {}) {
    if (!registry || typeof registry.resolveAnchorsForSemanticTarget !== 'function') return null;
    return registry.resolveAnchorsForSemanticTarget(target, options);
}

export function resolveSemanticTargetFocusPath(registry, target, options = {}) {
    if (!registry || typeof registry.resolveFocusPathForSemanticTarget !== 'function') return [];
    return registry.resolveFocusPathForSemanticTarget(target, options);
}
