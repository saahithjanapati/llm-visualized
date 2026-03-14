export const SMALL_SCREEN_VIEWPORT_MEDIA_QUERY = '(max-aspect-ratio: 1/1), (max-width: 880px)';
const SMALL_SCREEN_VIEWPORT_MAX_WIDTH = 880;

export function isSmallScreenViewport({ win = (typeof window !== 'undefined' ? window : null) } = {}) {
    if (!win) return false;
    if (typeof win.matchMedia === 'function') {
        return win.matchMedia(SMALL_SCREEN_VIEWPORT_MEDIA_QUERY).matches;
    }

    const width = Number.isFinite(win.innerWidth) ? win.innerWidth : 0;
    const height = Number.isFinite(win.innerHeight) ? win.innerHeight : 0;
    if (width <= 0 || height <= 0) return false;
    return width <= SMALL_SCREEN_VIEWPORT_MAX_WIDTH || height >= width;
}

export function resolveCameraMaxDistance({
    baseMaxDistance,
    isSmallScreen = false,
    isLargeDesktopViewport = false,
    smallScreenZoomOutMultiplier = 1.0,
    desktopZoomOutMultiplier = 1.45
} = {}) {
    if (!(typeof baseMaxDistance === 'number' && Number.isFinite(baseMaxDistance) && baseMaxDistance > 0)) {
        return null;
    }

    if (isSmallScreen) {
        return baseMaxDistance * ((
            typeof smallScreenZoomOutMultiplier === 'number'
            && Number.isFinite(smallScreenZoomOutMultiplier)
            && smallScreenZoomOutMultiplier > 0
        ) ? smallScreenZoomOutMultiplier : 1.0);
    }

    if (isLargeDesktopViewport) {
        return baseMaxDistance * ((
            typeof desktopZoomOutMultiplier === 'number'
            && Number.isFinite(desktopZoomOutMultiplier)
            && desktopZoomOutMultiplier > 0
        ) ? desktopZoomOutMultiplier : 1.0);
    }

    return baseMaxDistance;
}
