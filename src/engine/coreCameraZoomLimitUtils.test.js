import { describe, expect, it, vi } from 'vitest';

import {
    isSmallScreenViewport,
    resolveCameraMaxDistance,
    SMALL_SCREEN_VIEWPORT_MEDIA_QUERY
} from './coreCameraZoomLimitUtils.js';

describe('coreCameraZoomLimitUtils', () => {
    it('detects small screens via the shared media query', () => {
        const matchMedia = vi.fn((query) => ({
            matches: query === SMALL_SCREEN_VIEWPORT_MEDIA_QUERY
        }));

        expect(isSmallScreenViewport({
            win: {
                matchMedia,
                innerWidth: 1440,
                innerHeight: 900
            }
        })).toBe(true);
        expect(matchMedia).toHaveBeenCalledWith(SMALL_SCREEN_VIEWPORT_MEDIA_QUERY);
    });

    it('falls back to narrow-or-portrait viewport detection when matchMedia is unavailable', () => {
        expect(isSmallScreenViewport({
            win: {
                innerWidth: 480,
                innerHeight: 820
            }
        })).toBe(true);

        expect(isSmallScreenViewport({
            win: {
                innerWidth: 1440,
                innerHeight: 900
            }
        })).toBe(false);
    });

    it('applies the tighter zoom-out multiplier on small screens before desktop allowances', () => {
        expect(resolveCameraMaxDistance({
            baseMaxDistance: 1000,
            isSmallScreen: true,
            isLargeDesktopViewport: true,
            smallScreenZoomOutMultiplier: 0.9,
            desktopZoomOutMultiplier: 1.1
        })).toBe(900);
    });

    it('preserves the larger-screen zoom allowance when the viewport is not small', () => {
        expect(resolveCameraMaxDistance({
            baseMaxDistance: 1000,
            isSmallScreen: false,
            isLargeDesktopViewport: true,
            smallScreenZoomOutMultiplier: 0.9,
            desktopZoomOutMultiplier: 1.1
        })).toBe(1100);
    });
});
