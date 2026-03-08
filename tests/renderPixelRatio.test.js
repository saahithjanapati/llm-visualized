// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
    resolveRenderPixelRatio,
    setActiveRenderPixelRatioHint
} from '../src/utils/constants.js';
import { getEffectiveDevicePixelRatio } from '../src/utils/trailConstants.js';

function setViewport({ dpr, width, height }) {
    Object.defineProperty(window, 'devicePixelRatio', {
        configurable: true,
        value: dpr
    });
    Object.defineProperty(window, 'innerWidth', {
        configurable: true,
        value: width
    });
    Object.defineProperty(window, 'innerHeight', {
        configurable: true,
        value: height
    });
}

describe('render pixel ratio helpers', () => {
    beforeEach(() => {
        delete window.__RENDER_PIXEL_RATIO;
        delete window.__RENDER_DPR_CAP;
        setActiveRenderPixelRatioHint(null);
        setViewport({ dpr: 3, width: 390, height: 844 });
    });

    afterEach(() => {
        delete window.__RENDER_PIXEL_RATIO;
        delete window.__RENDER_DPR_CAP;
        setActiveRenderPixelRatioHint(null);
    });

    it('respects an explicit DPR cap override for high-density devices', () => {
        expect(resolveRenderPixelRatio({
            viewportWidth: window.innerWidth,
            viewportHeight: window.innerHeight
        })).toBe(2);

        expect(resolveRenderPixelRatio({
            viewportWidth: window.innerWidth,
            viewportHeight: window.innerHeight,
            dprCap: 2.4
        })).toBe(2.4);
    });

    it('lets trail scaling follow the actively applied renderer ratio', () => {
        expect(getEffectiveDevicePixelRatio()).toBe(2);

        setActiveRenderPixelRatioHint(2.35);
        expect(getEffectiveDevicePixelRatio()).toBeCloseTo(2.35, 5);

        setActiveRenderPixelRatioHint(null);
        expect(getEffectiveDevicePixelRatio()).toBe(2);
    });
});
