import { MHA_FINAL_Q_COLOR } from '../animations/LayerAnimationConstants.js';
import { buildHueRangeOptions } from './colors.js';

// Inactive 3D LayerNorm parameter banks stay monochrome until the animation activates them.
export const LAYER_NORM_PARAM_INACTIVE_COLOR_OPTIONS = Object.freeze({
    type: 'monochromatic',
    baseHue: 0,
    saturation: 0,
    minLightness: 0.03,
    maxLightness: 0.88,
    useData: true,
    valueMin: -1.8,
    valueMax: 1.8
});

// The 2D detail canvas should show the live/active parameter palette rather than the dormant gray bank color.
export const LAYER_NORM_PARAM_ACTIVE_COLOR_OPTIONS = Object.freeze(buildHueRangeOptions(
    MHA_FINAL_Q_COLOR,
    {
        hueSpread: 0.1,
        minLightness: 0.34,
        maxLightness: 0.74,
        valueMin: -1.8,
        valueMax: 1.8
    }
));

// Backward-compatible default for places that still mean the inactive bank palette.
export const LAYER_NORM_PARAM_COLOR_OPTIONS = LAYER_NORM_PARAM_INACTIVE_COLOR_OPTIONS;
