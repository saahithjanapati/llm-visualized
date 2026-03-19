import {
    MHA_FINAL_K_COLOR,
    MHA_FINAL_Q_COLOR,
    MHA_VALUE_CLAMP_MAX,
    MHA_VALUE_HUE_SPREAD,
    MHA_VALUE_LIGHTNESS_MAX,
    MHA_VALUE_LIGHTNESS_MIN,
    MHA_VALUE_RANGE_MAX,
    MHA_VALUE_RANGE_MIN,
    MHA_VALUE_SPECTRUM_COLOR
} from '../animations/LayerAnimationConstants.js';
import { buildHueRangeOptions } from './colors.js';

export function resolveAttentionVectorBaseColor(kind = '') {
    const safeKind = String(kind || '').trim().toLowerCase();
    if (safeKind === 'k') return MHA_FINAL_K_COLOR;
    if (safeKind === 'v') return MHA_VALUE_SPECTRUM_COLOR;
    return MHA_FINAL_Q_COLOR;
}

export function buildAttentionVectorRangeOptions(kind = '') {
    return buildHueRangeOptions(resolveAttentionVectorBaseColor(kind), {
        hueSpread: MHA_VALUE_HUE_SPREAD,
        minLightness: MHA_VALUE_LIGHTNESS_MIN,
        maxLightness: MHA_VALUE_LIGHTNESS_MAX,
        valueMin: MHA_VALUE_RANGE_MIN,
        valueMax: MHA_VALUE_RANGE_MAX,
        valueClampMax: MHA_VALUE_CLAMP_MAX
    });
}
