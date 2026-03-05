import { toDebugArray } from './gpt2LaneWatchdogUtils.js';

export function logLayerNormVectorDump({ layerIndex, kind, lane, vectors = {} } = {}) {
    const root = (typeof window !== 'undefined') ? window : globalThis;
    const hasExplicitToggle = !!(root && Object.prototype.hasOwnProperty.call(root, '__LN_VECTOR_DEBUG'));
    const enabled = hasExplicitToggle ? root.__LN_VECTOR_DEBUG !== false : true;
    if (!enabled || !lane) return;

    lane.__lnVectorDebugLogged = lane.__lnVectorDebugLogged || {};
    const key = String(kind || 'ln').toLowerCase();
    if (lane.__lnVectorDebugLogged[key]) return;
    lane.__lnVectorDebugLogged[key] = true;

    const tokenLabel = lane.tokenLabel || '(unknown)';
    const headline = `[LN Value Dump] layer=${layerIndex} lane=${lane.laneIndex ?? 'n/a'} token=${lane.tokenIndex ?? 'n/a'} ${tokenLabel} ${key.toUpperCase()}`;
    if (typeof console.groupCollapsed === 'function') {
        console.groupCollapsed(headline);
    } else {
        console.log(headline);
    }

    console.log('meta', {
        layerIndex,
        laneIndex: lane.laneIndex,
        tokenIndex: lane.tokenIndex,
        tokenLabel: lane.tokenLabel,
        layerNorm: key,
    });

    const orderedLabels = [
        'normalizedSaved',
        'normalizedRuntime',
        'scaleParamSaved',
        'productComputed',
        'productSaved',
        'productUsedForColor',
        'shiftParamSaved',
        'shiftRuntime',
        'productPlusShiftComputed',
        'productPlusShiftSaved',
        'productPlusShiftUsedForColor',
    ];

    for (let i = 0; i < orderedLabels.length; i++) {
        const label = orderedLabels[i];
        const values = toDebugArray(vectors[label]);
        if (values) {
            console.log(`${label} (len=${values.length})`, values);
        } else {
            console.log(`${label}`, null);
        }
    }

    if (typeof console.groupEnd === 'function') {
        console.groupEnd();
    }
}
