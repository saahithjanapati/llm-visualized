import { toDebugArray } from './gpt2LaneWatchdogUtils.js';
import {
    consoleGroupCollapsed,
    consoleGroupEnd,
    consoleLog,
    isConsoleLoggingEnabled
} from '../../utils/runtimeConsole.js';

export function logLayerNormVectorDump({ layerIndex, kind, lane, vectors = {} } = {}) {
    const root = (typeof window !== 'undefined') ? window : globalThis;
    const hasExplicitToggle = !!(root && Object.prototype.hasOwnProperty.call(root, '__LN_VECTOR_DEBUG'));
    const enabled = isConsoleLoggingEnabled() && (hasExplicitToggle ? root.__LN_VECTOR_DEBUG !== false : true);
    if (!enabled || !lane) return;

    lane.__lnVectorDebugLogged = lane.__lnVectorDebugLogged || {};
    const key = String(kind || 'ln').toLowerCase();
    if (lane.__lnVectorDebugLogged[key]) return;
    lane.__lnVectorDebugLogged[key] = true;

    const tokenLabel = lane.tokenLabel || '(unknown)';
    const headline = `[LN Value Dump] layer=${layerIndex} lane=${lane.laneIndex ?? 'n/a'} token=${lane.tokenIndex ?? 'n/a'} ${tokenLabel} ${key.toUpperCase()}`;
    consoleGroupCollapsed(headline);

    consoleLog('meta', {
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
            consoleLog(`${label} (len=${values.length})`, values);
        } else {
            consoleLog(`${label}`, null);
        }
    }

    consoleGroupEnd();
}
