import { describe, expect, it } from 'vitest';
import {
    MLP_OVERLAY_STAGE_DOWN,
    MLP_OVERLAY_STAGE_GELU,
    MLP_OVERLAY_STAGE_UP,
    resolveMlpOverlayStage
} from '../src/ui/statusOverlayMlpEquationUtils.js';

describe('statusOverlayMlpEquationUtils', () => {
    it('prefers the down-projection equation once down-projection begins', () => {
        expect(resolveMlpOverlayStage([
            { mlpUpStarted: true, mlpGeluComplete: true, mlpDownStarted: true }
        ])).toBe(MLP_OVERLAY_STAGE_DOWN);
    });

    it('shows the GELU equation after the up-projection and before down-projection', () => {
        expect(resolveMlpOverlayStage([
            { mlpUpStarted: true, mlpGeluActive: true, mlpDownStarted: false }
        ])).toBe(MLP_OVERLAY_STAGE_GELU);
    });

    it('falls back to the up-projection equation when the MLP handoff is ready but GELU has not started', () => {
        expect(resolveMlpOverlayStage([
            { ln2Phase: 'mlpReady' }
        ])).toBe(MLP_OVERLAY_STAGE_UP);
    });
});
