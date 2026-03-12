import { describe, expect, it } from 'vitest';
import {
    buildResidualRowHoverPayload,
    resolveActiveFocusLabel,
    resolveActiveSemanticTarget,
    resolveDetailTargetsFromSemanticTarget,
    resolveFocusSemanticTargets
} from '../src/view2d/transformerView2dTargets.js';
import { SPACE_TOKEN_DISPLAY } from '../src/ui/selectionPanelConstants.js';

describe('transformerView2dTargets', () => {
    it('derives head, concat, and projection detail targets from semantic targets', () => {
        expect(resolveDetailTargetsFromSemanticTarget({
            componentKind: 'mhsa',
            layerIndex: 2,
            headIndex: 5,
            stage: 'attention',
            role: 'head'
        })).toEqual({
            headDetailTarget: {
                layerIndex: 2,
                headIndex: 5
            },
            concatDetailTarget: null,
            outputProjectionDetailTarget: null
        });

        expect(resolveDetailTargetsFromSemanticTarget({
            componentKind: 'mhsa',
            layerIndex: 4,
            stage: 'concatenate',
            role: 'concat'
        })).toEqual({
            headDetailTarget: null,
            concatDetailTarget: {
                layerIndex: 4
            },
            outputProjectionDetailTarget: null
        });

        expect(resolveDetailTargetsFromSemanticTarget({
            componentKind: 'output-projection',
            layerIndex: 1,
            stage: 'attn-out',
            role: 'projection-weight'
        })).toEqual({
            headDetailTarget: null,
            concatDetailTarget: null,
            outputProjectionDetailTarget: {
                layerIndex: 1
            }
        });
    });

    it('prefers active detail targets over the base semantic target', () => {
        const targetState = {
            baseSemanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 0,
                stage: 'attention',
                role: 'module'
            },
            baseFocusLabel: 'Layer 1 Self-Attention',
            headDetailTarget: {
                layerIndex: 0,
                headIndex: 3
            }
        };

        expect(resolveActiveSemanticTarget(targetState)).toEqual({
            componentKind: 'mhsa',
            layerIndex: 0,
            headIndex: 3,
            stage: 'attention',
            role: 'head'
        });
        expect(resolveActiveFocusLabel(targetState)).toBe('Layer 1 Attention Head 4');
    });

    it('keeps the base focus label when no detail target is active', () => {
        expect(resolveActiveFocusLabel({
            baseSemanticTarget: {
                componentKind: 'mlp',
                layerIndex: 6,
                stage: 'mlp',
                role: 'module'
            },
            baseFocusLabel: 'Layer 7 MLP module'
        })).toBe('Layer 7 MLP module');
    });

    it('builds ordered focus semantic candidates for detail views', () => {
        expect(resolveFocusSemanticTargets({
            headDetailTarget: {
                layerIndex: 1,
                headIndex: 2
            }
        })).toEqual([
            {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 2,
                stage: 'attention',
                role: 'head-card'
            },
            {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 2,
                stage: 'attention',
                role: 'head'
            }
        ]);

        expect(resolveFocusSemanticTargets({
            outputProjectionDetailTarget: {
                layerIndex: 5
            }
        })).toEqual([
            {
                componentKind: 'output-projection',
                layerIndex: 5,
                stage: 'attn-out',
                role: 'projection-weight'
            },
            {
                componentKind: 'output-projection',
                layerIndex: 5,
                stage: 'attn-out',
                role: 'module'
            }
        ]);
    });

    it('formats whitespace residual-row labels the same way as the 3D token display', () => {
        const payload = buildResidualRowHoverPayload({
            rowItem: {
                label: ' ',
                semantic: {
                    componentKind: 'residual',
                    stage: 'incoming',
                    layerIndex: 0,
                    tokenIndex: 4
                }
            }
        }, {
            getTokenId(tokenIndex) {
                return tokenIndex + 100;
            },
            getTokenString() {
                return ' ';
            }
        });

        expect(payload).toEqual({
            label: 'Residual Stream Vector',
            info: {
                layerIndex: 0,
                tokenIndex: 4,
                tokenId: 104,
                tokenLabel: SPACE_TOKEN_DISPLAY,
                activationData: {
                    label: 'Residual Stream Vector',
                    stage: 'layer.incoming',
                    layerIndex: 0,
                    tokenIndex: 4,
                    tokenLabel: SPACE_TOKEN_DISPLAY
                }
            }
        });
    });
});
