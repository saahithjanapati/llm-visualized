import { describe, expect, it } from 'vitest';

import {
    buildResidualRowHoverPayload,
    isMlpOverviewEntry,
    isOutputProjectionOverviewEntry,
    resolveDetailTargetsFromSemanticTarget,
    resolveFocusSemanticTargets,
    resolveTransformerView2dActionContext,
    resolveTransformerView2dOpenTransitionMode
} from './transformerView2dTargets.js';

describe('transformerView2dTargets', () => {
    it('stages layer norm selections from the overview before focusing the module', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'LayerNorm 1 Scale',
            info: {
                layerIndex: 2,
                activationData: {
                    stage: 'ln1.param.scale',
                    layerIndex: 2
                }
            }
        }, 'LayerNorm 1 Scale');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'layer-norm',
                layerIndex: 2,
                stage: 'ln1',
                role: 'module'
            },
            transitionMode: 'staged-focus'
        });
    });

    it('uses the staged head-detail opening flow for attention head targets', () => {
        expect(resolveTransformerView2dOpenTransitionMode({
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 3,
                stage: 'attention',
                role: 'head'
            }
        })).toBe('staged-head-detail');
    });

    it('prefers the layer norm card and title when resolving focus candidates', () => {
        expect(resolveFocusSemanticTargets({
            semanticTarget: {
                componentKind: 'layer-norm',
                layerIndex: 0,
                stage: 'ln2',
                role: 'module'
            }
        })).toEqual([
            {
                componentKind: 'layer-norm',
                layerIndex: 0,
                stage: 'ln2',
                role: 'module-card'
            },
            {
                componentKind: 'layer-norm',
                layerIndex: 0,
                stage: 'ln2',
                role: 'module-title'
            },
            {
                componentKind: 'layer-norm',
                layerIndex: 0,
                stage: 'ln2',
                role: 'module'
            }
        ]);
    });

    it('leaves non-head detail routes on the existing direct-open path', () => {
        expect(resolveTransformerView2dOpenTransitionMode({
            semanticTarget: {
                componentKind: 'output-projection',
                layerIndex: 5,
                stage: 'attn-out',
                role: 'projection-weight'
            }
        })).toBe('');
    });

    it('redirects concat overview targets into output-projection detail', () => {
        expect(resolveDetailTargetsFromSemanticTarget({
            componentKind: 'mhsa',
            layerIndex: 2,
            stage: 'concatenate',
            role: 'concat'
        })).toEqual({
            headDetailTarget: null,
            concatDetailTarget: null,
            outputProjectionDetailTarget: {
                layerIndex: 2
            },
            mlpDetailTarget: null
        });
    });

    it('maps concatenate selections to output projection for 2D view actions', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Concatenate Heads',
            info: {
                layerIndex: 4,
                activationData: {
                    stage: 'attention.concatenate',
                    layerIndex: 4
                }
            }
        }, 'Concatenate Heads');

        expect(context).toMatchObject({
            semanticTarget: {
                componentKind: 'output-projection',
                layerIndex: 4,
                stage: 'attn-out',
                role: 'projection-weight'
            },
            focusLabel: 'Layer 5 Output Projection'
        });
    });

    it('treats both output-projection title lines as overview entries', () => {
        expect(isOutputProjectionOverviewEntry({
            role: 'module-title-top',
            semantic: {
                componentKind: 'output-projection',
                layerIndex: 5,
                stage: 'attn-out',
                role: 'module-title-top'
            }
        })).toBe(true);
        expect(isOutputProjectionOverviewEntry({
            role: 'module-title-bottom',
            semantic: {
                componentKind: 'output-projection',
                layerIndex: 5,
                stage: 'attn-out',
                role: 'module-title-bottom'
            }
        })).toBe(true);
    });

    it('treats both MLP title lines as overview entries', () => {
        expect(isMlpOverviewEntry({
            role: 'module-title-top',
            semantic: {
                componentKind: 'mlp',
                layerIndex: 5,
                stage: 'mlp',
                role: 'module-title-top'
            }
        })).toBe(true);
        expect(isMlpOverviewEntry({
            role: 'module-title-bottom',
            semantic: {
                componentKind: 'mlp',
                layerIndex: 5,
                stage: 'mlp',
                role: 'module-title-bottom'
            }
        })).toBe(true);
    });

    it('formats post-layernorm residual row hover labels from the semantic stage', () => {
        const payload = buildResidualRowHoverPayload({
            rowItem: {
                label: 'Token A',
                semantic: {
                    componentKind: 'residual',
                    layerIndex: 0,
                    stage: 'ln2.shift',
                    tokenIndex: 0
                }
            }
        }, {
            getTokenId() {
                return 42;
            }
        });

        expect(payload?.label).toBe('Post LayerNorm 2 Residual Vector');
        expect(payload?.info?.activationData?.label).toBe('Post LayerNorm 2 Residual Vector');
        expect(payload?.info?.activationData?.stage).toBe('ln2.shift');
    });
});
