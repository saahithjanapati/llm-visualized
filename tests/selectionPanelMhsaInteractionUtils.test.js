// @vitest-environment jsdom

import { describe, expect, it } from 'vitest';

import { resolveMhsaTokenMatrixProjectionStageTarget } from '../src/ui/selectionPanelMhsaInteractionUtils.js';

function createStageFixture() {
    document.body.innerHTML = `
        <div id="root">
            <div
                class="mhsa-token-matrix-preview__stage"
                data-mhsa-projection-stage-index="2"
                data-mhsa-projection-kind="v"
            >
                <div class="mhsa-token-matrix-preview__x-block">
                    <div class="mhsa-token-matrix-preview__x-matrix">
                        <div class="mhsa-token-matrix-preview__row">
                            <div class="mhsa-token-matrix-preview__row-bar"></div>
                        </div>
                    </div>
                </div>
                <div class="mhsa-token-matrix-preview__stage-gap">
                    <span class="mhsa-token-matrix-preview__gap-fill"></span>
                </div>
                <div class="mhsa-token-matrix-preview__weight-block">
                    <div class="mhsa-token-matrix-preview__weight-card"></div>
                    <div class="mhsa-token-matrix-preview__caption"></div>
                </div>
                <div class="mhsa-token-matrix-preview__operator mhsa-token-matrix-preview__operator--wb">+</div>
            </div>
        </div>
    `;

    return {
        root: document.getElementById('root'),
        rowBar: document.querySelector('.mhsa-token-matrix-preview__row-bar'),
        gapFill: document.querySelector('.mhsa-token-matrix-preview__gap-fill'),
        weightCard: document.querySelector('.mhsa-token-matrix-preview__weight-card'),
        plusOperator: document.querySelector('.mhsa-token-matrix-preview__operator--wb')
    };
}

describe('selectionPanelMhsaInteractionUtils', () => {
    it('keeps the broad stage shell for hover fallback', () => {
        const { root, gapFill } = createStageFixture();

        expect(resolveMhsaTokenMatrixProjectionStageTarget(gapFill, { root })).toEqual({
            stageIndex: 2,
            focusKey: 'v'
        });
    });

    it('treats empty stage gaps as background for pin clicks', () => {
        const { root, gapFill } = createStageFixture();

        expect(resolveMhsaTokenMatrixProjectionStageTarget(gapFill, {
            root,
            requirePinnableComponent: true
        })).toBeNull();
    });

    it('keeps actual stage blocks and operators pinnable', () => {
        const { root, rowBar, weightCard, plusOperator } = createStageFixture();

        expect(resolveMhsaTokenMatrixProjectionStageTarget(rowBar, {
            root,
            requirePinnableComponent: true
        })).toEqual({
            stageIndex: 2,
            focusKey: 'v'
        });
        expect(resolveMhsaTokenMatrixProjectionStageTarget(weightCard, {
            root,
            requirePinnableComponent: true
        })).toEqual({
            stageIndex: 2,
            focusKey: 'v'
        });
        expect(resolveMhsaTokenMatrixProjectionStageTarget(plusOperator, {
            root,
            requirePinnableComponent: true
        })).toEqual({
            stageIndex: 2,
            focusKey: 'v'
        });
    });
});
