import { beforeAll, describe, expect, it, vi } from 'vitest';

let SelectionPanel;

beforeAll(async () => {
    vi.stubGlobal('localStorage', {
        getItem: () => null,
        setItem: () => {},
        removeItem: () => {}
    });
    ({ SelectionPanel } = await import('./selectionPanel.js'));
});

function createPanelContext() {
    const panel = Object.create(SelectionPanel.prototype);
    panel.tokenInfoRow = {
        style: { display: '' },
        dataset: {}
    };
    panel.tokenInfoHeadPrimary = { textContent: '' };
    panel.tokenInfoHeadSecondary = { textContent: '' };
    panel.tokenInfoHeadTertiary = { textContent: '' };
    panel.tokenInfoText = { textContent: '', title: '' };
    panel.tokenInfoId = { textContent: '', title: '' };
    panel.tokenInfoPosition = { textContent: '', title: '' };
    panel._resolveVectorTokenPosition = vi.fn(() => ({
        tokenText: 'the',
        tokenDisplayText: 'the',
        tokenIdText: '464',
        positionText: '2',
        tokenIndex: 1,
        tokenId: 464
    }));
    return panel;
}

describe('SelectionPanel vector token metadata rows', () => {
    it('hides the token metadata row for LayerNorm product vectors', () => {
        const panel = createPanelContext();
        const selection = {
            label: 'LayerNorm 1 Product Vector',
            info: {
                activationData: {
                    stage: 'ln1.product',
                    layerIndex: 3,
                    tokenIndex: 1,
                    tokenLabel: 'the'
                }
            }
        };

        const metadata = panel._updateVectorTokenPositionRows(selection, selection.label);

        expect(metadata).toMatchObject({
            tokenText: 'the',
            tokenIdText: '464',
            positionText: '2'
        });
        expect(panel.tokenInfoRow.style.display).toBe('none');
        expect(panel.tokenInfoRow.dataset.empty).toBe('true');
        expect(panel.tokenInfoHeadPrimary.textContent).toBe('Raw token');
        expect(panel.tokenInfoHeadSecondary.textContent).toBe('Token ID');
        expect(panel.tokenInfoHeadTertiary.textContent).toBe('Position');
    });
});
