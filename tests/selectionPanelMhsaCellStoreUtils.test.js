import { describe, expect, it } from 'vitest';
import {
    createMhsaTokenMatrixCellStore,
    forEachMhsaTokenMatrixCell,
    registerMhsaTokenMatrixCell
} from '../src/ui/selectionPanelMhsaCellStoreUtils.js';

describe('selectionPanelMhsaCellStoreUtils', () => {
    it('keeps duplicate cells for the same logical attention score', () => {
        const store = createMhsaTokenMatrixCellStore(3);
        const firstCell = { id: 'first-post' };
        const secondCell = { id: 'second-post' };

        registerMhsaTokenMatrixCell(store, 1, 2, firstCell);
        registerMhsaTokenMatrixCell(store, 1, 2, secondCell);

        const visited = [];
        forEachMhsaTokenMatrixCell(store, (cellEl, rowIndex, colIndex) => {
            visited.push({
                cellEl,
                rowIndex,
                colIndex
            });
        });

        expect(visited).toEqual([
            { cellEl: firstCell, rowIndex: 1, colIndex: 2 },
            { cellEl: secondCell, rowIndex: 1, colIndex: 2 }
        ]);
    });

    it('ignores out-of-range or invalid registrations', () => {
        const store = createMhsaTokenMatrixCellStore(1);
        registerMhsaTokenMatrixCell(store, -1, 0, { id: 'bad-row' });
        registerMhsaTokenMatrixCell(store, 0, Number.NaN, { id: 'bad-col' });

        const visited = [];
        forEachMhsaTokenMatrixCell(store, (cellEl) => {
            visited.push(cellEl);
        });

        expect(visited).toEqual([]);
    });
});
