export function createMhsaTokenMatrixCellStore(rowCount = 0) {
    const safeRowCount = Number.isFinite(rowCount) && rowCount > 0
        ? Math.floor(rowCount)
        : 0;
    return Array.from({ length: safeRowCount }, () => []);
}

export function registerMhsaTokenMatrixCell(cellStore, rowIndex, colIndex, cellEl) {
    if (!Array.isArray(cellStore) || !cellEl) return;
    const safeRowIndex = Number.isFinite(rowIndex) ? Math.floor(rowIndex) : null;
    const safeColIndex = Number.isFinite(colIndex) ? Math.floor(colIndex) : null;
    if (!Number.isFinite(safeRowIndex) || !Number.isFinite(safeColIndex)) return;
    if (safeRowIndex < 0 || safeColIndex < 0) return;
    if (!Array.isArray(cellStore[safeRowIndex])) return;

    const existingEntry = cellStore[safeRowIndex][safeColIndex];
    if (Array.isArray(existingEntry)) {
        existingEntry.push(cellEl);
        return;
    }
    if (existingEntry) {
        cellStore[safeRowIndex][safeColIndex] = [existingEntry, cellEl];
        return;
    }
    cellStore[safeRowIndex][safeColIndex] = cellEl;
}

export function forEachMhsaTokenMatrixCell(cellStore, visitor) {
    if (!Array.isArray(cellStore) || typeof visitor !== 'function') return;

    cellStore.forEach((rowEntries, rowIndex) => {
        if (!Array.isArray(rowEntries)) return;
        rowEntries.forEach((entry, colIndex) => {
            if (Array.isArray(entry)) {
                entry.forEach((cellEl) => visitor(cellEl, rowIndex, colIndex));
                return;
            }
            visitor(entry, rowIndex, colIndex);
        });
    });
}
