export function rebuildSideCopyIndex(lane) {
    if (!lane) return null;
    const index = { byHead: [], count: 0 };
    const copies = Array.isArray(lane.sideCopies) ? lane.sideCopies : [];
    index.count = copies.length;
    copies.forEach((entry) => {
        if (!entry || !Number.isFinite(entry.headIndex) || !entry.type) return;
        const headIdx = entry.headIndex;
        let headMap = index.byHead[headIdx];
        if (!headMap) {
            headMap = {};
            index.byHead[headIdx] = headMap;
        }
        headMap[entry.type] = entry;
    });
    lane._mhsaSideCopyIndex = index;
    return index;
}

export function setSideCopyEntry(lane, headIdx, type, entry) {
    if (!lane || !Number.isFinite(headIdx) || !type) return;
    let index = lane._mhsaSideCopyIndex;
    if (!index) {
        index = rebuildSideCopyIndex(lane);
    }
    if (!index) return;
    let headMap = index.byHead[headIdx];
    if (!headMap) {
        headMap = {};
        index.byHead[headIdx] = headMap;
    }
    headMap[type] = entry;
    if (Array.isArray(lane.sideCopies)) {
        index.count = lane.sideCopies.length;
    }
}

export function getSideCopyEntry(lane, headIdx, type) {
    if (!lane || !Number.isFinite(headIdx) || !type) return null;
    const copies = Array.isArray(lane.sideCopies) ? lane.sideCopies : null;
    let index = lane._mhsaSideCopyIndex;
    const count = copies ? copies.length : 0;
    if (!index || index.count !== count) {
        index = rebuildSideCopyIndex(lane);
    }
    const headMap = index && index.byHead ? index.byHead[headIdx] : null;
    if (headMap && headMap[type]) return headMap[type];
    if (copies) {
        const found = copies.find(sc => sc && sc.headIndex === headIdx && sc.type === type) || null;
        if (found) {
            setSideCopyEntry(lane, headIdx, type, found);
        }
        return found;
    }
    return null;
}
