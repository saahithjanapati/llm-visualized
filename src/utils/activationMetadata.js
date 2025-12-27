export function buildActivationData({
    label,
    values,
    copyValues = false,
    stage,
    layerIndex,
    tokenIndex,
    tokenLabel,
    headIndex,
    keyTokenIndex,
    keyTokenLabel,
    segmentIndex,
    preScore,
    postScore,
    notes,
} = {}) {
    const data = {};
    if (label) data.label = label;
    if (stage) data.stage = stage;
    if (Number.isFinite(layerIndex)) data.layerIndex = layerIndex;
    if (Number.isFinite(tokenIndex)) data.tokenIndex = tokenIndex;
    if (tokenLabel) data.tokenLabel = tokenLabel;
    if (Number.isFinite(headIndex)) data.headIndex = headIndex;
    if (Number.isFinite(keyTokenIndex)) data.keyTokenIndex = keyTokenIndex;
    if (keyTokenLabel) data.keyTokenLabel = keyTokenLabel;
    if (Number.isFinite(segmentIndex)) data.segmentIndex = segmentIndex;
    if (Array.isArray(values) || ArrayBuffer.isView(values)) {
        data.values = copyValues ? Array.from(values) : values;
    }
    if (Number.isFinite(preScore)) data.preScore = preScore;
    if (Number.isFinite(postScore)) data.postScore = postScore;
    if (notes) data.notes = notes;
    return data;
}

export function applyActivationDataToObject(object3d, activationData, label = null) {
    if (!object3d) return;
    object3d.userData = object3d.userData || {};
    if (label) object3d.userData.label = label;
    if (activationData) object3d.userData.activationData = activationData;
}

export function applyActivationDataToVector(vector, activationData, label = null) {
    if (!vector) return;
    if (vector.userData) {
        vector.userData.activationData = activationData;
    } else {
        vector.userData = { activationData };
    }
    if (vector.group) {
        vector.group.userData = vector.group.userData || {};
        if (label) vector.group.userData.label = label;
        if (activationData) vector.group.userData.activationData = activationData;
    }
    if (vector.mesh) {
        const existing = vector.mesh.userData || {};
        vector.mesh.userData = {
            ...existing,
            ...(label ? { label } : {}),
            ...(activationData ? { activationData } : {}),
        };
    }
}
