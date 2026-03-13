export function sliceMlpActivationIntoSegments(values, segmentCount, segmentLength) {
    const safeSegmentCount = Math.max(1, Math.floor(segmentCount || 1));
    const safeSegmentLength = Math.max(1, Math.floor(segmentLength || 1));
    const raw = Array.isArray(values) || ArrayBuffer.isView(values)
        ? Array.from(values)
        : [];
    while (raw.length < safeSegmentCount * safeSegmentLength) raw.push(0);

    return Array.from({ length: safeSegmentCount }, (_, index) => {
        const start = index * safeSegmentLength;
        return raw.slice(start, start + safeSegmentLength);
    });
}
