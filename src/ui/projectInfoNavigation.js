export const PROJECT_INFO_PAGE_PATH = '/info/';

export function openProjectInfoPage(locationRef = null) {
    const resolvedLocation = locationRef
        || (typeof window !== 'undefined' ? window.location : null);
    if (!resolvedLocation || typeof resolvedLocation.assign !== 'function') {
        return false;
    }
    resolvedLocation.assign(PROJECT_INFO_PAGE_PATH);
    return true;
}
