import { initSelectionPanel } from '../../ui/selectionPanel.js';

export function createLazySelectionPanel(options = {}) {
    const baseOptions = { ...options };
    let latestData = {
        activationSource: baseOptions.activationSource,
        laneTokenIndices: baseOptions.laneTokenIndices,
        tokenLabels: baseOptions.tokenLabels
    };

    // Keep the existing factory/API shape, but initialize eagerly so the
    // selection panel bundle and styles are ready before the first user open.
    const panel = initSelectionPanel({
        ...baseOptions,
        activationSource: latestData.activationSource,
        laneTokenIndices: latestData.laneTokenIndices,
        tokenLabels: latestData.tokenLabels
    });

    panel.updateData?.(latestData);

    const updateLoadedPanelData = (data = {}) => {
        latestData = {
            ...latestData,
            ...data
        };
        panel.updateData?.(data);
    };

    return {
        handleSelection: (selection) => panel.handleSelection(selection),
        close: () => {
            panel.close?.();
        },
        updateData: (data) => {
            updateLoadedPanelData(data);
        },
        openTransformerView2d: (config) => panel.openTransformerView2d(config),
        prewarmTransformerView2d: (config) => panel.prewarmTransformerView2d?.(config),
        isTransformerView2dOpen: () => panel.isTransformerView2dOpen?.() === true
    };
}
