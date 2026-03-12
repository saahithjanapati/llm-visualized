import {
    findUserDataNumber,
    findUserDataString,
    getActivationDataFromSelection,
    resolveAttentionModeFromSelection
} from './selectionPanelSelectionUtils.js'
import { formatTokenLabelForPreview } from './selectionPanelFormatUtils.js'

export const SOFTMAX_PANEL_ACTION_OPEN = 'open-softmax-preview'

const SOFTMAX_PREVIEW_MAX_ROW_ITEMS = 8

function formatNumericValue(value, digits = 4, fallback = '--') {
    if (!Number.isFinite(value)) return fallback
    return value.toFixed(digits)
}

function resolveTokenLabel(selectionInfo = null, key = 'tokenLabel', activationSource = null, tokenIndex = null) {
    const explicit = findUserDataString(selectionInfo, key)
    if (explicit) return formatTokenLabelForPreview(explicit)
    if (activationSource && typeof activationSource.getTokenString === 'function' && Number.isFinite(tokenIndex)) {
        return formatTokenLabelForPreview(activationSource.getTokenString(tokenIndex))
    }
    return ''
}

function formatTokenDescriptor(tokenLabel = '', tokenIndex = null, fallback = 'Token') {
    const label = formatTokenLabelForPreview(tokenLabel)
    if (label && Number.isFinite(tokenIndex)) {
        return `${label} (position ${Math.floor(tokenIndex) + 1})`
    }
    if (label) return label
    if (Number.isFinite(tokenIndex)) return `${fallback} ${Math.floor(tokenIndex) + 1}`
    return fallback
}

function readAttentionRow(activationSource = null, layerIndex = null, mode = 'pre', headIndex = null, queryTokenIndex = null) {
    if (
        !activationSource
        || typeof activationSource.getAttentionScoresRow !== 'function'
        || !Number.isFinite(layerIndex)
        || !Number.isFinite(headIndex)
        || !Number.isFinite(queryTokenIndex)
    ) {
        return null
    }
    const row = activationSource.getAttentionScoresRow(
        Math.floor(layerIndex),
        mode,
        Math.floor(headIndex),
        Math.floor(queryTokenIndex)
    )
    return Array.isArray(row) ? row : null
}

function buildSoftmaxRowEntries({
    activationSource = null,
    layerIndex = null,
    headIndex = null,
    queryTokenIndex = null,
    selectedSourceTokenIndex = null
} = {}) {
    const preRow = readAttentionRow(activationSource, layerIndex, 'pre', headIndex, queryTokenIndex)
    const postRow = readAttentionRow(activationSource, layerIndex, 'post', headIndex, queryTokenIndex)
    const tokenCountValue = activationSource && typeof activationSource.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : null
    const maxTokenCount = Math.max(
        Number.isFinite(tokenCountValue) ? Math.floor(tokenCountValue) : 0,
        Array.isArray(preRow) ? preRow.length : 0,
        Array.isArray(postRow) ? postRow.length : 0,
        Number.isFinite(queryTokenIndex) ? Math.floor(queryTokenIndex) + 1 : 0,
        Number.isFinite(selectedSourceTokenIndex) ? Math.floor(selectedSourceTokenIndex) + 1 : 0
    )
    if (!maxTokenCount) {
        return {
            selectedEntry: null,
            displayEntries: [],
            rowSum: null,
            selectedRank: null,
            totalEntries: 0,
            hasFullRow: false
        }
    }

    const cappedQueryIndex = Number.isFinite(queryTokenIndex)
        ? Math.max(0, Math.floor(queryTokenIndex))
        : (maxTokenCount - 1)
    const fullEntries = []

    for (let tokenIndex = 0; tokenIndex < maxTokenCount; tokenIndex += 1) {
        if (tokenIndex > cappedQueryIndex) break
        const preScore = Array.isArray(preRow) ? preRow[tokenIndex] : null
        const postScore = Array.isArray(postRow) ? postRow[tokenIndex] : null
        const isSelected = Number.isFinite(selectedSourceTokenIndex)
            && tokenIndex === Math.floor(selectedSourceTokenIndex)
        if (!isSelected && !Number.isFinite(preScore) && !Number.isFinite(postScore)) continue

        fullEntries.push({
            tokenIndex,
            tokenLabel: activationSource && typeof activationSource.getTokenString === 'function'
                ? formatTokenLabelForPreview(activationSource.getTokenString(tokenIndex))
                : '',
            preScore,
            postScore,
            isSelected
        })
    }

    const sortedByWeight = fullEntries
        .filter((entry) => Number.isFinite(entry.postScore))
        .slice()
        .sort((a, b) => {
            if (b.postScore !== a.postScore) return b.postScore - a.postScore
            return a.tokenIndex - b.tokenIndex
        })

    const selectedEntry = fullEntries.find((entry) => entry.isSelected) || null
    const rowSum = sortedByWeight.reduce((sum, entry) => sum + entry.postScore, 0)
    const selectedRank = selectedEntry
        ? (sortedByWeight.findIndex((entry) => entry.tokenIndex === selectedEntry.tokenIndex) + 1 || null)
        : null

    const displayEntries = sortedByWeight.slice(0, SOFTMAX_PREVIEW_MAX_ROW_ITEMS)
    if (
        selectedEntry
        && !displayEntries.some((entry) => entry.tokenIndex === selectedEntry.tokenIndex)
    ) {
        if (displayEntries.length >= SOFTMAX_PREVIEW_MAX_ROW_ITEMS) {
            displayEntries[displayEntries.length - 1] = selectedEntry
        } else {
            displayEntries.push(selectedEntry)
        }
        displayEntries.sort((a, b) => {
            const postA = Number.isFinite(a.postScore) ? a.postScore : Number.NEGATIVE_INFINITY
            const postB = Number.isFinite(b.postScore) ? b.postScore : Number.NEGATIVE_INFINITY
            if (postB !== postA) return postB - postA
            return a.tokenIndex - b.tokenIndex
        })
    }

    return {
        selectedEntry,
        displayEntries,
        rowSum,
        selectedRank: Number.isFinite(selectedRank) && selectedRank > 0 ? selectedRank : null,
        totalEntries: fullEntries.length,
        hasFullRow: fullEntries.length === displayEntries.length
    }
}

function renderFormulaMarkup(tex = '') {
    const safeTex = typeof tex === 'string' ? tex.trim() : ''
    if (!safeTex) return ''
    const katex = (typeof window !== 'undefined') ? window.katex : null
    if (katex && typeof katex.renderToString === 'function') {
        try {
            return katex.renderToString(safeTex, { throwOnError: false, displayMode: true })
        } catch (_) {
            // Fall back to plain text below.
        }
    }
    return safeTex
}

function applySummaryValue(node, value, { wide = false } = {}) {
    if (!node) return
    node.textContent = value
    node.parentElement?.toggleAttribute('data-wide', wide)
}

export function isPostSoftmaxAttentionSelection(selectionInfo = null, normalizedLabel = '') {
    if (resolveAttentionModeFromSelection(selectionInfo) === 'post') return true
    const stageLower = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase()
    if (stageLower === 'attention.post') return true
    const lower = String(normalizedLabel || selectionInfo?.label || '').toLowerCase()
    return lower.includes('post-softmax attention score') || lower.includes('post softmax attention score')
}

export function setEquationsSoftmaxAction(equationsSectionEl, enabled = false) {
    if (!equationsSectionEl) return

    equationsSectionEl
        .querySelectorAll(`[data-detail-action="${SOFTMAX_PANEL_ACTION_OPEN}"]`)
        .forEach((node) => {
            const row = node.closest('.detail-equations-action-row')
            if (row) {
                row.remove()
                return
            }
            node.remove()
        })

    if (!enabled) return

    const actionRow = document.createElement('div')
    actionRow.className = 'detail-equations-action-row'

    const actionBtn = document.createElement('button')
    actionBtn.type = 'button'
    actionBtn.className = 'detail-description-action-link'
    actionBtn.dataset.detailAction = SOFTMAX_PANEL_ACTION_OPEN
    actionBtn.textContent = 'Explain softmax'
    actionBtn.setAttribute('aria-label', 'Open the softmax attention-weight explainer')

    actionRow.appendChild(actionBtn)
    equationsSectionEl.appendChild(actionRow)
}

export function resolveSoftmaxPreviewContext(selectionInfo = null, activationSource = null) {
    if (!isPostSoftmaxAttentionSelection(selectionInfo)) return null

    const activation = getActivationDataFromSelection(selectionInfo) || null
    const layerIndex = findUserDataNumber(selectionInfo, 'layerIndex')
    const headIndex = findUserDataNumber(selectionInfo, 'headIndex')
    const queryTokenIndex = findUserDataNumber(selectionInfo, 'tokenIndex')
    const sourceTokenIndex = findUserDataNumber(selectionInfo, 'keyTokenIndex')
    const queryTokenLabel = resolveTokenLabel(selectionInfo, 'tokenLabel', activationSource, queryTokenIndex)
    const sourceTokenLabel = resolveTokenLabel(selectionInfo, 'keyTokenLabel', activationSource, sourceTokenIndex)
    const rowContext = buildSoftmaxRowEntries({
        activationSource,
        layerIndex,
        headIndex,
        queryTokenIndex,
        selectedSourceTokenIndex: sourceTokenIndex
    })

    const selectedPreScore = Number.isFinite(activation?.preScore)
        ? activation.preScore
        : rowContext.selectedEntry?.preScore
    const selectedPostScore = Number.isFinite(activation?.postScore)
        ? activation.postScore
        : rowContext.selectedEntry?.postScore

    return {
        layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null,
        headIndex: Number.isFinite(headIndex) ? Math.floor(headIndex) : null,
        queryTokenIndex: Number.isFinite(queryTokenIndex) ? Math.floor(queryTokenIndex) : null,
        sourceTokenIndex: Number.isFinite(sourceTokenIndex) ? Math.floor(sourceTokenIndex) : null,
        queryTokenText: formatTokenDescriptor(queryTokenLabel, queryTokenIndex, 'Query token'),
        sourceTokenText: formatTokenDescriptor(sourceTokenLabel, sourceTokenIndex, 'Selected source token'),
        selectedPreScore,
        selectedPostScore,
        rowSum: rowContext.rowSum,
        selectedRank: rowContext.selectedRank,
        rowEntries: rowContext.displayEntries,
        rowEntryCount: rowContext.totalEntries,
        hasFullRow: rowContext.hasFullRow
    }
}

export function createSoftmaxDetailView(panelEl) {
    if (!panelEl || typeof document === 'undefined') return null

    const root = document.createElement('section')
    root.className = 'detail-softmax-view'
    root.setAttribute('aria-hidden', 'true')
    root.innerHTML = `
        <div class="detail-description detail-softmax-copy">
            <p data-softmax-copy-primary>Softmax turns one attention-score row into normalized weights.</p>
            <p data-softmax-copy-secondary>The selected score keeps only a share of the row total, so larger competitors pull probability mass away from it.</p>
        </div>
        <div class="detail-equations detail-softmax-equation-shell is-visible" aria-hidden="false">
            <div class="detail-equations-title">Equation</div>
            <div class="detail-equations-body detail-softmax-formula" data-softmax-formula></div>
        </div>
        <div class="detail-data detail-softmax-summary-card">
            <div class="detail-data-title">Selected attention weight</div>
            <div class="detail-softmax-summary-grid">
                <div class="detail-row detail-softmax-summary-item">
                    <span class="detail-label">Raw score</span>
                    <span class="detail-value" data-softmax-summary="pre">--</span>
                </div>
                <div class="detail-row detail-softmax-summary-item">
                    <span class="detail-label">Post-softmax weight</span>
                    <span class="detail-value" data-softmax-summary="post">--</span>
                </div>
                <div class="detail-row detail-softmax-summary-item">
                    <span class="detail-label">Row sum</span>
                    <span class="detail-value" data-softmax-summary="sum">--</span>
                </div>
                <div class="detail-row detail-softmax-summary-item">
                    <span class="detail-label">Rank in row</span>
                    <span class="detail-value" data-softmax-summary="rank">--</span>
                </div>
                <div class="detail-row detail-softmax-summary-item">
                    <span class="detail-label">Query row</span>
                    <span class="detail-value" data-softmax-summary="query">--</span>
                </div>
                <div class="detail-row detail-softmax-summary-item">
                    <span class="detail-label">Selected source</span>
                    <span class="detail-value" data-softmax-summary="source">--</span>
                </div>
            </div>
        </div>
        <div class="detail-data detail-softmax-notes-card">
            <div class="detail-data-title">What the expression does</div>
            <ul>
                <li>${'\\(e^{s_{t,j}}\\)'} turns each raw score into a positive strength.</li>
                <li>${'\\(\\sum_k e^{s_{t,k}}\\)'} adds the whole query row together.</li>
                <li>Dividing by that total converts the selected source token into its normalized share of attention.</li>
            </ul>
        </div>
        <div class="detail-data detail-softmax-row-card">
            <div class="detail-softmax-card-header">
                <div class="detail-data-title">Competing weights in this row</div>
                <div class="detail-softmax-row-hint" data-softmax-row-hint>Showing the strongest source tokens in this query row.</div>
            </div>
            <div class="detail-softmax-row-list" data-softmax-row-list></div>
        </div>
    `

    const header = panelEl.querySelector('.detail-header')
    if (header && header.nextSibling) {
        panelEl.insertBefore(root, header.nextSibling)
    } else {
        panelEl.appendChild(root)
    }

    const formulaEl = root.querySelector('[data-softmax-formula]')
    const primaryCopyEl = root.querySelector('[data-softmax-copy-primary]')
    const secondaryCopyEl = root.querySelector('[data-softmax-copy-secondary]')
    const rowHintEl = root.querySelector('[data-softmax-row-hint]')
    const rowListEl = root.querySelector('[data-softmax-row-list]')
    const summaryEls = {
        pre: root.querySelector('[data-softmax-summary="pre"]'),
        post: root.querySelector('[data-softmax-summary="post"]'),
        sum: root.querySelector('[data-softmax-summary="sum"]'),
        rank: root.querySelector('[data-softmax-summary="rank"]'),
        query: root.querySelector('[data-softmax-summary="query"]'),
        source: root.querySelector('[data-softmax-summary="source"]')
    }

    const state = {
        visible: false,
        context: null
    }

    function renderFormula() {
        if (!formulaEl) return
        const equation = String.raw`\begin{aligned}
\alpha_{t,j} &= \frac{\exp\left(s_{t,j}\right)}{\sum_k \exp\left(s_{t,k}\right)} \\
s_{t,j} &= \frac{q_{t,i} \cdot k_{j,i}}{\sqrt{d_h}} + M_{t,j} \\
\sum_j \alpha_{t,j} &= 1
\end{aligned}`
        const markup = renderFormulaMarkup(equation)
        if (markup === equation) {
            formulaEl.textContent = equation
        } else {
            formulaEl.innerHTML = markup
        }
    }

    function renderRowEntries(context = null) {
        if (!rowListEl) return
        rowListEl.innerHTML = ''
        const entries = Array.isArray(context?.rowEntries) ? context.rowEntries : []
        const maxPost = entries.reduce((maxValue, entry) => (
            Number.isFinite(entry?.postScore) && entry.postScore > maxValue
                ? entry.postScore
                : maxValue
        ), 0)

        if (!entries.length) {
            const emptyEl = document.createElement('div')
            emptyEl.className = 'detail-softmax-row-empty'
            emptyEl.textContent = 'No attention row data is available for this selection.'
            rowListEl.appendChild(emptyEl)
            return
        }

        entries.forEach((entry) => {
            const rowEl = document.createElement('div')
            rowEl.className = 'detail-softmax-row-item'
            if (entry.isSelected) rowEl.classList.add('is-selected')

            const headerEl = document.createElement('div')
            headerEl.className = 'detail-softmax-row-header'

            const tokenWrapEl = document.createElement('div')
            tokenWrapEl.className = 'detail-softmax-row-token-wrap'

            const tokenEl = document.createElement('div')
            tokenEl.className = 'detail-softmax-row-token'
            tokenEl.textContent = formatTokenDescriptor(entry.tokenLabel, entry.tokenIndex, 'Source token')

            const positionEl = document.createElement('div')
            positionEl.className = 'detail-softmax-row-position'
            positionEl.textContent = `Position ${entry.tokenIndex + 1}`

            tokenWrapEl.append(tokenEl, positionEl)

            const badgeEl = document.createElement('div')
            badgeEl.className = 'detail-softmax-row-badge'
            badgeEl.textContent = entry.isSelected ? 'Selected' : 'Row weight'

            headerEl.append(tokenWrapEl, badgeEl)

            const metricsEl = document.createElement('div')
            metricsEl.className = 'detail-softmax-row-metrics'
            metricsEl.innerHTML = `
                <span>score ${formatNumericValue(entry.preScore)}</span>
                <span>weight ${formatNumericValue(entry.postScore)}</span>
            `

            const trackEl = document.createElement('div')
            trackEl.className = 'detail-softmax-row-bar-track'

            const barEl = document.createElement('span')
            barEl.className = 'detail-softmax-row-bar'
            const barWidth = maxPost > 0 && Number.isFinite(entry.postScore)
                ? Math.max(0, Math.min(100, (entry.postScore / maxPost) * 100))
                : 0
            barEl.style.setProperty('--softmax-row-bar-width', `${barWidth.toFixed(2)}%`)

            trackEl.appendChild(barEl)
            rowEl.append(headerEl, metricsEl, trackEl)
            rowListEl.appendChild(rowEl)
        })
    }

    function renderContext() {
        const context = state.context
        applySummaryValue(summaryEls.pre, formatNumericValue(context?.selectedPreScore))
        applySummaryValue(summaryEls.post, formatNumericValue(context?.selectedPostScore))
        applySummaryValue(summaryEls.sum, formatNumericValue(context?.rowSum))
        applySummaryValue(
            summaryEls.rank,
            Number.isFinite(context?.selectedRank) && Number.isFinite(context?.rowEntryCount)
                ? `${context.selectedRank} / ${context.rowEntryCount}`
                : '--'
        )
        applySummaryValue(summaryEls.query, context?.queryTokenText || '--', { wide: true })
        applySummaryValue(summaryEls.source, context?.sourceTokenText || '--', { wide: true })

        if (primaryCopyEl) {
            if (context?.queryTokenText && context?.sourceTokenText) {
                primaryCopyEl.textContent = `For ${context.queryTokenText}, softmax compares every source token in the row and assigns ${context.sourceTokenText} one normalized attention weight.`
            } else {
                primaryCopyEl.textContent = 'Softmax turns one attention-score row into normalized weights.'
            }
        }
        if (secondaryCopyEl) {
            secondaryCopyEl.textContent = Number.isFinite(context?.selectedPostScore)
                ? `The selected source token keeps ${formatNumericValue(context.selectedPostScore)} of the row's total attention mass after normalization.`
                : 'The selected score keeps only a share of the row total, so larger competitors pull probability mass away from it.'
        }
        if (rowHintEl) {
            rowHintEl.textContent = context?.hasFullRow
                ? `Showing all ${context?.rowEntryCount || 0} allowed source tokens in this query row.`
                : `Showing the strongest ${Math.min(context?.rowEntries?.length || 0, context?.rowEntryCount || 0)} source tokens out of ${context?.rowEntryCount || 0} in this query row.`
        }
        renderRowEntries(context)
    }

    return {
        root,
        setContext(context = null) {
            state.context = context
            if (state.visible) {
                renderFormula()
                renderContext()
            }
        },
        setVisible(visible) {
            state.visible = !!visible
            root.classList.toggle('is-visible', state.visible)
            root.setAttribute('aria-hidden', state.visible ? 'false' : 'true')
            if (!state.visible) return
            renderFormula()
            renderContext()
        },
        resizeAndRender() {
            if (!state.visible) return
            renderContext()
        },
        remove() {
            root.remove()
        }
    }
}
