import {
    getActivationDataFromSelection,
    resolveAttentionModeFromSelection
} from './selectionPanelSelectionUtils.js'

export const SOFTMAX_PANEL_ACTION_OPEN = 'open-softmax-preview'

const SOFTMAX_EQUATION_TEX = 'p_j = \\frac{e^{s_j}}{\\sum_k e^{s_k}}'
const SOFTMAX_OVERVIEW_LINES = [
    'Softmax takes a list of scores and turns it into normalized weights.',
    'Higher scores get more weight, lower scores get less, and the full output adds up to 1.'
]
const SOFTMAX_WHAT_IT_DOES = 'The exponential makes every output positive and magnifies relative differences between scores. Dividing by the row total then converts each score into its share of the whole.'
const SOFTMAX_WHY_ATTENTION = 'Attention starts with raw query-key compatibility scores. Softmax turns that row into a stable, normalized set of weights so the model can mix value vectors in a controlled way.'
const SOFTMAX_COPY_PANEL_BLURB = 'This selection preview panel includes a short overview of softmax, the softmax equation itself, a brief note about what the equation does, and a brief note about why attention uses it.'

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

export function isPostSoftmaxAttentionSelection(selectionInfo = null, normalizedLabel = '') {
    if (resolveAttentionModeFromSelection(selectionInfo) === 'post') return true
    const stageLower = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase()
    if (stageLower === 'attention.post') return true
    const lower = String(normalizedLabel || selectionInfo?.label || '').toLowerCase()
    return lower.includes('post-softmax attention score') || lower.includes('post softmax attention score')
}

export function setDescriptionSoftmaxAction(descriptionEl, enabled = false) {
    if (!descriptionEl) return

    descriptionEl
        .querySelectorAll(`[data-detail-action="${SOFTMAX_PANEL_ACTION_OPEN}"]`)
        .forEach((node) => {
            const row = node.closest('.detail-description-action-row')
            if (row) {
                row.remove()
                return
            }
            node.remove()
        })

    if (!enabled) return

    const actionRow = document.createElement('div')
    actionRow.className = 'detail-description-action-row'

    const actionBtn = document.createElement('button')
    actionBtn.type = 'button'
    actionBtn.className = 'detail-description-action-link'
    actionBtn.dataset.detailAction = SOFTMAX_PANEL_ACTION_OPEN
    actionBtn.textContent = 'Explain softmax'
    actionBtn.setAttribute('aria-label', 'Open the softmax explainer')

    actionRow.appendChild(actionBtn)
    descriptionEl.appendChild(actionRow)
}

export function resolveSoftmaxPreviewContext(selectionInfo = null) {
    return isPostSoftmaxAttentionSelection(selectionInfo) ? {} : null
}

export function getSoftmaxCopyContextContent() {
    return {
        panelContentsBlurb: SOFTMAX_COPY_PANEL_BLURB,
        descriptionText: [
            ...SOFTMAX_OVERVIEW_LINES,
            `What it does: ${SOFTMAX_WHAT_IT_DOES}`,
            `Why attention uses it: ${SOFTMAX_WHY_ATTENTION}`
        ].join('\n\n'),
        equationText: `$$${SOFTMAX_EQUATION_TEX}$$`
    }
}

export function createSoftmaxDetailView(panelEl) {
    if (!panelEl || typeof document === 'undefined') return null

    const root = document.createElement('section')
    root.className = 'detail-softmax-view'
    root.hidden = true
    root.setAttribute('aria-hidden', 'true')
    root.innerHTML = `
        <div class="detail-description detail-softmax-copy">
            <p>${SOFTMAX_OVERVIEW_LINES[0]}</p>
            <p>${SOFTMAX_OVERVIEW_LINES[1]}</p>
        </div>
        <div class="detail-equations detail-softmax-equation-shell is-visible" aria-hidden="false">
            <div class="detail-equations-title">Equation</div>
            <div class="detail-equations-body detail-softmax-formula" data-softmax-formula></div>
        </div>
        <div class="detail-data detail-softmax-note-card">
            <div class="detail-data-title">What It Does</div>
            <p class="detail-softmax-note">
                ${SOFTMAX_WHAT_IT_DOES}
            </p>
        </div>
        <div class="detail-data detail-softmax-note-card">
            <div class="detail-data-title">Why Attention Uses It</div>
            <p class="detail-softmax-note">
                ${SOFTMAX_WHY_ATTENTION}
            </p>
        </div>
    `

    const header = panelEl.querySelector('.detail-header')
    if (header && header.nextSibling) {
        panelEl.insertBefore(root, header.nextSibling)
    } else {
        panelEl.appendChild(root)
    }

    const formulaEl = root.querySelector('[data-softmax-formula]')

    function renderFormula() {
        if (!formulaEl) return
        const markup = renderFormulaMarkup(SOFTMAX_EQUATION_TEX)
        if (markup === SOFTMAX_EQUATION_TEX) {
            formulaEl.textContent = SOFTMAX_EQUATION_TEX
        } else {
            formulaEl.innerHTML = markup
        }
    }

    return {
        root,
        setContext() {
            // This explainer is intentionally generic rather than selection-specific.
        },
        setVisible(visible) {
            const nextVisible = visible === true
            root.hidden = !nextVisible
            root.classList.toggle('is-visible', nextVisible)
            root.setAttribute('aria-hidden', nextVisible ? 'false' : 'true')
            if (!nextVisible) return
            renderFormula()
        },
        resizeAndRender() {
            if (root.hidden) return
            renderFormula()
        },
        remove() {
            root.remove()
        }
    }
}
