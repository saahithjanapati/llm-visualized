export function setupEquationsOverlay(pipeline, NUM_LAYERS, checkTopEmbeddingActivation) {
    const statusDiv = document.getElementById('statusOverlay');
    const equationsPanel = document.getElementById('equationsPanel');
    const equationsTitle = document.getElementById('equationsTitle');
    const equationsBody = document.getElementById('equationsBody');

    const EQ_STORAGE_KEY = 'showEquations';
    function getShowEquationsPref() {
        try {
            const v = localStorage.getItem(EQ_STORAGE_KEY);
            return v === null ? true : v === '1';
        } catch (_) { return true; }
    }
    function setShowEquationsPref(val) {
        try { localStorage.setItem(EQ_STORAGE_KEY, val ? '1' : '0'); } catch (_) {}
    }

    let showEquations = getShowEquationsPref();
    if (equationsPanel) equationsPanel.style.display = showEquations ? 'block' : 'none';

    const EQ = {
        ln1: String.raw`x_{\text{ln}} = \mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2+\epsilon}} + \beta` ,
        qkv_per_head: String.raw`Q_k = x_{\text{ln}} W_k^Q,\quad K_k = x_{\text{ln}} W_k^K,\quad V_k = x_{\text{ln}} W_k^V\quad (k=1..h)`,
        qkv_packed: String.raw`Q = x_{\text{ln}} W^Q,\quad K = x_{\text{ln}} W^K,\quad V = x_{\text{ln}} W^V,\quad Q,K,V\in\mathbb{R}^{B\times T\times h\times d_h}`,
        attn: String.raw`A_k = \frac{Q_k K_k^\top}{\sqrt{d_h}} + M_{\text{causal}},\quad \alpha_k = \mathrm{softmax}(A_k)\ (\text{row-wise}),\quad H_k = \alpha_k V_k`,
        concat_proj: String.raw`H = \mathrm{Concat}(H_1,\ldots,H_h)\in\mathbb{R}^{B\times T\times d},\quad \mathrm{SA}(x) = H W^O + b_o`,
        resid1: String.raw`u = x + \mathrm{SA}(x_{\text{ln}})`,
        ln2: String.raw`u_{\text{ln}} = \mathrm{LN}(u) = \gamma' \odot \frac{u - \mu'}{\sqrt{{\sigma'}^{2}+\epsilon}} + \beta'`,
        mlp: String.raw`z = u_{\text{ln}} W_{\text{up}} + b_{\text{up}},\quad z' = \mathrm{GELU}(z),\quad \mathrm{MLP}(u_{\text{ln}}) = z' W_{\text{down}} + b_{\text{down}}`,
        resid2: String.raw`x_{\text{out}} = u + \mathrm{MLP}(u_{\text{ln}})`
    };

    function renderEq(tex, title) {
        if (!equationsPanel || !equationsBody) return;
        if (!showEquations) return;
        equationsTitle.textContent = title || 'Equations';
        try {
            if (window.katex && window.katex.render) {
                equationsBody.innerHTML = '';
                window.katex.render(tex, equationsBody, { throwOnError: false, displayMode: true });
            } else {
                equationsBody.textContent = tex;
            }
        } catch (_) {
            equationsBody.textContent = tex;
        }
    }

    let __lastEqKey = '';
    function updateEquations(layer) {
        if (!showEquations) return;
        if (!layer) return;
        const lanes = Array.isArray(layer.lanes) ? layer.lanes : [];
        if (!lanes.length) return;

        const mlpActive = lanes.some(l => l.mlpUpStarted || l.ln2Phase === 'mlpReady' || l.ln2Phase === 'done');
        const ln2Active = !mlpActive && lanes.some(l => l.ln2Phase && l.ln2Phase !== 'notStarted');
        const ln1Active = !mlpActive && !ln2Active && lanes.some(l => ['waiting','right','insideLN'].includes(l.horizPhase));
        const mhsaActive = !mlpActive && !ln2Active && !ln1Active && (
            (layer && layer._mhsaStart === true) ||
            lanes.some(l => ['riseAboveLN','readyMHSA','travelMHSA','postMHSAAddition','waitingForLN2'].includes(l.horizPhase))
        );

        let key = '';
        let title = '';

        if (mlpActive) {
            const adding = lanes.some(l => l.ln2Phase === 'done' && !l.additionComplete);
            key = adding ? 'resid2' : 'mlp';
            title = adding ? 'Residual Add 2' : 'MLP (FFN)';
        } else if (ln2Active) {
            key = 'ln2';
            title = 'LayerNorm 2';
        } else if (mhsaActive) {
            const m = layer.mhsaAnimation;
            const phase = m && m.mhaPassThroughPhase;
            const outPhase = m && m.outputProjMatrixAnimationPhase;
            if (phase === 'parallel_pass_through_active') {
                key = 'attn';
                title = 'Scaled Dot-Product Attention';
            } else if (phase === 'mha_pass_through_complete') {
                const postAdd = lanes.some(l => ['postMHSAAddition','waitingForLN2'].includes(l.horizPhase));
                if (postAdd || outPhase === 'vectors_inside' || outPhase === 'completed') {
                    key = 'resid1';
                    title = 'Residual Add 1';
                } else if (m?.rowMergePhase === 'merging' || outPhase === 'vectors_entering') {
                    key = 'concat_proj';
                    title = 'Concat Heads + W^O';
                } else {
                    key = 'attn';
                    title = 'Scaled Dot-Product Attention';
                }
            } else {
                key = 'qkv_per_head';
                title = 'Q/K/V Projections';
            }
        } else if (ln1Active || layer.isActive) {
            key = 'ln1';
            title = 'LayerNorm 1';
        }

        if (!key) return;
        if (key === __lastEqKey) return;
        __lastEqKey = key;
        const t = (key === 'ln1') ? 'LayerNorm 1 (no in-place assign)' : title;
        renderEq(EQ[key], t);
    }

    function updateStatus() {
        if (!statusDiv) return;
        const idx = pipeline._currentLayerIdx;
        const total = NUM_LAYERS;
        const layer = pipeline._layers[idx];
        let displayStage = 'Loading...';

        if (layer && Array.isArray(layer.lanes) && layer.lanes.length) {
            const lanes = layer.lanes;
            const mlpActive = lanes.some(l => l.mlpUpStarted || l.ln2Phase === 'mlpReady' || l.ln2Phase === 'done');
            const ln2Active = !mlpActive && lanes.some(l => l.ln2Phase && l.ln2Phase !== 'notStarted');
            const ln1Active = !mlpActive && !ln2Active && lanes.some(l => ['waiting', 'right', 'insideLN'].includes(l.horizPhase));
            const mhsaActive = !mlpActive && !ln2Active && !ln1Active && (
                (layer && layer._mhsaStart === true) ||
                lanes.some(l => ['riseAboveLN', 'readyMHSA', 'travelMHSA', 'postMHSAAddition', 'waitingForLN2'].includes(l.horizPhase))
            );

            if (mlpActive) {
                displayStage = 'Multi-Layer Perceptron Block';
            } else if (ln2Active) {
                displayStage = 'LayerNorm 2';
            } else if (ln1Active) {
                displayStage = 'LayerNorm 1';
            } else if (mhsaActive) {
                displayStage = 'Multi Head Attention';
            } else if (layer.isActive) {
                displayStage = 'LayerNorm 1';
            }
        }

        statusDiv.textContent = `Layer ${idx + 1} / ${total}\n${displayStage}`;
        try { updateEquations(layer); } catch (_) {}
        try { checkTopEmbeddingActivation(); } catch (_) {}
    }

    setInterval(updateStatus, 250);

    function setShowEquations(val) {
        showEquations = val;
        setShowEquationsPref(val);
        if (equationsPanel) equationsPanel.style.display = val ? 'block' : 'none';
        __lastEqKey = '';
    }
    function getShowEquations() { return showEquations; }

    return { setShowEquations, getShowEquations };
}
