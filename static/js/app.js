/**
 * DocuMind RAG — Frontend Application Controller
 * Handles document upload, semantic query submission,
 * citation rendering, and grounded fact-checking audit inspection.
 */

let currentDocId = null;
let currentDocName = "";
let lastFactCheckData = null;

// DOM Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("pdf-file-input");
const uploadProgressWrap = document.getElementById("upload-progress-wrap");
const uploadStatusText = document.getElementById("upload-status-text");

const activeDocCard = document.getElementById("active-doc-card");
const docNameDisplay = document.getElementById("doc-name-display");
const docSubstatsDisplay = document.getElementById("doc-substats-display");
const statPages = document.getElementById("stat-pages");
const statChunks = document.getElementById("stat-chunks");
const statChars = document.getElementById("stat-chars");
const scannedWarningBox = document.getElementById("scanned-warning-box");
const textPreviewBox = document.getElementById("text-preview-box");
const textPreviewContent = document.getElementById("text-preview-content");
const activeDocsBadge = document.getElementById("active-docs-badge");

const btnRemoveDoc = document.getElementById("btn-remove-doc");
const btnTogglePreview = document.getElementById("btn-toggle-preview");
const btnViewChunks = document.getElementById("btn-view-chunks");

const chatStream = document.getElementById("chat-stream");
const emptyState = document.getElementById("empty-state");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const btnSubmitQuestion = document.getElementById("btn-submit-question");
const btnClearChat = document.getElementById("btn-clear-chat");

// Fact-Check Modal Elements
const factModal = document.getElementById("fact-modal");
const btnCloseModal = document.getElementById("btn-close-modal");
const scoreNumber = document.getElementById("score-number");
const scoreVerdictTitle = document.getElementById("score-verdict-title");
const scoreVerdictDesc = document.getElementById("score-verdict-desc");
const modalClaimsList = document.getElementById("modal-claims-list");
const modalAuditSummary = document.getElementById("modal-audit-summary");
const modalCitationPills = document.getElementById("modal-citation-pills");

// Chunks Modal Elements
const chunksModal = document.getElementById("chunks-modal");
const btnCloseChunksModal = document.getElementById("btn-close-chunks-modal");
const chunksDisplayList = document.getElementById("chunks-display-list");
const chunksModalSubtitle = document.getElementById("chunks-modal-subtitle");

// ============================================================================
// Initialization & Event Listeners
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
    setupUploadHandlers();
    setupChatHandlers();
    setupModalHandlers();
});

function setupUploadHandlers() {
    dropzone.addEventListener("click", (e) => {
        if (e.target !== fileInput) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag and drop events
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            handleFileUpload(dt.files[0]);
        }
    });

    btnRemoveDoc.addEventListener("click", removeActiveDocument);
    btnTogglePreview.addEventListener("click", () => {
        textPreviewBox.style.display = textPreviewBox.style.display === "none" ? "block" : "none";
    });
    btnViewChunks.addEventListener("click", openChunksModal);
}

function setupChatHandlers() {
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitQuestion();
    });

    btnClearChat.addEventListener("click", () => {
        chatStream.innerHTML = "";
        chatStream.appendChild(emptyState);
        emptyState.style.display = "flex";
    });
}

function setupModalHandlers() {
    btnCloseModal.addEventListener("click", () => {
        factModal.style.display = "none";
    });
    factModal.addEventListener("click", (e) => {
        if (e.target === factModal) factModal.style.display = "none";
    });

    btnCloseChunksModal.addEventListener("click", () => {
        chunksModal.style.display = "none";
    });
    chunksModal.addEventListener("click", (e) => {
        if (e.target === chunksModal) chunksModal.style.display = "none";
    });
}

// Global chip helper for sample questions
window.setQuery = function(text) {
    questionInput.value = text;
    questionInput.focus();
};

// ============================================================================
// PDF Upload & Processing
// ============================================================================

async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        alert("Please upload a valid PDF document.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    // Show upload progress UI
    dropzone.querySelector(".dropzone-content").style.display = "none";
    uploadProgressWrap.style.display = "flex";
    uploadStatusText.textContent = "Extracting text & generating vector embeddings...";

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Failed to upload and process PDF.");
        }

        // Successfully processed
        currentDocId = data.doc_id;
        currentDocName = data.filename;

        // Update Document card UI
        docNameDisplay.textContent = data.filename;
        docNameDisplay.title = data.filename;
        docSubstatsDisplay.textContent = `${data.total_pages} page(s) • ${data.total_chunks} FAISS chunks`;
        statPages.textContent = data.total_pages;
        statChunks.textContent = data.total_chunks;
        statChars.textContent = (data.total_characters || 0).toLocaleString();
        textPreviewContent.textContent = data.preview || "(Empty)";

        if (data.is_scanned) {
            scannedWarningBox.style.display = "block";
        } else {
            scannedWarningBox.style.display = "none";
        }

        activeDocCard.style.display = "flex";
        activeDocsBadge.textContent = "1 Active";
        questionInput.placeholder = `Ask anything about '${data.filename}'...`;
        questionInput.focus();

        // Add a system welcome row into the chat
        appendSystemMessage(`📄 <b>${escapeHtml(data.filename)}</b> ready! Indexed ${data.total_chunks} passages across ${data.total_pages} page(s). You can now ask questions.`);

    } catch (err) {
        alert("Upload error: " + err.message);
    } finally {
        dropzone.querySelector(".dropzone-content").style.display = "block";
        uploadProgressWrap.style.display = "none";
        fileInput.value = "";
    }
}

async function removeActiveDocument() {
    if (!currentDocId) return;
    if (!confirm(`Are you sure you want to remove '${currentDocName}' and clear the FAISS index?`)) return;

    try {
        await fetch(`/api/document/${currentDocId}`, { method: "DELETE" });
    } catch (e) {
        console.warn("Delete request error:", e);
    }

    currentDocId = null;
    currentDocName = "";
    activeDocCard.style.display = "none";
    activeDocsBadge.textContent = "0 Active";
    questionInput.placeholder = "Ask a question about the document...";
    appendSystemMessage("🗑️ Document removed. Vector index cleared.");
}

// ============================================================================
// Question & Grounded Answer Handling
// ============================================================================

async function submitQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    if (!currentDocId) {
        alert("Please upload a PDF document first so DocuMind can ground answers in its text.");
        return;
    }

    // Hide empty state if visible
    emptyState.style.display = "none";

    // 1. Render User Message
    appendUserMessage(question);
    questionInput.value = "";

    // 2. Render AI Loading Placeholder
    const aiMessageId = "msg-" + Date.now();
    appendLoadingMessage(aiMessageId);

    // Disable send button while awaiting response
    btnSubmitQuestion.disabled = true;
    btnSubmitQuestion.querySelector(".send-icon").style.display = "none";
    btnSubmitQuestion.querySelector(".send-spinner").style.display = "inline-block";

    try {
        const response = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                doc_id: currentDocId,
                question: question
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Error generating grounded answer.");
        }

        // Render actual answer with Fact-Checking audit
        updateAiMessage(aiMessageId, data);

    } catch (err) {
        updateAiMessageError(aiMessageId, err.message);
    } finally {
        btnSubmitQuestion.disabled = false;
        btnSubmitQuestion.querySelector(".send-icon").style.display = "inline-block";
        btnSubmitQuestion.querySelector(".send-spinner").style.display = "none";
        chatStream.scrollTop = chatStream.scrollHeight;
    }
}

function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row message-user";
    row.innerHTML = `<div class="message-bubble">${escapeHtml(text)}</div>`;
    chatStream.appendChild(row);
    chatStream.scrollTop = chatStream.scrollHeight;
}

function appendSystemMessage(htmlText) {
    const row = document.createElement("div");
    row.className = "message-row";
    row.innerHTML = `
        <div style="margin: 0 auto; background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 0.5rem 1rem; font-size: 0.78rem; color: #7DD3FC;">
            ${htmlText}
        </div>
    `;
    chatStream.appendChild(row);
    chatStream.scrollTop = chatStream.scrollHeight;
}

function appendLoadingMessage(msgId) {
    const row = document.createElement("div");
    row.className = "message-row message-ai";
    row.id = msgId;
    row.innerHTML = `
        <div class="ai-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
        </div>
        <div class="ai-content-wrap">
            <div class="ai-card">
                <div style="display: flex; align-items: center; gap: 0.65rem; color: var(--text-muted); font-size: 0.85rem;">
                    <div class="spinner" style="width: 18px; height: 18px; border-width: 2px;"></div>
                    <span>Retrieving FAISS passages & running fact-check audit...</span>
                </div>
            </div>
        </div>
    `;
    chatStream.appendChild(row);
    chatStream.scrollTop = chatStream.scrollHeight;
}

function updateAiMessage(msgId, data) {
    const row = document.getElementById(msgId);
    if (!row) return;

    const factCheck = data.fact_check || {};
    const badgeClass = factCheck.badge_color ? `badge-${factCheck.badge_color}` : "badge-success";
    const badgeText = factCheck.badge || "Fact-Checked: Grounded";

    // Build citations markup
    let citationsHtml = "";
    if (data.citations && data.citations.length > 0) {
        const pills = data.citations.map((c, i) => {
            return `<button class="citation-pill" onclick="inspectCitation(${i}, '${encodeURIComponent(JSON.stringify(c))}')" title="${escapeHtml(c.snippet)}">
                📄 Page ${c.page_number} (sim: ${c.score})
            </button>`;
        }).join(" ");

        citationsHtml = `
            <div class="citations-footer">
                <span class="citations-label">Sources:</span>
                ${pills}
            </div>
        `;
    }

    // Attach fact check data globally for the inspector
    const factCheckJson = encodeURIComponent(JSON.stringify(factCheck));

    row.innerHTML = `
        <div class="ai-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
        </div>
        <div class="ai-content-wrap">
            <div class="ai-card">
                <!-- Fact-Check Top Badge -->
                <div class="fact-check-badge-bar">
                    <span class="fact-badge ${badgeClass}">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                        </svg>
                        ${escapeHtml(badgeText)}
                    </span>
                    <button class="btn-inspect-fact" onclick="openFactCheckModal('${factCheckJson}')">
                        Inspect Fact-Check & Claims ↗
                    </button>
                </div>

                <!-- Answer Content -->
                <div class="ai-answer-text">${formatAnswer(data.answer)}</div>

                <!-- Citations -->
                ${citationsHtml}
            </div>

            <div class="ai-meta-footer">
                <span>Model: ${escapeHtml(data.model_used || "Gemini")} ${data.cached ? "(Cached)" : ""}</span>
                <span>Latency: ${data.latency_ms || 0}ms</span>
            </div>
        </div>
    `;
}

function updateAiMessageError(msgId, errorMsg) {
    const row = document.getElementById(msgId);
    if (!row) return;

    row.innerHTML = `
        <div class="ai-avatar" style="background: var(--danger-red);">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
        </div>
        <div class="ai-content-wrap">
            <div class="ai-card" style="border-color: rgba(239, 68, 68, 0.4);">
                <div style="color: var(--danger-red); font-size: 0.88rem; font-weight: 600; margin-bottom: 0.35rem;">
                    ⚠️ Error Processing Request
                </div>
                <p style="font-size: 0.82rem; color: var(--text-muted);">${escapeHtml(errorMsg)}</p>
            </div>
        </div>
    `;
}

// ============================================================================
// Fact-Check Audit Inspector Modal
// ============================================================================

window.openFactCheckModal = function(encodedData) {
    const data = JSON.parse(decodeURIComponent(encodedData));
    lastFactCheckData = data;

    const scorePct = Math.round((data.groundedness_score || 0) * 100);
    scoreNumber.textContent = `${scorePct}%`;
    modalAuditSummary.textContent = data.summary || "Audit results for this answer";

    // Verdict styling
    if (data.status === "VERIFIED") {
        scoreVerdictTitle.textContent = "Fully Verified & Grounded";
        scoreVerdictDesc.textContent = "All claims and numbers are verified in retrieved PDF passages.";
    } else if (data.status === "PARTIALLY_VERIFIED") {
        scoreVerdictTitle.textContent = "Partially Grounded";
        scoreVerdictDesc.textContent = "Most claims match, but some statements have moderate support.";
    } else if (data.status === "HONEST_REFUSAL") {
        scoreVerdictTitle.textContent = "Honest Grounded Refusal";
        scoreVerdictDesc.textContent = "The model correctly refrained from answering what is missing from the document.";
    } else {
        scoreVerdictTitle.textContent = "Potential Unverified Claims";
        scoreVerdictDesc.textContent = "Some claims lack direct supporting evidence in the source text.";
    }

    // Render citations summary in modal
    const citAudit = data.citation_audit || {};
    let citHtml = "";
    if (citAudit.cited_pages && citAudit.cited_pages.length > 0) {
        citHtml = citAudit.cited_pages.map(p => {
            const isValid = citAudit.valid_citations.includes(p);
            return `<span class="claim-status-tag ${isValid ? 'tag-supported' : 'tag-unsupported'}">
                ${isValid ? '✓' : '✗'} Page ${p} ${isValid ? 'Verified' : 'Invalid Page'}
            </span>`;
        }).join(" ");
    }
    modalCitationPills.innerHTML = citHtml;

    // Render Claim by Claim List
    const claims = data.claims || [];
    modalClaimsList.innerHTML = claims.map(c => {
        let tagClass = "tag-supported";
        let tagText = "SUPPORTED";

        if (c.status === "PARTIALLY_SUPPORTED") {
            tagClass = "tag-partial";
            tagText = "PARTIAL MATCH";
        } else if (c.status === "UNSUPPORTED") {
            tagClass = "tag-unsupported";
            tagText = "UNVERIFIED";
        }

        const pageLabel = c.matched_page ? `Page ${c.matched_page}` : "No specific page";

        return `
            <div class="claim-item">
                <div class="claim-header">
                    <p class="claim-text">"${escapeHtml(c.claim)}"</p>
                    <span class="claim-status-tag ${tagClass}">${tagText} (${Math.round((c.confidence || 0) * 100)}%)</span>
                </div>
                <div class="evidence-box">
                    <span class="evidence-page-ref">[${pageLabel}]</span>
                    <span>${escapeHtml(c.evidence_snippet)}</span>
                </div>
            </div>
        `;
    }).join("");

    factModal.style.display = "flex";
};

// ============================================================================
// Chunks Inspector Modal
// ============================================================================

async function openChunksModal() {
    if (!currentDocId) return;

    chunksModal.style.display = "flex";
    chunksDisplayList.innerHTML = `<div style="text-align: center; padding: 1.5rem;"><div class="spinner"></div></div>`;

    try {
        const resp = await fetch(`/api/document/${currentDocId}/chunks`);
        const data = await resp.json();

        if (!resp.ok) throw new Error(data.error || "Failed to load chunks");

        chunksModalSubtitle.textContent = `${data.chunks.length} chunks indexed in FAISS for '${currentDocName}'`;

        chunksDisplayList.innerHTML = data.chunks.map((c, i) => `
            <div class="chunk-card">
                <div class="chunk-card-meta">
                    <span>Chunk #${i + 1} (${c.chunk_id})</span>
                    <span>Page ${c.page_number} • ${c.char_count} chars</span>
                </div>
                <div class="chunk-card-body">${escapeHtml(c.text)}</div>
            </div>
        `).join("");

    } catch (e) {
        chunksDisplayList.innerHTML = `<p style="color: var(--danger-red);">Error: ${escapeHtml(e.message)}</p>`;
    }
}

window.inspectCitation = function(idx, encodedChunk) {
    const chunk = JSON.parse(decodeURIComponent(encodedChunk));
    alert(`Source Passage (Page ${chunk.page_number}, Similarity: ${chunk.score}):\n\n"${chunk.snippet}"`);
};

// ============================================================================
// Utilities
// ============================================================================

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatAnswer(text) {
    if (!text) return "";
    let safe = escapeHtml(text);
    // Convert [Page X] into styled span
    safe = safe.replace(/\[Page\s*(\d+)\]/gi, '<span style="color: #38BDF8; font-weight: 700; background: rgba(56, 189, 248, 0.15); padding: 0.1rem 0.35rem; border-radius: 4px;">[Page $1]</span>');
    // Convert bold **text**
    safe = safe.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    return safe;
}
