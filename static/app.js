function initThemeToggle() {
    var root = document.documentElement;
    var toggle = document.getElementById('themeToggle');
    var label = document.getElementById('themeLabel');
    function isDark() { return root.classList.contains('theme-dark'); }
    function updateUI() {
        if (toggle) {
            toggle.setAttribute('aria-pressed', isDark() ? 'true' : 'false');
            toggle.setAttribute('aria-label', isDark() ? 'Switch to light mode' : 'Switch to dark mode');
        }
        if (label) label.textContent = isDark() ? 'Light mode' : 'Dark mode';
    }
    if (toggle) {
        toggle.addEventListener('click', function() {
            root.classList.toggle('theme-dark');
            localStorage.setItem('theme', isDark() ? 'dark' : 'light');
            updateUI();
        });
    }
    updateUI();
}

let chats = [];
let activeChatId = null;
let nextChatId = 1;
let nextMessageId = 1;
let docChatHistory = [];
const NO_RAG_OPTION_VALUE = 'NO_RAG';
const WEB_RETRIEVAL_OPTION_VALUE = 'WEB_RETRIEVAL';
const SESSION_STORAGE_KEY = 'image_generator_session_id';
function generateClientSessionId() {
    const randomPart = Math.random().toString(36).slice(2, 10);
    return 'sess_' + Date.now().toString(36) + '_' + randomPart;
}

let CLIENT_SESSION_ID = generateClientSessionId();

function withActiveChatId(payload) {
    return {
        ...(payload || {}),
        session_id: CLIENT_SESSION_ID,
    };
}

const DEFAULT_ASPECT_RATIO = '16:9';
const ALLOWED_ASPECT_RATIOS = ['1:1', '4:3', '3:4', '16:9', '9:16', '3:2', '2:3', '21:9'];

function getSelectedAspectRatio() {
    const el = document.getElementById('aspectRatio');
    const v = el && el.value ? String(el.value).trim() : '';
    return ALLOWED_ASPECT_RATIOS.includes(v) ? v : DEFAULT_ASPECT_RATIO;
}

async function resetServerSession(sessionId) {
    if (!sessionId) return;
    try {
        await fetch('/session/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId }),
            keepalive: true,
        });
    } catch (error) {
        // Best-effort cleanup; ignore network issues during tab close/reload.
    }
}

function isNoRagSelected(docNames) {
    const selected = Array.isArray(docNames) ? docNames : getSelectedDocNames();
    return selected.includes(NO_RAG_OPTION_VALUE);
}

function updateRetrievalActionsState() {
    const noRag = isNoRagSelected();
    const errorEl = document.getElementById('ragRetrievalError');

    if (errorEl && noRag) {
        errorEl.textContent = 'NO RAG is selected. No document context will be retrieved.';
        errorEl.classList.remove('hidden');
    } else if (errorEl) {
        errorEl.textContent = '';
        errorEl.classList.add('hidden');
    }
}

function renderDocChatHistory() {
    const container = document.getElementById('docChatMessages');
    if (!container) return;

    if (!docChatHistory.length) {
        container.innerHTML = '<p class="doc-chat-empty">Your messages will show up here. Enter a question below to begin.</p>';
        return;
    }

    container.innerHTML = '';
    docChatHistory.forEach(entry => {
        const row = document.createElement('div');
        row.className = 'doc-chat-row ' + (entry.role === 'user' ? 'doc-chat-user' : 'doc-chat-assistant');

        const roleLabel = document.createElement('div');
        roleLabel.className = 'doc-chat-role';
        roleLabel.textContent = entry.role === 'user' ? 'You' : 'Answer';

        const content = document.createElement('div');
        content.className = 'doc-chat-content';
        content.textContent = entry.text;

        row.appendChild(roleLabel);
        row.appendChild(content);
        container.appendChild(row);
    });
    container.scrollTop = container.scrollHeight;
}

function clearDocChatHistory() {
    docChatHistory = [];
    const errorEl = document.getElementById('docChatError');
    if (errorEl) {
        errorEl.textContent = '';
        errorEl.classList.add('hidden');
    }
    renderDocChatHistory();
}

function buildDocChatHistoryContext() {
    if (!Array.isArray(docChatHistory) || docChatHistory.length === 0) {
        return '';
    }
    return docChatHistory
        .map(entry => {
            const role = entry && entry.role === 'assistant' ? 'Assistant' : 'User';
            const text = entry && entry.text ? String(entry.text) : '';
            return role + ': ' + text;
        })
        .join('\n');
}

async function askDocsQuestion() {
    const inputEl = document.getElementById('docChatQuestion');
    const loadingEl = document.getElementById('docChatLoading');
    const errorEl = document.getElementById('docChatError');
    const askBtn = document.getElementById('askDocsBtn');

    const question = inputEl ? inputEl.value.trim() : '';
    if (!question) {
        if (errorEl) {
            errorEl.textContent = 'Enter a question first.';
            errorEl.classList.remove('hidden');
        }
        return;
    }

    const selectedDocNames = getSelectedDocNames();
    const chatHistoryContext = buildDocChatHistoryContext();
    if (isNoRagSelected(selectedDocNames)) {
        if (errorEl) {
            errorEl.textContent = 'Select at least one document or source. Turn off the option “Don\u2019t use my documents” to get answers from your materials.';
            errorEl.classList.remove('hidden');
        }
        return;
    }

    docChatHistory.push({ role: 'user', text: question });
    renderDocChatHistory();
    if (inputEl) inputEl.value = '';

    if (loadingEl) loadingEl.classList.remove('hidden');
    if (errorEl) {
        errorEl.textContent = '';
        errorEl.classList.add('hidden');
    }
    if (askBtn) askBtn.disabled = true;

    try {
        const response = await fetch('/chat-with-docs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...withActiveChatId({
                    user_question: question,
                    selected_doc_names: selectedDocNames,
                    chat_history: chatHistoryContext
                })
            })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to get answer');
        }

        docChatHistory.push({
            role: 'assistant',
            text: String(data.answer || 'No answer generated.')
        });
        renderDocChatHistory();

        if (Array.isArray(data.selected_doc_names)) {
            setSelectedDocNames(data.selected_doc_names);
        }
    } catch (error) {
        docChatHistory.push({
            role: 'assistant',
            text: 'Error: ' + (error.message || 'Unknown error')
        });
        renderDocChatHistory();
        if (errorEl) {
            errorEl.textContent = 'Error: ' + (error.message || 'Unknown error');
            errorEl.classList.remove('hidden');
        }
    } finally {
        if (loadingEl) loadingEl.classList.add('hidden');
        if (askBtn) askBtn.disabled = false;
    }
}

function normalizeDocSelection() {
    const baseSelect = document.getElementById('ragDocNameSelect');
    const sessionSelect = document.getElementById('ragSessionDocNameSelect');
    if (!baseSelect) return;

    const selectedValues = Array.from(baseSelect.selectedOptions || []).map(option => option.value);
    if (selectedValues.includes(NO_RAG_OPTION_VALUE)) {
        Array.from(baseSelect.options || []).forEach(option => {
            option.selected = option.value === NO_RAG_OPTION_VALUE;
        });
        if (sessionSelect) {
            Array.from(sessionSelect.options || []).forEach(option => {
                option.selected = false;
            });
        }
    }

    updateRetrievalActionsState();
}

function getSelectedDocNames() {
    const baseSelect = document.getElementById('ragDocNameSelect');
    const sessionSelect = document.getElementById('ragSessionDocNameSelect');
    const baseValues = baseSelect
        ? Array.from(baseSelect.selectedOptions || [])
            .map(option => option.value)
            .filter(value => !!value)
            .map(value => {
                if (value === NO_RAG_OPTION_VALUE || value === WEB_RETRIEVAL_OPTION_VALUE) {
                    return value;
                }
                return value.startsWith('base::') ? value : ('base::' + value);
            })
        : [];

    const sessionValues = sessionSelect
        ? Array.from(sessionSelect.selectedOptions || [])
            .map(option => option.value)
            .filter(value => !!value)
            .map(value => value.startsWith('session::') ? value : ('session::' + value))
        : [];

    return [...baseValues, ...sessionValues].filter(value => !!value);
}

function setSelectedDocNames(docNames) {
    const baseSelect = document.getElementById('ragDocNameSelect');
    const sessionSelect = document.getElementById('ragSessionDocNameSelect');
    if (!baseSelect) return;
    const selectedSet = new Set(Array.isArray(docNames) ? docNames : []);
    const noRagOnly = selectedSet.has(NO_RAG_OPTION_VALUE);
    Array.from(baseSelect.options || []).forEach(option => {
        const baseToken = option.value.startsWith('base::')
            ? option.value
            : (option.value === NO_RAG_OPTION_VALUE || option.value === WEB_RETRIEVAL_OPTION_VALUE
                ? option.value
                : ('base::' + option.value));
        option.selected = noRagOnly
            ? option.value === NO_RAG_OPTION_VALUE
            : selectedSet.has(baseToken) || selectedSet.has(option.value);
    });
    if (sessionSelect) {
        Array.from(sessionSelect.options || []).forEach(option => {
            const sessionToken = option.value.startsWith('session::')
                ? option.value
                : ('session::' + option.value);
            option.selected = !noRagOnly && (selectedSet.has(sessionToken) || selectedSet.has(option.value));
        });
    }
    normalizeDocSelection();
}

async function loadDocNames() {
    const select = document.getElementById('ragDocNameSelect');
    const sessionSelect = document.getElementById('ragSessionDocNameSelect');
    const errorEl = document.getElementById('ragRetrievalError');
    if (!select || !sessionSelect) return;

    const previousSelection = getSelectedDocNames();
    select.innerHTML = '';

    try {
        const response = await fetch('/doc-names?session_id=' + encodeURIComponent(CLIENT_SESSION_ID));
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Failed to load document names');
        }

        const docNames = Array.isArray(data.base_doc_names || data.doc_names)
            ? (data.base_doc_names || data.doc_names)
            : [];
        const sessionDocNames = Array.isArray(data.session_doc_names)
            ? data.session_doc_names
            : [];

        const noRagOption = document.createElement('option');
        noRagOption.value = NO_RAG_OPTION_VALUE;
        noRagOption.textContent = "Don't use my documents";
        select.appendChild(noRagOption);

        const webRetrievalOption = document.createElement('option');
        webRetrievalOption.value = WEB_RETRIEVAL_OPTION_VALUE;
        webRetrievalOption.textContent = 'Include web search';
        select.appendChild(webRetrievalOption);

        if (docNames.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No library PDFs available';
            option.disabled = true;
            select.appendChild(option);
        } else {
            docNames.forEach(docName => {
                const option = document.createElement('option');
                option.value = docName;
                option.textContent = docName;
                select.appendChild(option);
            });
        }

        sessionSelect.innerHTML = '';
        if (sessionDocNames.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No PDF uploaded in this session yet';
            option.disabled = true;
            sessionSelect.appendChild(option);
        } else {
            sessionDocNames.forEach(docName => {
                const option = document.createElement('option');
                option.value = docName;
                option.textContent = docName;
                sessionSelect.appendChild(option);
            });
        }

        setSelectedDocNames(previousSelection);
        normalizeDocSelection();
        if (errorEl) {
            errorEl.textContent = '';
            errorEl.classList.add('hidden');
        }
    } catch (error) {
        if (errorEl) {
            errorEl.textContent = 'Could not load document names: ' + (error.message || 'Unknown error');
            errorEl.classList.remove('hidden');
        }
    }
}

function createNewChat(name) {
    const id = nextChatId++;
    const chat = {
        id,
        name: name || `Chat ${id}`,
        history: [],
        currentImageUrl: null,
        currentImageDataUrl: null,
        imageCount: 0
    };
    chats.push(chat);
    activeChatId = id;
    updateChatSelector();
    renderConversation();
    updateDownloadButtonVisibility();
    updateChatImageCountDisplay();
    return chat;
}

function getActiveChat() {
    return chats.find(chat => chat.id === activeChatId) || null;
}

function getActiveChatImageCount() {
    const chat = getActiveChat();
    return chat && typeof chat.imageCount === 'number' ? chat.imageCount : 0;
}

function updateChatImageCountDisplay() {
    const countEl = document.getElementById('chatImageCount');

    if (!countEl) return;

    const count = getActiveChatImageCount();
    if (count === 0) {
        countEl.textContent = 'No images generated in this chat yet';
    } else if (count === 1) {
        countEl.textContent = '1 image in this chat';
    } else {
        countEl.textContent = `${count} images in this chat`;
    }
}

function updateChatSelector() {
    const selector = document.getElementById('chatSelector');
    if (!selector) return;

    selector.innerHTML = '';

    chats.forEach(chat => {
        const option = document.createElement('option');
        option.value = chat.id;
        option.textContent = chat.name;
        if (chat.id === activeChatId) {
            option.selected = true;
        }
        selector.appendChild(option);
    });
}

function handleChatChange(event) {
    const newId = Number(event.target.value);
    if (!newId || newId === activeChatId) return;
    activeChatId = newId;
    renderConversation();
    updateDownloadButtonVisibility();
    updateChatImageCountDisplay();
}

function startNewChat() {
    createNewChat();
    const generatedPromptEl = document.getElementById('generatedPrompt');
    if (generatedPromptEl) generatedPromptEl.value = '';
}

function addConversationEntry(entry) {
    let chat = getActiveChat();
    if (!chat) {
        chat = createNewChat();
    }
    const newEntry = { id: nextMessageId++, ...entry };
    chat.history.push(newEntry);
    renderConversation();
    return newEntry.id;
}

function truncateHistoryAfterIndex(chat, index) {
    if (!chat || !chat.history || index < 0) return;
    chat.history = chat.history.slice(0, index + 1);
    let lastImageUrl = null;
    let count = 0;
    chat.history.forEach(e => {
        if (e.imageUrl) {
            lastImageUrl = e.imageUrl;
            count++;
        }
    });
    chat.currentImageUrl = lastImageUrl;
    const lastImageEntry = [...chat.history].reverse().find(e => e && e.imageUrl);
    chat.currentImageDataUrl = lastImageEntry ? (lastImageEntry.imageDataUrl || null) : null;
    chat.imageCount = count;
    renderConversation();
    updateDownloadButtonVisibility();
    updateChatImageCountDisplay();
}

function getHistoryEntryByIndex(entryIndex) {
    const chat = getActiveChat();
    if (!chat || !chat.history || entryIndex < 0 || entryIndex >= chat.history.length) return null;
    return chat.history[entryIndex];
}

function getOriginalPromptForEntry(entryIndex) {
    const chat = getActiveChat();
    if (!chat || !chat.history || entryIndex < 0 || entryIndex >= chat.history.length) return '';

    const current = chat.history[entryIndex];
    if (current && current.sourcePrompt) return current.sourcePrompt;

    for (let i = entryIndex - 1; i >= 0; i--) {
        const prior = chat.history[i];
        if (prior && prior.role === 'user' && prior.type === 'prompt' && prior.text) {
            return prior.text;
        }
    }

    return '';
}

function renderConversation() {
    const container = document.getElementById('conversationContainer');
    if (!container) return;

    const chat = getActiveChat();
    const history = chat ? chat.history : [];

    if (!history || history.length === 0) {
        container.innerHTML = `<p style="color: #999;">Your prompts and generated images will appear here</p>`;
        return;
    }

    container.innerHTML = '';

    history.forEach((entry, idx) => {
        const row = document.createElement('div');
        row.className = `message-row ${entry.role === 'user' ? 'user' : 'assistant'}`;
        row.dataset.entryIndex = idx;

        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${entry.role === 'user' ? 'user' : 'assistant'}`;

        if (entry.role === 'user') {
            const textEl = document.createElement('div');
            textEl.className = 'message-text';
            textEl.textContent = entry.text;
            bubble.appendChild(textEl);
            if (entry.aspectRatio) {
                const tag = document.createElement('span');
                tag.className = 'aspect-ratio-tag';
                tag.textContent = 'Aspect: ' + entry.aspectRatio;
                bubble.appendChild(tag);
            }
            const editLink = document.createElement('button');
            editLink.type = 'button';
            editLink.className = 'message-edit-link';
            editLink.textContent = 'Edit';
            editLink.style.background = 'none';
            editLink.style.border = 'none';
            editLink.style.padding = '0';
            editLink.style.marginLeft = '0';
            editLink.onclick = () => toggleEditUserMessage(idx);
            bubble.appendChild(editLink);
            const editWrap = document.createElement('div');
            editWrap.className = 'message-text-edit-wrap';
            editWrap.setAttribute('style', 'display: none;');
            editWrap.innerHTML = '<textarea class="message-edit-textarea" rows="4" placeholder="Edit prompt or edit request..."></textarea><div class="inline-edit-actions"><button type="button" class="rag-btn rag-btn-primary" data-action="save-edit">Save &amp; regenerate</button><button type="button" class="rag-btn rag-btn-secondary" data-action="cancel-edit">Cancel</button></div>';
            const textarea = editWrap.querySelector('.message-edit-textarea');
            const saveBtn = editWrap.querySelector('[data-action="save-edit"]');
            const cancelBtn = editWrap.querySelector('[data-action="cancel-edit"]');
            saveBtn.onclick = () => saveEditedUserMessage(idx);
            cancelBtn.onclick = () => toggleEditUserMessage(idx);
            bubble.appendChild(editWrap);
        }

        if (entry.role === 'assistant' && entry.imageUrl) {
            const displaySrc = entry.imageDataUrl || entry.imageUrl;
            const img = document.createElement('img');
            img.src = displaySrc;
            img.className = 'message-image';
            img.alt = entry.type === 'edited_image' ? 'Edited image' : 'Generated image';
            img.onclick = () => openImageFullscreen(displaySrc, entry.sourcePrompt ? 'Prompt: ' + entry.sourcePrompt.slice(0, 80) + '…' : (entry.meta || 'Image'));
            bubble.appendChild(img);
            if (entry.meta) {
                const metaEl = document.createElement('div');
                metaEl.className = 'message-meta';
                metaEl.textContent = entry.meta;
                bubble.appendChild(metaEl);
            }
            if (entry.accuracyTrace && entry.accuracyTrace.length) {
                const tracePanel = buildAccuracyTracePanel(entry.accuracyTrace);
                if (tracePanel) bubble.appendChild(tracePanel);
            }
            const inlineEdit = document.createElement('div');
            inlineEdit.className = 'message-edit-inline';
            inlineEdit.innerHTML = '<div class="message-edit-label">Suggest changes to this image</div><textarea class="inline-changes-textarea" rows="2" placeholder="e.g., zoom on lesion, adjust lighting..."></textarea><div class="inline-edit-actions inline-accurate-actions"><button type="button" class="rag-btn rag-btn-primary inline-apply-changes-btn">Apply changes</button><button type="button" class="rag-btn rag-btn-secondary inline-get-accurate-btn">Get Accurate</button><button type="button" class="rag-btn rag-btn-secondary inline-get-accurate-trace-btn">Get Accurate + log</button></div>';
            const changeTa = inlineEdit.querySelector('.inline-changes-textarea');
            const applyBtn = inlineEdit.querySelector('.inline-apply-changes-btn');
            const accurateBtn = inlineEdit.querySelector('.inline-get-accurate-btn');
            const accurateTraceBtn = inlineEdit.querySelector('.inline-get-accurate-trace-btn');
            applyBtn.onclick = () => applyChangesToImage(idx, changeTa.value.trim());
            accurateBtn.onclick = () => getAccurateImage(idx);
            accurateTraceBtn.onclick = () => getAccurateImage(idx, { includeTrace: true });
            bubble.appendChild(inlineEdit);
        }

        row.appendChild(bubble);
        container.appendChild(row);
    });

    container.scrollTop = container.scrollHeight;
}

function toggleEditUserMessage(entryIndex) {
    const chat = getActiveChat();
    if (!chat || !chat.history) return;
    const entry = chat.history[entryIndex];
    if (!entry || entry.role !== 'user') return;
    const row = document.querySelector(`[data-entry-index="${entryIndex}"]`);
    if (!row) return;
    const bubble = row.querySelector('.message-bubble');
    const textEl = bubble.querySelector('.message-text');
    const editLink = bubble.querySelector('.message-edit-link');
    const editWrap = bubble.querySelector('.message-text-edit-wrap');
    const textarea = editWrap.querySelector('.message-edit-textarea');
    if (editWrap.style.display === 'none') {
        textarea.value = entry.text;
        editWrap.style.display = 'block';
        textEl.style.display = 'none';
        editLink.style.display = 'none';
    } else {
        editWrap.style.display = 'none';
        textEl.style.display = '';
        editLink.style.display = '';
    }
}

function saveEditedUserMessage(entryIndex) {
    const chat = getActiveChat();
    if (!chat || !chat.history) return;
    const entry = chat.history[entryIndex];
    if (!entry || entry.role !== 'user') return;
    const row = document.querySelector(`[data-entry-index="${entryIndex}"]`);
    if (!row) return;
    const textarea = row.querySelector('.message-edit-textarea');
    const newText = (textarea && textarea.value && textarea.value.trim()) || entry.text;
    entry.text = newText;
    truncateHistoryAfterIndex(chat, entryIndex);
    if (entry.type === 'prompt') {
        regenerateFromPromptMessage(entryIndex);
    } else if (entry.type === 'edit_request') {
        regenerateFromEditMessage(entryIndex);
    }
}

async function regenerateFromPromptMessage(entryIndex) {
    const chat = getActiveChat();
    if (!chat || !chat.history) return;
    const entry = chat.history[entryIndex];
    if (!entry || entry.role !== 'user' || entry.type !== 'prompt') return;
    const prompt = entry.text;
    const loading = document.getElementById('imageLoading');
    const errorDiv = document.getElementById('imageError');
    const successDiv = document.getElementById('imageSuccess');
    loading.classList.add('active');
    errorDiv.classList.remove('active');
    successDiv.classList.remove('active');
    try {
        const aspectRatio = getSelectedAspectRatio();
        const response = await fetch('/generate-image', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(withActiveChatId({ prompt, aspect_ratio: aspectRatio })) });
        const data = await response.json();
        if (response.ok) {
            const displaySrc = data.image_data_url || data.image_url;
            const usedRatio = data.aspect_ratio || aspectRatio;
            addConversationEntry({ role: 'assistant', imageUrl: data.image_url, imageDataUrl: data.image_data_url || null, filename: data.filename, type: 'image', sourcePrompt: prompt, aspectRatio: usedRatio, meta: 'Generated image · ' + usedRatio });
            displayImage(displaySrc, data.image_data_url || null, data.image_url || null);
            showSuccess('imageSuccess', 'Image regenerated from edited prompt.');
        } else {
            showError('imageError', data.error || 'Failed to regenerate image');
        }
    } catch (e) {
        showError('imageError', 'Error: ' + (e.message || 'Network error'));
    } finally {
        loading.classList.remove('active');
    }
}

async function regenerateFromEditMessage(entryIndex) {
    const chat = getActiveChat();
    if (!chat || !chat.history) return;
    const entry = chat.history[entryIndex];
    if (!entry || entry.role !== 'user' || entry.type !== 'edit_request') return;
    if (entryIndex === 0) return;
    const prevEntry = chat.history[entryIndex - 1];
    if (!prevEntry || (!prevEntry.imageUrl && !prevEntry.imageDataUrl)) {
        showError('imageError', 'Previous image not found for edit.');
        return;
    }
    const prevFilename = prevEntry.filename || (prevEntry.imageUrl ? prevEntry.imageUrl.split('/').pop() : null);
    const prevImageDataUrl = prevEntry.imageDataUrl || prevEntry.imageUrl;
    if (!prevFilename && !prevImageDataUrl) {
        showError('imageError', 'Previous image reference missing.');
        return;
    }
    const changes = entry.text;
    const loading = document.getElementById('imageLoading');
    const errorDiv = document.getElementById('imageError');
    const successDiv = document.getElementById('imageSuccess');
    loading.classList.add('active');
    errorDiv.classList.remove('active');
    successDiv.classList.remove('active');
    try {
        const aspectRatio = getSelectedAspectRatio();
        const response = await fetch('/edit-image', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(withActiveChatId({ filename: prevFilename, image_data_url: prevImageDataUrl, changes, aspect_ratio: aspectRatio })) });
        const data = await response.json();
        if (response.ok) {
            const displaySrc = data.image_data_url || data.image_url;
            const usedRatio = data.aspect_ratio || aspectRatio;
            addConversationEntry({ role: 'assistant', imageUrl: data.image_url, imageDataUrl: data.image_data_url || null, filename: data.filename, type: 'edited_image', sourcePrompt: changes, aspectRatio: usedRatio, meta: 'Edited image · ' + usedRatio });
            displayImage(displaySrc, data.image_data_url || null, data.image_url || null);
            showSuccess('imageSuccess', 'Edit reapplied from updated request.');
        } else {
            showError('imageError', data.error || 'Failed to re-apply edit');
        }
    } catch (e) {
        showError('imageError', 'Error: ' + (e.message || 'Network error'));
    } finally {
        loading.classList.remove('active');
    }
}

async function applyChangesToImage(entryIndex, changes) {
    if (!changes) {
        showError('imageError', 'Please describe the changes you want to make.');
        return;
    }
    const entry = getHistoryEntryByIndex(entryIndex);
    if (!entry || (!entry.imageUrl && !entry.imageDataUrl)) {
        showError('imageError', 'This image cannot be edited (missing reference).');
        return;
    }
    const filename = entry.filename || (entry.imageUrl ? entry.imageUrl.split('/').pop() : null);
    const imageDataUrl = entry.imageDataUrl || entry.imageUrl;
    if (!filename && !imageDataUrl) {
        showError('imageError', 'This image cannot be edited (missing reference).');
        return;
    }
    const btn = document.querySelector(`[data-entry-index="${entryIndex}"] .inline-apply-changes-btn`);
    if (btn) btn.disabled = true;
    const loading = document.getElementById('imageLoading');
    const errorDiv = document.getElementById('imageError');
    const successDiv = document.getElementById('imageSuccess');
    loading.classList.add('active');
    errorDiv.classList.remove('active');
    successDiv.classList.remove('active');
    const aspectRatio = getSelectedAspectRatio();
    addConversationEntry({ role: 'user', text: changes, aspectRatio: aspectRatio, type: 'edit_request' });
    try {
        const response = await fetch('/edit-image', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(withActiveChatId({ filename, image_data_url: imageDataUrl, changes, aspect_ratio: aspectRatio })) });
        const data = await response.json();
        if (response.ok) {
            const displaySrc = data.image_data_url || data.image_url;
            const usedRatio = data.aspect_ratio || aspectRatio;
            addConversationEntry({ role: 'assistant', imageUrl: data.image_url, imageDataUrl: data.image_data_url || null, filename: data.filename, type: 'edited_image', sourcePrompt: changes, aspectRatio: usedRatio, meta: 'Edited image · ' + usedRatio });
            displayImage(displaySrc, data.image_data_url || null, data.image_url || null);
            showSuccess('imageSuccess', 'Changes applied.');
        } else {
            showError('imageError', data.error || 'Failed to update image');
            const chat = getActiveChat();
            if (chat && chat.history.length) chat.history.pop();
            renderConversation();
        }
    } catch (e) {
        showError('imageError', 'Error: ' + (e.message || 'Network error'));
        const chat = getActiveChat();
        if (chat && chat.history.length) chat.history.pop();
        renderConversation();
    } finally {
        if (btn) btn.disabled = false;
        loading.classList.remove('active');
    }
}

function buildAccuracyTracePanel(steps) {
    if (!steps || !steps.length) return null;
    const wrap = document.createElement('details');
    wrap.className = 'accuracy-trace-panel llm-expand-item';
    const sum = document.createElement('summary');
    sum.className = 'llm-expand-summary';
    const label = document.createElement('span');
    label.className = 'llm-expand-label';
    label.textContent = 'Accuracy pipeline';
    const total = document.createElement('span');
    total.className = 'llm-expand-total';
    total.textContent = String(steps.length) + ' steps';
    sum.appendChild(label);
    sum.appendChild(total);
    const body = document.createElement('div');
    body.className = 'llm-expand-body accuracy-trace-body';
    steps.forEach((step, i) => {
        const block = document.createElement('div');
        block.className = 'accuracy-trace-step';
        const h = document.createElement('div');
        h.className = 'accuracy-trace-step-title';
        h.textContent = (i + 1) + '. ' + (step.title || step.id || 'Step');
        block.appendChild(h);
        const prov = [step.provider, step.model].filter(Boolean).join(' · ');
        if (prov) {
            const meta = document.createElement('div');
            meta.className = 'accuracy-trace-step-meta';
            meta.textContent = prov;
            block.appendChild(meta);
        }
        const preIn = document.createElement('pre');
        preIn.className = 'accuracy-trace-io';
        preIn.textContent = 'Input\n' + JSON.stringify(step.input != null ? step.input : {}, null, 2);
        const preOut = document.createElement('pre');
        preOut.className = 'accuracy-trace-io';
        preOut.textContent = 'Output\n' + JSON.stringify(step.output != null ? step.output : {}, null, 2);
        block.appendChild(preIn);
        block.appendChild(preOut);
        body.appendChild(block);
    });
    wrap.appendChild(sum);
    wrap.appendChild(body);
    return wrap;
}

async function getAccurateImage(entryIndex, options) {
    const opts = options && typeof options === 'object' ? options : {};
    const includeTrace = !!opts.includeTrace;

    const entry = getHistoryEntryByIndex(entryIndex);
    if (!entry || (!entry.imageUrl && !entry.imageDataUrl)) {
        showError('imageError', 'This image cannot be processed (missing reference).');
        return;
    }
    const filename = entry.filename || (entry.imageUrl ? entry.imageUrl.split('/').pop() : null);
    const imageDataUrl = entry.imageDataUrl || entry.imageUrl;
    const originalPrompt = getOriginalPromptForEntry(entryIndex);
    if (!filename && !imageDataUrl) {
        showError('imageError', 'This image cannot be processed (missing reference).');
        return;
    }

    const rowSel = `[data-entry-index="${entryIndex}"]`;
    const btn = document.querySelector(`${rowSel} .inline-get-accurate-btn`);
    const traceBtn = document.querySelector(`${rowSel} .inline-get-accurate-trace-btn`);
    const applyBtn = document.querySelector(`${rowSel} .inline-apply-changes-btn`);
    if (btn) btn.disabled = true;
    if (traceBtn) traceBtn.disabled = true;
    if (applyBtn) applyBtn.disabled = true;

    const loading = document.getElementById('imageLoading');
    const errorDiv = document.getElementById('imageError');
    const successDiv = document.getElementById('imageSuccess');
    loading.classList.add('active');
    errorDiv.classList.remove('active');
    successDiv.classList.remove('active');

    const userLine = includeTrace
        ? 'Get Accurate (with step log): checking illustration accuracy and fixing issues…'
        : 'Get Accurate: checking anatomy, view, labels, and pedagogy — fixing issues…';
    addConversationEntry({ role: 'user', text: userLine, type: 'get_accurate_request' });

    const aspectRatio = getSelectedAspectRatio();
    try {
        const response = await fetch('/get-accurate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(withActiveChatId({
                filename,
                image_data_url: imageDataUrl,
                original_prompt: originalPrompt,
                include_trace: includeTrace,
                aspect_ratio: aspectRatio
            }))
        });
        const data = await response.json();
        if (response.ok) {
            const displaySrc = data.image_data_url || data.image_url;
            const flaws = data.flaws_detected || 0;
            const iters = data.iterations || 0;
            const usedRatio = data.aspect_ratio || aspectRatio;
            const ratioSuffix = ' · ' + usedRatio;
            const metaLabel = (flaws > 0
                ? `Accurate image (${flaws} flaw${flaws !== 1 ? 's' : ''} fixed in ${iters} pass${iters !== 1 ? 'es' : ''})`
                : 'Accurate image (no flaws detected)') + ratioSuffix;
            const assistantEntry = {
                role: 'assistant',
                imageUrl: data.image_url,
                imageDataUrl: data.image_data_url || null,
                filename: data.filename,
                type: 'accurate_image',
                aspectRatio: usedRatio,
                meta: metaLabel
            };
            if (includeTrace && Array.isArray(data.accuracy_trace)) {
                assistantEntry.accuracyTrace = data.accuracy_trace;
            }
            addConversationEntry(assistantEntry);
            displayImage(displaySrc, data.image_data_url || null, data.image_url || null);
            showSuccess('imageSuccess', flaws > 0 ? `Accuracy refined: ${flaws} issue(s) corrected.` : 'No issues found — image looks educationally sound.');
        } else {
            showError('imageError', data.error || 'Failed to refine image accuracy');
            const chat = getActiveChat();
            if (chat && chat.history.length) chat.history.pop();
            renderConversation();
        }
    } catch (e) {
        showError('imageError', 'Error: ' + (e.message || 'Network error'));
        const chat = getActiveChat();
        if (chat && chat.history.length) chat.history.pop();
        renderConversation();
    } finally {
        if (btn) btn.disabled = false;
        if (traceBtn) traceBtn.disabled = false;
        if (applyBtn) applyBtn.disabled = false;
        loading.classList.remove('active');
    }
}

async function uploadRagDocument() {
    const fileInput = document.getElementById('ragUploadFileInput');
    const uploadBtn = document.getElementById('uploadRagDocBtn');
    const statusEl = document.getElementById('ragUploadStatus');
    const sessionSelect = document.getElementById('ragSessionDocNameSelect');

    const file = fileInput && fileInput.files && fileInput.files.length > 0
        ? fileInput.files[0]
        : null;

    if (!file) {
        if (statusEl) statusEl.textContent = 'Choose a PDF file first.';
        return;
    }
    if (!String(file.name || '').toLowerCase().endsWith('.pdf')) {
        if (statusEl) statusEl.textContent = 'Only PDF uploads are supported.';
        return;
    }

    const formData = new FormData();
    formData.append('session_id', CLIENT_SESSION_ID);
    formData.append('file', file);

    if (uploadBtn) {
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Uploading...';
    }
    if (statusEl) statusEl.textContent = 'Processing your PDF… this may take a moment.';

    try {
        const response = await fetch('/upload-doc', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }

        if (statusEl) {
            statusEl.textContent = 'Uploaded “' + data.doc_name + '”. Select it above if needed, then ask your question.';
        }

        await loadDocNames();
        if (Array.isArray(data.session_doc_names) && sessionSelect) {
            Array.from(sessionSelect.options || []).forEach(option => {
                option.selected = data.session_doc_names.includes(option.value);
            });
        }
        const baseSelect = document.getElementById('ragDocNameSelect');
        if (baseSelect) {
            Array.from(baseSelect.options || []).forEach(option => {
                if (option.value === NO_RAG_OPTION_VALUE || option.value === WEB_RETRIEVAL_OPTION_VALUE) {
                    option.selected = false;
                } else {
                    option.selected = false;
                }
            });
        }
        normalizeDocSelection();
        if (fileInput) fileInput.value = '';
    } catch (error) {
        if (statusEl) statusEl.textContent = 'Upload failed: ' + (error.message || 'Unknown error');
    } finally {
        if (uploadBtn) {
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload PDF';
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ''));
        reader.onerror = () => reject(new Error('Failed to read uploaded image'));
        reader.readAsDataURL(file);
    });
}

async function uploadAndEditImage() {
    const fileInput = document.getElementById('uploadImageInput');
    const changesEl = document.getElementById('uploadImageChanges');
    const uploadBtn = document.getElementById('uploadEditBtn');
    const progressEl = document.getElementById('uploadEditProgress');

    const file = fileInput && fileInput.files && fileInput.files.length > 0
        ? fileInput.files[0]
        : null;
    const changes = changesEl ? changesEl.value.trim() : '';

    if (!file) {
        showError('imageError', 'Please upload an image first.');
        return;
    }
    if (!changes) {
        showError('imageError', 'Please describe the edits you want to apply.');
        return;
    }

    const loading = document.getElementById('imageLoading');
    const errorDiv = document.getElementById('imageError');
    const successDiv = document.getElementById('imageSuccess');

    if (uploadBtn) {
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'In progress...';
    }
    if (progressEl) {
        progressEl.textContent = 'In progress: creating image from uploaded input...';
        progressEl.classList.add('active');
    }
    if (loading) loading.classList.add('active');
    if (errorDiv) errorDiv.classList.remove('active');
    if (successDiv) successDiv.classList.remove('active');

    const aspectRatio = getSelectedAspectRatio();
    try {
        const imageDataUrl = await readFileAsDataUrl(file);
        addConversationEntry({
            role: 'user',
            text: 'Uploaded image edit request: ' + changes,
            aspectRatio: aspectRatio,
            type: 'edit_request'
        });

        const response = await fetch('/edit-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...withActiveChatId({
                    filename: file.name || 'uploaded-image.png',
                    image_data_url: imageDataUrl,
                    changes,
                    aspect_ratio: aspectRatio
                })
            })
        });

        const data = await response.json();

        if (response.ok) {
            const displaySrc = data.image_data_url || data.image_url;
            const usedRatio = data.aspect_ratio || aspectRatio;
            addConversationEntry({
                role: 'assistant',
                imageUrl: data.image_url,
                imageDataUrl: data.image_data_url || null,
                filename: data.filename,
                type: 'edited_image',
                sourcePrompt: changes,
                aspectRatio: usedRatio,
                meta: 'Created image from upload · ' + usedRatio
            });
            displayImage(displaySrc, data.image_data_url || null, data.image_url || null);
            showSuccess('imageSuccess', 'Image created successfully from uploaded image.');
            if (changesEl) changesEl.value = '';
            if (fileInput) fileInput.value = '';
        } else {
            showError('imageError', data.error || 'Failed to edit uploaded image');
            const chat = getActiveChat();
            if (chat && chat.history.length) chat.history.pop();
            renderConversation();
        }
    } catch (error) {
        showError('imageError', 'Error: ' + (error.message || 'Network error'));
        const chat = getActiveChat();
        if (chat && chat.history.length) chat.history.pop();
        renderConversation();
    } finally {
        if (uploadBtn) {
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Create Image';
        }
        if (progressEl) {
            progressEl.classList.remove('active');
            progressEl.textContent = '';
        }
        if (loading) loading.classList.remove('active');
    }
}

async function generateImage() {
    const prompt = document.getElementById('generatedPrompt').value;
    if (!prompt.trim()) {
        showError('imageError', 'Please enter a prompt');
        return;
    }

    const btn = document.getElementById('generateImageBtn');
    const loading = document.getElementById('imageLoading');
    const errorDiv = document.getElementById('imageError');
    const successDiv = document.getElementById('imageSuccess');

    const aspectRatio = getSelectedAspectRatio();
    addConversationEntry({
        role: 'user',
        text: prompt,
        aspectRatio: aspectRatio,
        type: 'prompt'
    });

    btn.disabled = true;
    loading.classList.add('active');
    errorDiv.classList.remove('active');
    successDiv.classList.remove('active');

    try {
        const response = await fetch('/generate-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                ...withActiveChatId({
                    prompt: prompt,
                    aspect_ratio: aspectRatio
                })
            })
        });

        const data = await response.json();

        if (response.ok) {
            const displaySrc = data.image_data_url || data.image_url;
            const usedRatio = data.aspect_ratio || aspectRatio;
            displayImage(displaySrc, data.image_data_url || null, data.image_url || null);
            addConversationEntry({
                role: 'assistant',
                imageUrl: data.image_url,
                imageDataUrl: data.image_data_url || null,
                filename: data.filename,
                type: 'image',
                sourcePrompt: prompt,
                aspectRatio: usedRatio,
                meta: 'Generated image · ' + usedRatio
            });
            showSuccess('imageSuccess', 'Image generated successfully!');
        } else {
            showError('imageError', data.error || 'Failed to generate image');
        }
    } catch (error) {
        showError('imageError', 'Error connecting to server: ' + error.message);
    } finally {
        btn.disabled = false;
        loading.classList.remove('active');
    }
}

function displayImage(imageSrc, imageDataUrl, canonicalUrl) {
    let chat = getActiveChat();
    if (!chat) {
        chat = createNewChat();
    }
    chat.currentImageUrl = canonicalUrl || imageSrc;
    chat.currentImageDataUrl = imageDataUrl || (typeof imageSrc === 'string' && imageSrc.startsWith('data:') ? imageSrc : null);
    chat.imageCount = (chat.imageCount || 0) + 1;

    const previewImg = document.getElementById('currentImagePreview');
    const previewEmpty = document.getElementById('imagePreviewEmpty');
    const previewMeta = document.getElementById('imagePreviewMeta');
    if (previewImg) {
        previewImg.src = imageSrc;
        previewImg.classList.add('visible');
    }
    if (previewEmpty) {
        previewEmpty.classList.add('hidden');
    }
    if (previewMeta) {
        const now = new Date();
        const timeLabel = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        previewMeta.textContent = `Updated ${timeLabel}`;
    }

    updateDownloadButtonVisibility();
    updateChatImageCountDisplay();
}

function downloadImage() {
    const currentSrc = getCurrentImageDataUrl() || getCurrentImageUrl();
    if (!currentSrc) return;

    const link = document.createElement('a');
    link.href = currentSrc;
    link.download = `generated-image-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function showError(elementId, message) {
    const errorDiv = document.getElementById(elementId);
    errorDiv.textContent = message;
    errorDiv.classList.add('active');
}

function showSuccess(elementId, message) {
    const successDiv = document.getElementById(elementId);
    successDiv.textContent = message;
    successDiv.classList.add('active');
    setTimeout(() => {
        successDiv.classList.remove('active');
    }, 3000);
}

function getCurrentImageUrl() {
    const chat = getActiveChat();
    return chat ? chat.currentImageUrl : null;
}

function getCurrentImageDataUrl() {
    const chat = getActiveChat();
    return chat ? chat.currentImageDataUrl : null;
}

function updateDownloadButtonVisibility() {
    const downloadBtn = document.getElementById('downloadBtn');
    if (!downloadBtn) return;
    const hasImage = !!getCurrentImageUrl();
    downloadBtn.style.display = hasImage ? 'inline-block' : 'none';
}

function openImageFullscreen(imageUrl, captionText) {
    const url = imageUrl || getCurrentImageUrl();
    if (!url) return;

    const overlay = document.getElementById('imageFullscreenOverlay');
    const img = document.getElementById('imageFullscreen');
    const caption = document.getElementById('imageFullscreenCaption');

    if (!overlay || !img) return;

    img.src = url;
    if (caption) {
        caption.textContent = captionText || 'Full-screen view of latest generated image';
    }

    overlay.classList.add('active');
    overlay.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
}

function closeImageFullscreen() {
    const overlay = document.getElementById('imageFullscreenOverlay');
    if (!overlay) return;
    overlay.classList.remove('active');
    overlay.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
}

function initImageInteractions() {
    const previewImg = document.getElementById('currentImagePreview');
    if (previewImg) {
        previewImg.addEventListener('click', () => {
            if (previewImg.classList.contains('visible')) {
                openImageFullscreen();
            }
        });
    }

    const conversationContainer = document.getElementById('conversationContainer');
    if (conversationContainer) {
        conversationContainer.addEventListener('click', (event) => {
            const img = event.target.closest('.message-image');
            if (img && img.src) {
                const caption = img.alt || 'Generated image';
                openImageFullscreen(img.src, caption);
            }
        });
    }

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            closeImageFullscreen();
        }
    });
}

async function initApp() {
    const previousSessionId = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (previousSessionId) {
        await resetServerSession(previousSessionId);
    }
    CLIENT_SESSION_ID = generateClientSessionId();
    sessionStorage.setItem(SESSION_STORAGE_KEY, CLIENT_SESSION_ID);

    initThemeToggle();
    createNewChat('Chat 1');
    loadDocNames();
    const ragDocNameSelect = document.getElementById('ragDocNameSelect');
    if (ragDocNameSelect) {
        ragDocNameSelect.addEventListener('change', normalizeDocSelection);
    }
    const ragSessionDocNameSelect = document.getElementById('ragSessionDocNameSelect');
    if (ragSessionDocNameSelect) {
        ragSessionDocNameSelect.addEventListener('change', normalizeDocSelection);
    }
    updateRetrievalActionsState();
    renderDocChatHistory();
    initImageInteractions();
}

window.handleChatChange = handleChatChange;
window.startNewChat = startNewChat;
window.generateImage = generateImage;
window.uploadAndEditImage = uploadAndEditImage;
window.uploadRagDocument = uploadRagDocument;
window.askDocsQuestion = askDocsQuestion;
window.clearDocChatHistory = clearDocChatHistory;
window.downloadImage = downloadImage;
window.closeImageFullscreen = closeImageFullscreen;

document.addEventListener('DOMContentLoaded', initApp);
window.addEventListener('beforeunload', function() {
    const activeSessionId = sessionStorage.getItem(SESSION_STORAGE_KEY) || CLIENT_SESSION_ID;
    if (!activeSessionId) return;
    resetServerSession(activeSessionId);
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
});

