/* AI Chat → Image Generation page.
 *
 * Standalone module — does not depend on app.js.
 * Maintains multiple chat sessions; each session has its own message history,
 * generated images, and LLM metrics (split GPT vs Gemini).
 */

(function() {
    'use strict';

    // ---------------- Theme toggle ----------------
    function initThemeToggle() {
        const root = document.documentElement;
        const toggle = document.getElementById('themeToggle');
        const label = document.getElementById('themeLabel');
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

    // ---------------- Session model ----------------
    let sessions = [];
    let activeSessionId = null;
    let nextSessionId = 1;
    let nextMessageId = 1;
    let nextImageId = 1;

    function makeMetrics() {
        return {
            calls: 0,
            gptTokens: 0,
            geminiTokens: 0,
        };
    }

    function createSession(name) {
        const id = nextSessionId++;
        const session = {
            id,
            name: name || ('Chat ' + id),
            messages: [],
            images: [],
            metrics: makeMetrics(),
        };
        sessions.push(session);
        activeSessionId = id;
        return session;
    }

    function getActiveSession() {
        return sessions.find(s => s.id === activeSessionId) || null;
    }

    function setActiveSession(id) {
        const target = sessions.find(s => s.id === id);
        if (!target) return;
        activeSessionId = target.id;
        renderAll();
    }

    function deleteSession(id) {
        const idx = sessions.findIndex(s => s.id === id);
        if (idx === -1) return;
        sessions.splice(idx, 1);
        if (sessions.length === 0) {
            createSession();
        } else if (activeSessionId === id) {
            activeSessionId = sessions[Math.max(0, idx - 1)].id;
        }
        renderAll();
    }

    // ---------------- Token accumulation ----------------
    function addGptUsage(usage) {
        const session = getActiveSession();
        if (!session || !usage) return;
        const total = Number(usage.total_tokens || 0)
            || (Number(usage.prompt_tokens || 0) + Number(usage.completion_tokens || 0));
        if (total > 0) session.metrics.gptTokens += total;
        session.metrics.calls += 1;
    }

    function addGeminiUsage(usage) {
        const session = getActiveSession();
        if (!session || !usage) return;
        const total = Number(usage.total_tokens || 0)
            || (Number(usage.prompt_tokens || 0) + Number(usage.completion_tokens || 0));
        if (total > 0) session.metrics.geminiTokens += total;
        session.metrics.calls += 1;
    }

    function addAccurateUsage(usageBundle) {
        // /get-accurate returns { openai: {...}, gemini: {...} } — both prepaid
        // as a single user-initiated call so we increment calls once.
        const session = getActiveSession();
        if (!session || !usageBundle) return;
        const oa = usageBundle.openai || {};
        const ge = usageBundle.gemini || {};
        const oaTotal = Number(oa.total_tokens || 0);
        const geTotal = Number(ge.total_tokens || 0);
        if (oaTotal > 0) session.metrics.gptTokens += oaTotal;
        if (geTotal > 0) session.metrics.geminiTokens += geTotal;
        session.metrics.calls += 1;
    }

    // ---------------- Rendering: sidebar ----------------
    function renderSidebar() {
        const list = document.getElementById('aiSessionList');
        const countLabel = document.getElementById('aiSessionCountLabel');
        if (!list) return;

        list.innerHTML = '';
        sessions.forEach(session => {
            const li = document.createElement('li');
            li.className = 'ai-chat-session-item' + (session.id === activeSessionId ? ' active' : '');

            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'ai-chat-session-btn';
            btn.title = session.name;
            const lastUser = [...session.messages].reverse().find(m => m.role === 'user');
            const subtitle = lastUser ? lastUser.content : 'No messages yet';
            btn.innerHTML =
                '<span class="ai-chat-session-name"></span>' +
                '<span class="ai-chat-session-subtitle"></span>';
            btn.querySelector('.ai-chat-session-name').textContent = session.name;
            btn.querySelector('.ai-chat-session-subtitle').textContent =
                subtitle.length > 60 ? subtitle.slice(0, 60) + '…' : subtitle;
            btn.onclick = () => setActiveSession(session.id);
            li.appendChild(btn);

            const renameBtn = document.createElement('button');
            renameBtn.type = 'button';
            renameBtn.className = 'ai-chat-session-icon-btn';
            renameBtn.title = 'Rename chat';
            renameBtn.textContent = '✎';
            renameBtn.onclick = (e) => {
                e.stopPropagation();
                const newName = prompt('Rename chat', session.name);
                if (newName && newName.trim()) {
                    session.name = newName.trim();
                    renderAll();
                }
            };
            li.appendChild(renameBtn);

            const closeBtn = document.createElement('button');
            closeBtn.type = 'button';
            closeBtn.className = 'ai-chat-session-icon-btn';
            closeBtn.title = 'Delete chat';
            closeBtn.textContent = '×';
            closeBtn.onclick = (e) => {
                e.stopPropagation();
                if (sessions.length === 1 || confirm('Delete this chat session and its history?')) {
                    deleteSession(session.id);
                }
            };
            li.appendChild(closeBtn);

            list.appendChild(li);
        });

        if (countLabel) {
            countLabel.textContent = sessions.length === 1 ? '1 session' : (sessions.length + ' sessions');
        }
    }

    // ---------------- Rendering: messages ----------------
    function renderMessages() {
        const container = document.getElementById('aiChatMessages');
        if (!container) return;
        const session = getActiveSession();
        const messages = session ? session.messages : [];

        if (!messages.length) {
            container.innerHTML = '<p class="ai-chat-empty">Your conversation will appear here. Type a question below to start.</p>';
            return;
        }

        container.innerHTML = '';
        messages.forEach((msg) => {
            const row = document.createElement('div');
            row.className = 'ai-chat-message ' + (msg.role === 'user' ? 'ai-chat-user' : 'ai-chat-assistant');
            row.dataset.messageId = msg.id;

            const roleLabel = document.createElement('div');
            roleLabel.className = 'ai-chat-role';
            roleLabel.textContent = msg.role === 'user' ? 'You' : 'Assistant';
            row.appendChild(roleLabel);

            const content = document.createElement('div');
            content.className = 'ai-chat-content';
            content.textContent = msg.content;
            row.appendChild(content);

            if (msg.role === 'assistant' && msg.content && msg.content.trim()) {
                const actions = document.createElement('div');
                actions.className = 'ai-chat-actions';

                const generateBtn = document.createElement('button');
                generateBtn.type = 'button';
                generateBtn.className = 'rag-btn rag-btn-primary ai-chat-generate-btn';
                generateBtn.textContent = '🎨 Generate image from this';
                generateBtn.onclick = () => generateImageFromMessage(msg.id);
                actions.appendChild(generateBtn);

                const copyBtn = document.createElement('button');
                copyBtn.type = 'button';
                copyBtn.className = 'rag-btn rag-btn-secondary ai-chat-copy-btn';
                copyBtn.textContent = 'Copy';
                copyBtn.onclick = () => {
                    if (navigator.clipboard && navigator.clipboard.writeText) {
                        navigator.clipboard.writeText(msg.content).catch(() => {});
                    }
                };
                actions.appendChild(copyBtn);

                row.appendChild(actions);
            }

            container.appendChild(row);
        });

        container.scrollTop = container.scrollHeight;
    }

    // ---------------- Rendering: image preview & list ----------------
    function getLatestImage() {
        const session = getActiveSession();
        return session && session.images.length
            ? session.images[session.images.length - 1]
            : null;
    }

    function renderImages() {
        const session = getActiveSession();
        const previewImg = document.getElementById('aiImagePreview');
        const previewEmpty = document.getElementById('aiImagePreviewEmpty');
        const previewMeta = document.getElementById('aiImagePreviewMeta');
        const grid = document.getElementById('aiChatImagesGrid');
        const emptyMsg = document.getElementById('aiChatImagesEmpty');
        const controls = document.getElementById('aiImageControls');

        const latest = getLatestImage();

        if (previewImg && previewEmpty) {
            if (latest) {
                previewImg.src = latest.imageDataUrl || latest.imageUrl;
                previewImg.classList.add('visible');
                previewEmpty.classList.add('hidden');
            } else {
                previewImg.removeAttribute('src');
                previewImg.classList.remove('visible');
                previewEmpty.classList.remove('hidden');
            }
        }

        if (previewMeta) {
            previewMeta.textContent = latest
                ? ('Updated ' + new Date(latest.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
                : 'No image generated yet';
        }

        if (controls) {
            controls.hidden = !latest;
        }
        const downloadBtn = document.getElementById('aiDownloadBtn');
        if (downloadBtn) downloadBtn.hidden = !latest;

        if (grid) {
            grid.innerHTML = '';
            if (session && session.images.length) {
                session.images.forEach((img, idx) => {
                    const item = document.createElement('div');
                    item.className = 'ai-chat-image-item';
                    const el = document.createElement('img');
                    el.src = img.imageDataUrl || img.imageUrl;
                    el.alt = 'Generated image ' + (idx + 1);
                    el.onclick = () => openImageFullscreen(el.src, img.prompt);
                    item.appendChild(el);
                    const cap = document.createElement('div');
                    cap.className = 'ai-chat-image-caption';
                    const promptText = img.prompt || '';
                    cap.textContent = promptText.slice(0, 80) + (promptText.length > 80 ? '…' : '');
                    item.appendChild(cap);
                    grid.appendChild(item);
                });
            }
        }

        if (emptyMsg) {
            emptyMsg.style.display = (session && session.images.length) ? 'none' : 'block';
        }
    }

    // ---------------- Rendering: metrics ----------------
    function formatNumber(n) {
        const num = Number(n) || 0;
        return num.toLocaleString();
    }

    function renderMetrics() {
        const session = getActiveSession();
        const m = session ? session.metrics : null;
        const set = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        };

        if (!m) {
            set('aiMetricCalls', '0');
            set('aiMetricTotalTokens', '0');
            set('aiMetricGptTokens', '0');
            set('aiMetricGeminiTokens', '0');
            return;
        }

        set('aiMetricCalls', String(m.calls));
        set('aiMetricGptTokens', formatNumber(m.gptTokens));
        set('aiMetricGeminiTokens', formatNumber(m.geminiTokens));
        set('aiMetricTotalTokens', formatNumber(m.gptTokens + m.geminiTokens));
    }

    function renderHeader() {
        const session = getActiveSession();
        const titleEl = document.getElementById('aiActiveChatTitle');
        const metaEl = document.getElementById('aiActiveChatMeta');
        if (titleEl) titleEl.textContent = session ? session.name : 'Chat';
        if (metaEl) {
            if (!session || !session.messages.length) {
                metaEl.textContent = 'Type a question below to start.';
            } else {
                metaEl.textContent = session.messages.length + ' message' + (session.messages.length !== 1 ? 's' : '')
                    + ' · ' + session.images.length + ' image' + (session.images.length !== 1 ? 's' : '');
            }
        }
    }

    function renderAll() {
        renderSidebar();
        renderHeader();
        renderMessages();
        renderImages();
        renderMetrics();
    }

    // ---------------- Send message ----------------
    async function sendMessage() {
        const inputEl = document.getElementById('aiChatInput');
        const sendBtn = document.getElementById('aiChatSendBtn');
        const loadingEl = document.getElementById('aiChatLoading');
        const errorEl = document.getElementById('aiChatError');

        const text = inputEl ? inputEl.value.trim() : '';
        if (!text) return;

        let session = getActiveSession();
        if (!session) session = createSession();

        // History to send to backend = everything in this session BEFORE the new
        // user message (the server appends the user message itself).
        const historyForApi = session.messages.map(m => ({
            role: m.role,
            content: m.content,
        }));

        session.messages.push({ id: nextMessageId++, role: 'user', content: text });
        if (inputEl) inputEl.value = '';
        renderMessages();
        renderHeader();
        renderSidebar();

        if (loadingEl) loadingEl.classList.remove('hidden');
        if (errorEl) {
            errorEl.textContent = '';
            errorEl.classList.add('hidden');
        }
        if (sendBtn) sendBtn.disabled = true;

        try {
            const response = await fetch('/ai-chat-message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_message: text,
                    history: historyForApi,
                }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Failed to get answer');
            }

            const metrics = data.metrics || {};
            session.messages.push({
                id: nextMessageId++,
                role: 'assistant',
                content: String(data.answer || 'No answer generated.'),
                metrics,
            });
            addGptUsage(metrics);
            renderAll();
        } catch (error) {
            session.messages.push({
                id: nextMessageId++,
                role: 'assistant',
                content: 'Error: ' + (error.message || 'Unknown error'),
            });
            renderMessages();
            if (errorEl) {
                errorEl.textContent = 'Error: ' + (error.message || 'Unknown error');
                errorEl.classList.remove('hidden');
            }
        } finally {
            if (loadingEl) loadingEl.classList.add('hidden');
            if (sendBtn) sendBtn.disabled = false;
        }
    }

    // ---------------- Generate image from message ----------------
    function showImageActionLoading(text) {
        const el = document.getElementById('aiImageActionLoading');
        const txt = document.getElementById('aiImageActionLoadingText');
        if (txt && text) txt.textContent = text;
        if (el) el.classList.remove('hidden');
    }
    function hideImageActionLoading() {
        const el = document.getElementById('aiImageActionLoading');
        if (el) el.classList.add('hidden');
    }
    function showImageError(msg) {
        const el = document.getElementById('aiImageError');
        if (!el) return;
        el.textContent = msg;
        el.classList.remove('hidden');
    }
    function clearImageError() {
        const el = document.getElementById('aiImageError');
        if (!el) return;
        el.textContent = '';
        el.classList.add('hidden');
    }

    async function generateImageFromMessage(messageId) {
        const session = getActiveSession();
        if (!session) return;
        const message = session.messages.find(m => m.id === messageId);
        if (!message || !message.content) return;

        const prompt = message.content;
        const generateBtns = document.querySelectorAll(
            '[data-message-id="' + messageId + '"] .ai-chat-generate-btn'
        );
        generateBtns.forEach(btn => { btn.disabled = true; btn.textContent = 'Generating…'; });

        showImageActionLoading('Generating image…');
        clearImageError();

        try {
            const response = await fetch('/generate-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    session_id: 'ai_chat_' + session.id,
                }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Failed to generate image');
            }

            session.images.push({
                id: nextImageId++,
                prompt,
                imageUrl: data.image_url || null,
                imageDataUrl: data.image_data_url || null,
                filename: data.filename || null,
                createdAt: Date.now(),
                fromMessageId: messageId,
                kind: 'generated',
            });
            const usage = (data.usage || {}).gemini || {};
            addGeminiUsage(usage);

            renderImages();
            renderMetrics();
            renderHeader();
        } catch (error) {
            showImageError('Error: ' + (error.message || 'Unknown error'));
        } finally {
            hideImageActionLoading();
            generateBtns.forEach(btn => {
                btn.disabled = false;
                btn.textContent = '🎨 Generate image from this';
            });
        }
    }

    // ---------------- Apply changes ----------------
    async function applyChangesToLatest() {
        const session = getActiveSession();
        const latest = getLatestImage();
        if (!session || !latest) return;
        const ta = document.getElementById('aiImageChanges');
        const changes = (ta ? ta.value : '').trim();
        if (!changes) {
            showImageError('Please describe the changes you want to apply.');
            return;
        }

        const applyBtn = document.getElementById('aiApplyChangesBtn');
        if (applyBtn) applyBtn.disabled = true;
        showImageActionLoading('Applying changes…');
        clearImageError();

        try {
            const filename = latest.filename
                || (latest.imageUrl ? latest.imageUrl.split('/').pop() : null);
            const response = await fetch('/edit-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename,
                    image_data_url: latest.imageDataUrl || latest.imageUrl,
                    changes,
                    session_id: 'ai_chat_' + session.id,
                }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to update image');

            session.images.push({
                id: nextImageId++,
                prompt: changes,
                imageUrl: data.image_url || null,
                imageDataUrl: data.image_data_url || null,
                filename: data.filename || null,
                createdAt: Date.now(),
                kind: 'edited',
                parentImageId: latest.id,
            });
            const usage = (data.usage || {}).gemini || {};
            addGeminiUsage(usage);

            if (ta) ta.value = '';
            renderImages();
            renderMetrics();
            renderHeader();
        } catch (error) {
            showImageError('Error: ' + (error.message || 'Unknown error'));
        } finally {
            hideImageActionLoading();
            if (applyBtn) applyBtn.disabled = false;
        }
    }

    // ---------------- Get Accurate ----------------
    async function getAccurateLatest(includeTrace) {
        const session = getActiveSession();
        const latest = getLatestImage();
        if (!session || !latest) return;

        const accurateBtn = document.getElementById('aiGetAccurateBtn');
        const accurateTraceBtn = document.getElementById('aiGetAccurateTraceBtn');
        if (accurateBtn) accurateBtn.disabled = true;
        if (accurateTraceBtn) accurateTraceBtn.disabled = true;
        showImageActionLoading(includeTrace ? 'Get Accurate (with log)…' : 'Get Accurate — refining…');
        clearImageError();

        try {
            const filename = latest.filename
                || (latest.imageUrl ? latest.imageUrl.split('/').pop() : null);
            const response = await fetch('/get-accurate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename,
                    image_data_url: latest.imageDataUrl || latest.imageUrl,
                    original_prompt: latest.prompt || '',
                    include_trace: !!includeTrace,
                    session_id: 'ai_chat_' + session.id,
                }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to refine image');

            const flaws = data.flaws_detected || 0;
            const iters = data.iterations || 0;
            const meta = flaws > 0
                ? `Accurate image (${flaws} flaw${flaws !== 1 ? 's' : ''} fixed in ${iters} pass${iters !== 1 ? 'es' : ''})`
                : 'Accurate image (no flaws detected)';

            session.images.push({
                id: nextImageId++,
                prompt: latest.prompt || '',
                imageUrl: data.image_url || null,
                imageDataUrl: data.image_data_url || null,
                filename: data.filename || null,
                createdAt: Date.now(),
                kind: 'accurate',
                meta,
                accuracyTrace: includeTrace ? (data.accuracy_trace || null) : null,
                parentImageId: latest.id,
            });
            addAccurateUsage(data.usage || {});

            renderImages();
            renderMetrics();
            renderHeader();
        } catch (error) {
            showImageError('Error: ' + (error.message || 'Unknown error'));
        } finally {
            hideImageActionLoading();
            if (accurateBtn) accurateBtn.disabled = false;
            if (accurateTraceBtn) accurateTraceBtn.disabled = false;
        }
    }

    // ---------------- Download ----------------
    function downloadLatest() {
        const latest = getLatestImage();
        if (!latest) return;
        const src = latest.imageDataUrl || latest.imageUrl;
        if (!src) return;
        const link = document.createElement('a');
        link.href = src;
        link.download = (latest.filename || ('generated-image-' + Date.now())) + (
            latest.filename && /\.(png|jpe?g|webp)$/i.test(latest.filename) ? '' : '.png'
        );
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // ---------------- Fullscreen overlay ----------------
    function openImageFullscreen(src, captionText) {
        const overlay = document.getElementById('imageFullscreenOverlay');
        const img = document.getElementById('imageFullscreen');
        const caption = document.getElementById('imageFullscreenCaption');
        if (!overlay || !img || !src) return;
        img.src = src;
        if (caption) {
            const text = captionText
                ? ('Prompt: ' + captionText.slice(0, 140) + (captionText.length > 140 ? '…' : ''))
                : 'Generated image';
            caption.textContent = text;
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

    // ---------------- Clear messages in current chat ----------------
    function clearActiveChat() {
        const session = getActiveSession();
        if (!session) return;
        if (!session.messages.length && !session.images.length) return;
        if (!confirm('Clear all messages and images in this chat?')) return;
        session.messages = [];
        session.images = [];
        session.metrics = makeMetrics();
        renderAll();
    }

    // ---------------- Sidebar collapse ----------------
    function initSidebarToggle() {
        const layout = document.getElementById('aiChatLayout');
        const sidebar = document.getElementById('aiChatSidebar');
        const toggle = document.getElementById('aiSidebarToggle');
        if (!layout || !sidebar || !toggle) return;

        function apply(collapsed) {
            layout.classList.toggle('sidebar-collapsed', collapsed);
            sidebar.classList.toggle('collapsed', collapsed);
            toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            toggle.setAttribute(
                'aria-label',
                collapsed ? 'Expand chat sessions sidebar' : 'Collapse chat sessions sidebar'
            );
            toggle.setAttribute('title', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
        }

        const saved = localStorage.getItem('aiChatSidebarCollapsed') === '1';
        apply(saved);

        toggle.addEventListener('click', () => {
            const next = !sidebar.classList.contains('collapsed');
            apply(next);
            localStorage.setItem('aiChatSidebarCollapsed', next ? '1' : '0');
        });
    }

    // ---------------- Init ----------------
    function init() {
        initThemeToggle();
        initSidebarToggle();
        createSession('Chat 1');
        renderAll();

        const newBtn = document.getElementById('aiNewChatBtn');
        if (newBtn) {
            newBtn.addEventListener('click', () => {
                createSession();
                renderAll();
                const inputEl = document.getElementById('aiChatInput');
                if (inputEl) inputEl.focus();
            });
        }

        const sendBtn = document.getElementById('aiChatSendBtn');
        if (sendBtn) sendBtn.addEventListener('click', sendMessage);

        const inputEl = document.getElementById('aiChatInput');
        if (inputEl) {
            inputEl.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }

        const clearBtn = document.getElementById('aiClearChatBtn');
        if (clearBtn) clearBtn.addEventListener('click', clearActiveChat);

        const applyBtn = document.getElementById('aiApplyChangesBtn');
        if (applyBtn) applyBtn.addEventListener('click', applyChangesToLatest);
        const accurateBtn = document.getElementById('aiGetAccurateBtn');
        if (accurateBtn) accurateBtn.addEventListener('click', () => getAccurateLatest(false));
        const accurateTraceBtn = document.getElementById('aiGetAccurateTraceBtn');
        if (accurateTraceBtn) accurateTraceBtn.addEventListener('click', () => getAccurateLatest(true));
        const downloadBtn = document.getElementById('aiDownloadBtn');
        if (downloadBtn) downloadBtn.addEventListener('click', downloadLatest);

        const overlay = document.getElementById('imageFullscreenOverlay');
        const closeBtn = document.getElementById('aiImageFullscreenClose');
        if (closeBtn) closeBtn.addEventListener('click', closeImageFullscreen);
        if (overlay) {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) closeImageFullscreen();
            });
        }
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeImageFullscreen();
        });

        const previewImg = document.getElementById('aiImagePreview');
        if (previewImg) {
            previewImg.addEventListener('click', () => {
                if (previewImg.classList.contains('visible')) {
                    const last = getLatestImage();
                    openImageFullscreen(previewImg.src, last ? last.prompt : '');
                }
            });
        }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
