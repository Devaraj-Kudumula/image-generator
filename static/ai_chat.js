/* AI Chat → Image Generation page.
 *
 * Standalone module — does not depend on app.js.
 * Maintains multiple chat sessions; each session has its own message history
 * and generated images.
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

    const AI_ALLOWED_ASPECT_RATIOS = ['1:1', '4:3', '3:4', '16:9', '9:16', '3:2', '2:3', '21:9'];
    const AI_DEFAULT_ASPECT_RATIO = '16:9';
    function getSelectedAspectRatio() {
        const el = document.getElementById('aspectRatio');
        const v = el && el.value ? String(el.value).trim() : '';
        return AI_ALLOWED_ASPECT_RATIOS.indexOf(v) >= 0 ? v : AI_DEFAULT_ASPECT_RATIO;
    }

    function createSession(name) {
        const id = nextSessionId++;
        const session = {
            id,
            name: name || ('Chat ' + id),
            messages: [],
            images: [],
            systemPromptOverride: null,
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
            if (msg.role === 'theme') {
                const row = document.createElement('div');
                row.className = 'ai-chat-message ai-chat-theme';
                row.dataset.messageId = msg.id;

                const roleLabel = document.createElement('div');
                roleLabel.className = 'ai-chat-role';
                roleLabel.textContent = 'Theme · ' + (msg.themeLabel || msg.themeId || 'Custom');
                row.appendChild(roleLabel);

                const content = document.createElement('div');
                content.className = 'ai-chat-content';
                content.textContent = msg.content;
                row.appendChild(content);

                const meta = document.createElement('div');
                meta.className = 'ai-chat-msg-meta';
                meta.textContent = 'This text is sent to the model as the system prompt for this chat until you pick another theme or clear messages.';
                row.appendChild(meta);

                container.appendChild(row);
                return;
            }

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
            if (latest) {
                const ts = new Date(latest.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                previewMeta.textContent = 'Updated ' + ts
                    + (latest.aspectRatio ? ' · ' + latest.aspectRatio : '');
            } else {
                previewMeta.textContent = 'No image generated yet';
            }
        }

        const latestPromptWrap = document.getElementById('aiLatestViewPromptWrap');
        const latestPromptBtn = document.getElementById('aiLatestViewPromptBtn');
        if (latestPromptWrap && latestPromptBtn) {
            const p = latest && latest.prompt ? String(latest.prompt).trim() : '';
            if (p) {
                latestPromptWrap.hidden = false;
                latestPromptBtn.textContent = latest.kind === 'refined_prompt'
                    ? 'View refined prompt'
                    : 'View generation prompt';
                latestPromptBtn.onclick = function() {
                    openPromptModal(
                        latest.kind === 'refined_prompt'
                            ? 'Refined generation prompt'
                            : 'Generation prompt',
                        p
                    );
                };
            } else {
                latestPromptWrap.hidden = true;
            }
        }

        if (controls) {
            controls.hidden = !latest;
        }
        const canvasActions = document.getElementById('aiCanvasActions');
        if (canvasActions) {
            canvasActions.hidden = !latest;
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
                    el.onclick = function() {
                        openImageFullscreen(el.src, img.prompt, img.kind);
                    };
                    item.appendChild(el);
                    const cap = document.createElement('div');
                    cap.className = 'ai-chat-image-caption';
                    const promptText = img.prompt || '';
                    cap.textContent = promptText.slice(0, 80) + (promptText.length > 80 ? '…' : '');
                    item.appendChild(cap);
                    if (img.aspectRatio) {
                        const tag = document.createElement('span');
                        tag.className = 'aspect-ratio-tag';
                        tag.textContent = 'Aspect: ' + img.aspectRatio;
                        item.appendChild(tag);
                    }
                    const pTrim = promptText.trim();
                    if (pTrim) {
                        const vp = document.createElement('button');
                        vp.type = 'button';
                        vp.className = 'ai-view-prompt-btn';
                        vp.textContent = img.kind === 'refined_prompt'
                            ? 'View refined prompt'
                            : 'View prompt';
                        vp.onclick = function(e) {
                            e.stopPropagation();
                            openPromptModal(
                                img.kind === 'refined_prompt'
                                    ? 'Refined generation prompt'
                                    : 'Generation prompt',
                                pTrim
                            );
                        };
                        item.appendChild(vp);
                    }
                    grid.appendChild(item);
                });
            }
        }

        if (emptyMsg) {
            emptyMsg.style.display = (session && session.images.length) ? 'none' : 'block';
        }
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
    }

    function buildHistoryForApi(session) {
        if (!session) return [];
        return session.messages
            .filter(function(m) { return m.role === 'user' || m.role === 'assistant'; })
            .map(function(m) { return { role: m.role, content: m.content }; });
    }

    async function completeChatTurn(session, userText, historyForApi) {
        const loadingEl = document.getElementById('aiChatLoading');
        const errorEl = document.getElementById('aiChatError');
        const sendBtn = document.getElementById('aiChatSendBtn');

        const body = {
            user_message: userText,
            history: historyForApi || [],
        };
        const ov = session.systemPromptOverride;
        if (ov && String(ov).trim()) {
            body.system_prompt_override = String(ov).trim();
        }

        if (loadingEl) loadingEl.classList.remove('hidden');
        if (errorEl) {
            errorEl.textContent = '';
            errorEl.classList.add('hidden');
        }
        if (sendBtn) sendBtn.disabled = true;

        // Assistant placeholder we stream tokens into.
        const assistantMsg = { id: nextMessageId++, role: 'assistant', content: '' };
        session.messages.push(assistantMsg);
        renderMessages();

        // Update just this message's content node so we don't re-render every token.
        const liveContentEl = () => {
            const container = document.getElementById('aiChatMessages');
            if (!container) return null;
            const row = container.querySelector('[data-message-id="' + assistantMsg.id + '"]');
            return row ? row.querySelector('.ai-chat-content') : null;
        };
        const paintDelta = () => {
            const el = liveContentEl();
            if (el) {
                el.textContent = assistantMsg.content;
                el.parentElement.scrollIntoView({ block: 'end' });
            }
        };

        try {
            const response = await fetch('/ai-chat-message/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });

            if (!response.ok || !response.body) {
                // Validation errors (400/503) come back as plain JSON.
                let msg = 'Failed to get answer';
                try { msg = (await response.json()).error || msg; } catch (e) {}
                throw new Error(msg);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let streamError = null;
            let gotAnyText = false;

            // Hide the global spinner once streaming starts; tokens are the indicator.
            if (loadingEl) loadingEl.classList.add('hidden');

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                // SSE events are separated by a blank line.
                let sep;
                while ((sep = buffer.indexOf('\n\n')) !== -1) {
                    const rawEvent = buffer.slice(0, sep);
                    buffer = buffer.slice(sep + 2);
                    const line = rawEvent.split('\n').find((l) => l.startsWith('data:'));
                    if (!line) continue;
                    let payload;
                    try {
                        payload = JSON.parse(line.slice(5).trim());
                    } catch (e) {
                        continue;
                    }
                    if (payload.delta) {
                        assistantMsg.content += payload.delta;
                        gotAnyText = true;
                        paintDelta();
                    } else if (payload.error) {
                        streamError = payload.error;
                    }
                    // payload.done carries metrics; nothing to render for now.
                }
            }

            if (streamError) {
                throw new Error(streamError);
            }
            if (!gotAnyText && !assistantMsg.content) {
                assistantMsg.content = 'No answer generated.';
            }
            // Final render so the assistant action buttons attach.
            renderAll();
        } catch (error) {
            assistantMsg.content = 'Error: ' + (error.message || 'Unknown error');
            renderMessages();
            renderHeader();
            renderSidebar();
            if (errorEl) {
                errorEl.textContent = 'Error: ' + (error.message || 'Unknown error');
                errorEl.classList.remove('hidden');
            }
        } finally {
            if (loadingEl) loadingEl.classList.add('hidden');
            if (sendBtn) sendBtn.disabled = false;
        }
    }

    // ---------------- Send message ----------------
    async function sendMessage() {
        const inputEl = document.getElementById('aiChatInput');
        const text = inputEl ? inputEl.value.trim() : '';
        if (!text) return;

        let session = getActiveSession();
        if (!session) session = createSession();

        const historyForApi = buildHistoryForApi(session);

        session.messages.push({ id: nextMessageId++, role: 'user', content: text });
        if (inputEl) inputEl.value = '';
        renderMessages();
        renderHeader();
        renderSidebar();

        await completeChatTurn(session, text, historyForApi);
    }

    function themeKickoffUserMessage(label) {
        return (
            'The user selected the "' + label + '" theme; the full system instructions were shown in the chat. ' +
            'Reply in one or two short sentences that you will follow those instructions for this conversation, ' +
            'then invite their next message.'
        );
    }

    async function applyChatTheme(themeId, label, promptText) {
        const prompt = (promptText || '').trim();
        if (!prompt) {
            const err = document.getElementById('aiChatError');
            if (err) {
                err.textContent = 'This theme has no prompt text yet. Edit AI_CHAT_THEME_PROMPTS in prompts.py.';
                err.classList.remove('hidden');
            }
            return;
        }

        let session = getActiveSession();
        if (!session) session = createSession();

        const historyForApi = buildHistoryForApi(session);

        session.systemPromptOverride = prompt;
        session.messages.push({
            id: nextMessageId++,
            role: 'theme',
            themeId: themeId,
            themeLabel: label,
            content: prompt,
        });
        renderMessages();
        renderHeader();
        renderSidebar();

        await completeChatTurn(session, themeKickoffUserMessage(label), historyForApi);
    }

    // ---------------- Theme menu (conversation style) ----------------

    function setThemeMenuOpen(open) {
        const btn = document.getElementById('aiChatThemeBtn');
        const menu = document.getElementById('aiChatThemeMenu');
        if (!menu || !btn) return;
        if (open) {
            menu.classList.remove('hidden');
            btn.setAttribute('aria-expanded', 'true');
            menu.setAttribute('aria-hidden', 'false');
        } else {
            menu.classList.add('hidden');
            btn.setAttribute('aria-expanded', 'false');
            menu.setAttribute('aria-hidden', 'true');
        }
    }

    function isThemeMenuOpen() {
        const menu = document.getElementById('aiChatThemeMenu');
        return menu && !menu.classList.contains('hidden');
    }

    function initChatThemeMenu() {
        const btn = document.getElementById('aiChatThemeBtn');
        const menu = document.getElementById('aiChatThemeMenu');
        if (!btn || !menu) return;

        fetch('/ai-chat-themes')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                const themes = (data && data.themes) || {};
                const preferredOrder = [
                    'realistic',
                    'general',
                    'histology',
                    'organ_images',
                    'radiology',
                ];
                const keys = [];
                preferredOrder.forEach(function(k) {
                    if (themes[k]) keys.push(k);
                });
                Object.keys(themes).sort().forEach(function(k) {
                    if (keys.indexOf(k) === -1) keys.push(k);
                });

                menu.innerHTML = '';
                keys.forEach(function(id) {
                    const meta = themes[id];
                    const label = (meta && meta.label) || id;
                    const prompt = (meta && meta.prompt) || '';
                    const opt = document.createElement('button');
                    opt.type = 'button';
                    opt.className = 'ai-chat-theme-option';
                    opt.setAttribute('role', 'menuitem');
                    opt.dataset.themeId = id;
                    const title = document.createElement('span');
                    title.textContent = label;
                    opt.appendChild(title);
                    const desc = document.createElement('span');
                    desc.className = 'ai-chat-theme-option-desc';
                    desc.textContent = prompt.trim()
                        ? 'Apply as system prompt'
                        : 'No prompt configured';
                    opt.appendChild(desc);
                    opt.addEventListener('click', function() {
                        setThemeMenuOpen(false);
                        applyChatTheme(id, label, prompt);
                    });
                    menu.appendChild(opt);
                });

                btn.disabled = keys.length === 0;
            })
            .catch(function() {
                btn.disabled = true;
                const err = document.getElementById('aiChatError');
                if (err) {
                    err.textContent = 'Could not load themes. Check your connection and try again.';
                    err.classList.remove('hidden');
                }
            });

        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            if (btn.disabled) return;
            setThemeMenuOpen(!isThemeMenuOpen());
        });

        document.addEventListener('click', function() {
            if (isThemeMenuOpen()) setThemeMenuOpen(false);
        });
        menu.addEventListener('click', function(e) {
            e.stopPropagation();
        });
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

        const aspectRatio = getSelectedAspectRatio();
        try {
            const response = await fetch('/generate-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    aspect_ratio: aspectRatio,
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
                aspectRatio: data.aspect_ratio || aspectRatio,
                createdAt: Date.now(),
                fromMessageId: messageId,
                kind: 'generated',
            });
            renderImages();
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

    function openCanvasEditorForLatest() {
        const latest = getLatestImage();
        if (!latest) {
            showImageError('No image to edit in canvas.');
            return;
        }
        if (typeof window.openCanvasEditor !== 'function') {
            showImageError('Canvas editor failed to load. Refresh the page and try again.');
            return;
        }
        const filename = latest.filename
            || (latest.imageUrl ? latest.imageUrl.split('/').pop() : null);
        const imageDataUrl = latest.imageDataUrl || latest.imageUrl;
        window.openCanvasEditor({
            filename: filename,
            imageDataUrl: imageDataUrl,
            onSave: function (pngDataUrl) {
                saveCanvasEditToSession(pngDataUrl);
            },
        });
    }

    function saveCanvasEditToSession(pngDataUrl) {
        const session = getActiveSession();
        if (!session || !pngDataUrl) return;
        const aspectRatio = getSelectedAspectRatio();
        session.images.push({
            id: nextImageId++,
            prompt: 'Canvas edit',
            imageUrl: null,
            imageDataUrl: pngDataUrl,
            filename: null,
            aspectRatio: aspectRatio,
            createdAt: Date.now(),
            kind: 'canvas_edited',
        });
        renderImages();
        renderHeader();
        clearImageError();
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
            const aspectRatio = getSelectedAspectRatio();
            const response = await fetch('/edit-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename,
                    image_data_url: latest.imageDataUrl || latest.imageUrl,
                    changes,
                    aspect_ratio: aspectRatio,
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
                aspectRatio: data.aspect_ratio || aspectRatio,
                createdAt: Date.now(),
                kind: 'edited',
                parentImageId: latest.id,
            });

            if (ta) ta.value = '';
            renderImages();
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
        const refinedBtn = document.getElementById('aiRefinedPromptImageBtn');
        if (accurateBtn) accurateBtn.disabled = true;
        if (accurateTraceBtn) accurateTraceBtn.disabled = true;
        if (refinedBtn) refinedBtn.disabled = true;
        showImageActionLoading(includeTrace ? 'Get Accurate (with log)…' : 'Get Accurate — refining…');
        clearImageError();

        try {
            const filename = latest.filename
                || (latest.imageUrl ? latest.imageUrl.split('/').pop() : null);
            const aspectRatio = getSelectedAspectRatio();
            const response = await fetch('/get-accurate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename,
                    image_data_url: latest.imageDataUrl || latest.imageUrl,
                    original_prompt: latest.prompt || '',
                    include_trace: !!includeTrace,
                    aspect_ratio: aspectRatio,
                    session_id: 'ai_chat_' + session.id,
                }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to refine image');

            const flaws = data.flaws_detected || 0;
            const iters = data.iterations || 0;
            const usedRatio = data.aspect_ratio || aspectRatio;
            const meta = (flaws > 0
                ? `Accurate image (${flaws} flaw${flaws !== 1 ? 's' : ''} fixed in ${iters} pass${iters !== 1 ? 'es' : ''})`
                : 'Accurate image (no flaws detected)') + ' · ' + usedRatio;

            session.images.push({
                id: nextImageId++,
                prompt: latest.prompt || '',
                imageUrl: data.image_url || null,
                imageDataUrl: data.image_data_url || null,
                filename: data.filename || null,
                aspectRatio: usedRatio,
                createdAt: Date.now(),
                kind: 'accurate',
                meta,
                accuracyTrace: includeTrace ? (data.accuracy_trace || null) : null,
                parentImageId: latest.id,
            });
            renderImages();
            renderHeader();
        } catch (error) {
            showImageError('Error: ' + (error.message || 'Unknown error'));
        } finally {
            hideImageActionLoading();
            if (accurateBtn) accurateBtn.disabled = false;
            if (accurateTraceBtn) accurateTraceBtn.disabled = false;
            if (refinedBtn) refinedBtn.disabled = false;
        }
    }

    // ---------------- Refined prompt image (vision QA → GPT prompt → Gemini) ----------------
    async function refinedPromptImageLatest() {
        const session = getActiveSession();
        const latest = getLatestImage();
        if (!session || !latest) return;

        const accurateBtn = document.getElementById('aiGetAccurateBtn');
        const accurateTraceBtn = document.getElementById('aiGetAccurateTraceBtn');
        const refinedBtn = document.getElementById('aiRefinedPromptImageBtn');
        if (accurateBtn) accurateBtn.disabled = true;
        if (accurateTraceBtn) accurateTraceBtn.disabled = true;
        if (refinedBtn) refinedBtn.disabled = true;
        showImageActionLoading('Refined prompt — analyzing image, rewriting prompt, generating…');
        clearImageError();

        try {
            const filename = latest.filename
                || (latest.imageUrl ? latest.imageUrl.split('/').pop() : null);
            const aspectRatio = getSelectedAspectRatio();
            const response = await fetch('/refined-prompt-image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename,
                    image_data_url: latest.imageDataUrl || latest.imageUrl,
                    original_prompt: latest.prompt || '',
                    aspect_ratio: aspectRatio,
                    session_id: 'ai_chat_' + session.id,
                }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed refined prompt regeneration');

            const refined = String(data.refined_prompt || '').trim();
            const usedRatio = data.aspect_ratio || aspectRatio;
            session.images.push({
                id: nextImageId++,
                prompt: refined || latest.prompt || '',
                imageUrl: data.image_url || null,
                imageDataUrl: data.image_data_url || null,
                filename: data.filename || null,
                aspectRatio: usedRatio,
                createdAt: Date.now(),
                kind: 'refined_prompt',
                meta: 'Refined prompt image (vision QA → GPT → Gemini) · ' + usedRatio,
                parentImageId: latest.id,
            });
            renderImages();
            renderHeader();
        } catch (error) {
            showImageError('Error: ' + (error.message || 'Unknown error'));
        } finally {
            hideImageActionLoading();
            if (accurateBtn) accurateBtn.disabled = false;
            if (accurateTraceBtn) accurateTraceBtn.disabled = false;
            if (refinedBtn) refinedBtn.disabled = false;
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
    function openImageFullscreen(src, fullPrompt, imageKind) {
        const overlay = document.getElementById('imageFullscreenOverlay');
        const img = document.getElementById('imageFullscreen');
        const caption = document.getElementById('imageFullscreenCaption');
        const viewBtn = document.getElementById('aiFullscreenViewPromptBtn');
        if (!overlay || !img || !src) return;
        img.src = src;
        const p = fullPrompt ? String(fullPrompt).trim() : '';
        if (caption) {
            const text = p
                ? ('Prompt: ' + p.slice(0, 140) + (p.length > 140 ? '…' : ''))
                : 'Generated image';
            caption.textContent = text;
        }
        if (viewBtn) {
            if (p) {
                viewBtn.hidden = false;
                viewBtn.textContent = imageKind === 'refined_prompt'
                    ? 'View refined prompt'
                    : 'View full prompt';
                viewBtn.onclick = function() {
                    openPromptModal(
                        imageKind === 'refined_prompt'
                            ? 'Refined generation prompt'
                            : 'Generation prompt',
                        p
                    );
                };
            } else {
                viewBtn.hidden = true;
                viewBtn.onclick = null;
            }
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
        const pm = document.getElementById('aiPromptModalOverlay');
        if (pm && pm.classList.contains('active')) {
            document.body.style.overflow = 'hidden';
        }
    }

    // ---------------- Prompt text modal (full generation / refined prompt) ----------------
    function openPromptModal(title, fullText) {
        const overlay = document.getElementById('aiPromptModalOverlay');
        const titleEl = document.getElementById('aiPromptModalTitle');
        const bodyEl = document.getElementById('aiPromptModalBody');
        if (!overlay || !titleEl || !bodyEl) return;
        titleEl.textContent = title || 'Prompt';
        bodyEl.textContent = fullText || '';
        overlay.classList.add('active');
        overlay.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
    }

    function closePromptModal() {
        const overlay = document.getElementById('aiPromptModalOverlay');
        if (!overlay) return;
        overlay.classList.remove('active');
        overlay.setAttribute('aria-hidden', 'true');
        const fs = document.getElementById('imageFullscreenOverlay');
        if (!fs || !fs.classList.contains('active')) {
            document.body.style.overflow = '';
        }
    }

    function copyPromptModalBody() {
        const bodyEl = document.getElementById('aiPromptModalBody');
        if (!bodyEl) return;
        const t = bodyEl.textContent || '';
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(t).catch(function() {});
        }
    }

    // ---------------- Clear messages in current chat ----------------
    function clearActiveChat() {
        const session = getActiveSession();
        if (!session) return;
        if (!session.messages.length && !session.images.length) return;
        if (!confirm('Clear all messages and images in this chat?')) return;
        session.messages = [];
        session.images = [];
        session.systemPromptOverride = null;
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
        initChatThemeMenu();
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
        const editCanvasBtn = document.getElementById('aiEditCanvasBtn');
        if (editCanvasBtn) editCanvasBtn.addEventListener('click', openCanvasEditorForLatest);
        const editCanvasInlineBtn = document.getElementById('aiEditCanvasBtnInline');
        if (editCanvasInlineBtn) editCanvasInlineBtn.addEventListener('click', openCanvasEditorForLatest);
        const accurateBtn = document.getElementById('aiGetAccurateBtn');
        if (accurateBtn) accurateBtn.addEventListener('click', () => getAccurateLatest(false));
        const accurateTraceBtn = document.getElementById('aiGetAccurateTraceBtn');
        if (accurateTraceBtn) accurateTraceBtn.addEventListener('click', () => getAccurateLatest(true));
        const refinedPromptBtn = document.getElementById('aiRefinedPromptImageBtn');
        if (refinedPromptBtn) refinedPromptBtn.addEventListener('click', refinedPromptImageLatest);
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
        document.addEventListener('keydown', function(e) {
            if (e.key !== 'Escape') return;
            if (isThemeMenuOpen()) {
                setThemeMenuOpen(false);
                return;
            }
            const promptOv = document.getElementById('aiPromptModalOverlay');
            if (promptOv && promptOv.classList.contains('active')) {
                closePromptModal();
                return;
            }
            closeImageFullscreen();
        });

        const promptModalOverlay = document.getElementById('aiPromptModalOverlay');
        const promptModalClose = document.getElementById('aiPromptModalClose');
        const promptModalCopy = document.getElementById('aiPromptModalCopy');
        if (promptModalClose) promptModalClose.addEventListener('click', closePromptModal);
        if (promptModalCopy) promptModalCopy.addEventListener('click', copyPromptModalBody);
        if (promptModalOverlay) {
            promptModalOverlay.addEventListener('click', function(e) {
                if (e.target === promptModalOverlay) closePromptModal();
            });
        }

        const previewImg = document.getElementById('aiImagePreview');
        if (previewImg) {
            previewImg.addEventListener('click', function() {
                if (previewImg.classList.contains('visible')) {
                    const last = getLatestImage();
                    openImageFullscreen(
                        previewImg.src,
                        last ? last.prompt : '',
                        last ? last.kind : undefined
                    );
                }
            });
        }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
