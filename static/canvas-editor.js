/**
 * Fabric.js canvas editor for vectorized medical illustrations.
 * Exposes window.openCanvasEditor({ filename, imageDataUrl, onSave }).
 */
(function () {
    'use strict';

    var overlay = null;
    var subtitleEl = null;
    var toolbarEl = null;
    var loadingEl = null;
    var errorEl = null;
    var workspaceEl = null;
    var fillColorInput = null;
    var fabricCanvas = null;
    var currentZoom = 1;
    var baseCanvasWidth = 800;
    var baseCanvasHeight = 600;
    var saveCallback = null;
    var isOpen = false;

    function getEl(id) {
        return document.getElementById(id);
    }

    function showOverlay() {
        if (!overlay) return;
        overlay.classList.add('active');
        overlay.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        isOpen = true;
    }

    function hideOverlay() {
        if (!overlay) return;
        overlay.classList.remove('active');
        overlay.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        isOpen = false;
    }

    function setLoading(visible, message) {
        if (loadingEl) {
            loadingEl.hidden = !visible;
            if (message) {
                var p = loadingEl.querySelector('p');
                if (p) p.textContent = message;
            }
        }
        if (workspaceEl) workspaceEl.hidden = visible;
        if (errorEl) errorEl.hidden = true;
        if (toolbarEl) toolbarEl.hidden = visible;
    }

    function showError(message) {
        setLoading(false);
        if (workspaceEl) workspaceEl.hidden = true;
        if (toolbarEl) toolbarEl.hidden = true;
        if (errorEl) {
            errorEl.hidden = false;
            errorEl.textContent = message || 'Something went wrong.';
        }
        if (subtitleEl) subtitleEl.textContent = 'Vectorization failed';
    }

    function setReady(message) {
        setLoading(false);
        if (workspaceEl) workspaceEl.hidden = false;
        if (toolbarEl) toolbarEl.hidden = false;
        if (errorEl) errorEl.hidden = true;
        if (subtitleEl) subtitleEl.textContent = message || 'Drag shapes, change colors, add text, resize or rotate.';
    }

    function destroyFabricCanvas() {
        if (fabricCanvas) {
            try {
                fabricCanvas.dispose();
            } catch (e) {
                /* ignore */
            }
            fabricCanvas = null;
        }
        var canvasEl = getEl('fabricCanvas');
        if (canvasEl && canvasEl.parentNode) {
            var parent = canvasEl.parentNode;
            var fresh = document.createElement('canvas');
            fresh.id = 'fabricCanvas';
            parent.replaceChild(fresh, canvasEl);
        }
    }

    function fitCanvasToContent() {
        if (!fabricCanvas) return;
        var objects = fabricCanvas.getObjects();
        var maxW = Math.min(window.innerWidth - 80, 1400);
        var maxH = Math.min(window.innerHeight - 200, 900);

        if (!objects.length) {
            fabricCanvas.setWidth(Math.max(320, Math.min(baseCanvasWidth, maxW)));
            fabricCanvas.setHeight(Math.max(240, Math.min(baseCanvasHeight, maxH)));
            fabricCanvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
            fabricCanvas.setZoom(1);
            fabricCanvas.renderAll();
            currentZoom = 1;
            return;
        }

        var bounds = fabricCanvas.getObjects().reduce(function (acc, obj) {
            var b = obj.getBoundingRect(true, true);
            if (!acc) {
                return { left: b.left, top: b.top, right: b.left + b.width, bottom: b.top + b.height };
            }
            return {
                left: Math.min(acc.left, b.left),
                top: Math.min(acc.top, b.top),
                right: Math.max(acc.right, b.left + b.width),
                bottom: Math.max(acc.bottom, b.top + b.height),
            };
        }, null);

        var pad = 32;
        var contentW = bounds.right - bounds.left;
        var contentH = bounds.bottom - bounds.top;
        var scale = Math.min(
            (maxW - pad * 2) / Math.max(contentW, 1),
            (maxH - pad * 2) / Math.max(contentH, 1),
            1
        );

        fabricCanvas.setWidth(Math.max(320, maxW));
        fabricCanvas.setHeight(Math.max(240, maxH));
        fabricCanvas.setZoom(1);
        fabricCanvas.setViewportTransform([1, 0, 0, 1, 0, 0]);

        var offsetX = (fabricCanvas.getWidth() - contentW * scale) / 2 - bounds.left * scale;
        var offsetY = (fabricCanvas.getHeight() - contentH * scale) / 2 - bounds.top * scale;
        fabricCanvas.setViewportTransform([scale, 0, 0, scale, offsetX, offsetY]);
        fabricCanvas.renderAll();
        currentZoom = scale;
    }

    function applyZoom(factor) {
        if (!fabricCanvas) return;
        var vpt = fabricCanvas.viewportTransform.slice();
        var zoom = vpt[0] * factor;
        zoom = Math.max(0.05, Math.min(zoom, 8));
        var center = fabricCanvas.getCenter();
        fabricCanvas.zoomToPoint(new fabric.Point(center.left, center.top), zoom);
        fabricCanvas.renderAll();
    }

    function resetZoomFit() {
        fitCanvasToContent();
    }

    function getActiveObject() {
        if (!fabricCanvas) return null;
        return fabricCanvas.getActiveObject();
    }

    function applyFillColor(color) {
        var obj = getActiveObject();
        if (!obj) return;
        if (obj.type === 'group') {
            obj.getObjects().forEach(function (child) {
                if ('fill' in child) child.set('fill', color);
                if ('stroke' in child && child.stroke) child.set('stroke', color);
            });
        } else {
            if ('fill' in obj) obj.set('fill', color);
            if ('stroke' in obj && obj.stroke) obj.set('stroke', color);
        }
        fabricCanvas.requestRenderAll();
    }

    function addTextObject() {
        if (!fabricCanvas) return;
        var text = new fabric.IText('Label', {
            left: fabricCanvas.getWidth() / 2 - 40,
            top: fabricCanvas.getHeight() / 2 - 12,
            fontFamily: 'Arial, Helvetica, sans-serif',
            fontSize: 22,
            fill: fillColorInput ? fillColorInput.value : '#1a1d24',
        });
        fabricCanvas.add(text);
        fabricCanvas.setActiveObject(text);
        text.enterEditing();
        text.selectAll();
        fabricCanvas.requestRenderAll();
    }

    function deleteSelection() {
        var obj = getActiveObject();
        if (!obj || !fabricCanvas) return;
        if (obj.type === 'activeSelection') {
            obj.getObjects().forEach(function (item) {
                fabricCanvas.remove(item);
            });
        } else {
            fabricCanvas.remove(obj);
        }
        fabricCanvas.discardActiveObject();
        fabricCanvas.requestRenderAll();
    }

    function moveLayer(direction) {
        var obj = getActiveObject();
        if (!obj || !fabricCanvas) return;
        if (direction === 'forward') {
            fabricCanvas.bringForward(obj);
        } else {
            fabricCanvas.sendBackwards(obj);
        }
        fabricCanvas.requestRenderAll();
    }

    function downloadBlob(filename, mime, content) {
        var blob = new Blob([content], { type: mime });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function exportPngDataUrl() {
        if (!fabricCanvas) return null;
        return fabricCanvas.toDataURL({
            format: 'png',
            quality: 1,
            multiplier: 3,
        });
    }

    function exportSvgString() {
        if (!fabricCanvas) return '';
        return fabricCanvas.toSVG();
    }

    function handleToolbarClick(event) {
        var btn = event.target.closest('[data-tool]');
        if (!btn) return;
        var tool = btn.getAttribute('data-tool');

        switch (tool) {
            case 'select':
                if (fabricCanvas) fabricCanvas.isDrawingMode = false;
                break;
            case 'text':
                addTextObject();
                break;
            case 'delete':
                deleteSelection();
                break;
            case 'forward':
                moveLayer('forward');
                break;
            case 'backward':
                moveLayer('backward');
                break;
            case 'zoom-in':
                applyZoom(1.15);
                break;
            case 'zoom-out':
                applyZoom(1 / 1.15);
                break;
            case 'zoom-reset':
                resetZoomFit();
                break;
            case 'download-svg':
                downloadBlob('canvas-edit-' + Date.now() + '.svg', 'image/svg+xml', exportSvgString());
                break;
            case 'download-png':
                var pngUrl = exportPngDataUrl();
                if (pngUrl) {
                    var link = document.createElement('a');
                    link.href = pngUrl;
                    link.download = 'canvas-edit-' + Date.now() + '.png';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }
                break;
            case 'save':
                if (typeof saveCallback === 'function') {
                    var dataUrl = exportPngDataUrl();
                    if (dataUrl) {
                        saveCallback(dataUrl);
                        closeCanvasEditor();
                    }
                }
                break;
            default:
                break;
        }
    }

    function loadSvgIntoCanvas(svgString) {
        return new Promise(function (resolve, reject) {
            if (typeof fabric === 'undefined') {
                reject(new Error('Fabric.js failed to load'));
                return;
            }

            destroyFabricCanvas();

            var canvasEl = getEl('fabricCanvas');
            if (!canvasEl) {
                reject(new Error('Canvas element not found'));
                return;
            }

            fabricCanvas = new fabric.Canvas('fabricCanvas', {
                selection: true,
                preserveObjectStacking: true,
                backgroundColor: '#ffffff',
            });

            fabric.loadSVGFromString(svgString, function (objects, options) {
                if (!objects || !objects.length) {
                    reject(new Error('No editable shapes found in vectorized image'));
                    return;
                }

                var svgEl = options && options.svg ? options.svg : null;
                var vb = null;
                if (svgEl) {
                    vb = svgEl.viewBox;
                    if (!vb && svgEl.getAttribute) {
                        var vbAttr = svgEl.getAttribute('viewBox');
                        if (vbAttr) {
                            var parts = vbAttr.split(/\s+/).map(Number);
                            if (parts.length === 4) {
                                vb = { width: parts[2], height: parts[3] };
                            }
                        }
                    }
                }

                if (vb && vb.width && vb.height) {
                    baseCanvasWidth = Math.ceil(vb.width);
                    baseCanvasHeight = Math.ceil(vb.height);
                }

                objects.forEach(function (obj) {
                    fabricCanvas.add(obj);
                });
                fabricCanvas.renderAll();
                fitCanvasToContent();
                resolve();
            }, function (_el, obj) {
                obj.set({
                    selectable: true,
                    evented: true,
                });
            });
        });
    }

    async function fetchVectorSvg(filename, imageDataUrl) {
        var payload = {};
        if (imageDataUrl) {
            payload.image_data_url = imageDataUrl;
        }
        if (filename) {
            payload.filename = filename;
        }

        var response = await fetch('/vectorize-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        var data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Vectorization failed');
        }
        if (!data.svg) {
            throw new Error('Server returned no SVG');
        }
        return data.svg;
    }

    function closeCanvasEditor() {
        destroyFabricCanvas();
        hideOverlay();
        saveCallback = null;
        setLoading(true, 'Converting image to editable vectors…');
        if (subtitleEl) subtitleEl.textContent = 'Vectorizing image…';
    }

    async function openCanvasEditor(options) {
        options = options || {};
        saveCallback = typeof options.onSave === 'function' ? options.onSave : null;

        if (typeof fabric === 'undefined') {
            alert('Canvas editor could not load (Fabric.js). Check your network connection and refresh.');
            return;
        }

        overlay = getEl('canvasEditorOverlay');
        subtitleEl = getEl('canvasEditorSubtitle');
        toolbarEl = getEl('canvasEditorToolbar');
        loadingEl = getEl('canvasEditorLoading');
        errorEl = getEl('canvasEditorError');
        workspaceEl = getEl('canvasEditorWorkspace');
        fillColorInput = getEl('canvasFillColor');

        if (!overlay) {
            console.error('Canvas editor overlay not found');
            return;
        }

        var filename = options.filename || '';
        var imageDataUrl = options.imageDataUrl || '';

        if (!filename && !imageDataUrl) {
            showError('No image reference provided.');
            showOverlay();
            return;
        }

        setLoading(true);
        showOverlay();

        try {
            var svg = await fetchVectorSvg(filename, imageDataUrl);
            await loadSvgIntoCanvas(svg);
            setReady('Select shapes to move, recolor, or transform. Add text with Add text.');
        } catch (err) {
            showError(err.message || 'Failed to open canvas editor');
        }
    }

    function initCanvasEditor() {
        overlay = getEl('canvasEditorOverlay');
        toolbarEl = getEl('canvasEditorToolbar');
        fillColorInput = getEl('canvasFillColor');

        var closeBtn = getEl('canvasEditorCloseBtn');
        if (closeBtn) {
            closeBtn.addEventListener('click', closeCanvasEditor);
        }
        if (toolbarEl) {
            toolbarEl.addEventListener('click', handleToolbarClick);
        }
        if (fillColorInput) {
            fillColorInput.addEventListener('input', function () {
                applyFillColor(fillColorInput.value);
            });
        }

        document.addEventListener('keydown', function (event) {
            if (!isOpen) return;
            if (event.key === 'Escape') {
                closeCanvasEditor();
                return;
            }
            if (event.key === 'Delete' || event.key === 'Backspace') {
                var tag = (document.activeElement && document.activeElement.tagName) || '';
                if (tag === 'INPUT' || tag === 'TEXTAREA') return;
                var active = getActiveObject();
                if (active && active.isEditing) return;
                event.preventDefault();
                deleteSelection();
            }
        });
    }

    window.openCanvasEditor = openCanvasEditor;
    window.closeCanvasEditor = closeCanvasEditor;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initCanvasEditor);
    } else {
        initCanvasEditor();
    }
})();
