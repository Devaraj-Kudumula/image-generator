"""
Image routes: generate image, serve image, edit image.
"""
import time
import traceback
import logging
from io import BytesIO

from flask import request, jsonify, send_file, send_from_directory

import config
from app_state import state
from services import image_service

logger = logging.getLogger(__name__)


def register(app):
    @app.route('/generate-image', methods=['POST'])
    def generate_image():
        """
        Generate an image using Google Gemini based on the provided prompt
        """
        request_start = time.time()
        logger.info("=" * 50)
        logger.info("[/generate-image] Request received")

        try:
            data = request.get_json()
            logger.info(
                "Request data keys: %s",
                list(data.keys()) if data else 'None',
            )
            prompt = data.get('prompt', '')

            if not prompt:
                logger.warning("Request missing prompt")
                return jsonify({'error': 'Prompt is required'}), 400

            logger.info("Prompt length: %d", len(prompt))

            if not config.GOOGLE_API_KEY:
                logger.error("Google API key not configured")
                return jsonify({
                    'error': 'Google Generative AI API key not configured. Please set GOOGLE_GENERATIVE_AI_API_KEY environment variable.'
                }), 500

            if not state.gemini_client:
                logger.error("Gemini client not initialized")
                return jsonify({
                    'error': 'Gemini client not initialized'
                }), 500

            logger.info("Generating image with prompt: %s...", prompt[:100])

            logger.info("Calling Gemini API...")
            api_start = time.time()
            filename, image_bytes, image_data_url, gemini_usage = image_service.generate_image(
                prompt
            )
            api_time = time.time() - api_start
            logger.info("Gemini API response received in %.2fs", api_time)
            logger.info("Extracting image...")

            image_url = (
                image_data_url
                if config.IS_SERVERLESS and image_data_url
                else f'{request.host_url}images/{filename}'
            )

            request_time = time.time() - request_start
            logger.info("[/generate-image] Success in %.2fs", request_time)
            logger.info("Image URL: %s", image_url)
            logger.info("=" * 50)

            return jsonify({
                'image_url': image_url,
                'filename': filename,
                'image_data_url': image_data_url,
                'success': True,
                'usage': {'gemini': gemini_usage or {}},
            })

        except ValueError as e:
            err_msg = str(e)
            if "No image generated" in err_msg and "Error processing" not in err_msg:
                return jsonify({
                    'error': 'No image generated in response. Check server logs for details.'
                }), 500
            return jsonify({'error': err_msg}), 500
        except Exception as e:
            request_time = time.time() - request_start
            logger.error(
                "[/generate-image] Error after %.2fs: %s",
                request_time,
                e,
            )
            logger.error(traceback.format_exc())
            logger.info("=" * 50)
            return jsonify({'error': f'Error generating image: {str(e)}'}), 500

    @app.route('/images/<filename>')
    def serve_image(filename):
        """
        Serve generated images from in-memory store or, if not found, from static dir (local).
        """
        logger.info("Serving image: %s", filename)
        image_bytes = image_service.get_image_bytes(filename)
        if image_bytes is not None:
            return send_file(
                BytesIO(image_bytes),
                mimetype='image/png',
                as_attachment=False,
                download_name=filename,
            )
        if not config.IS_SERVERLESS and (config.IMAGES_DIR / filename).exists():
            return send_from_directory(
                config.IMAGES_DIR.resolve(), filename
            )
        return jsonify({'error': 'Image not found'}), 404

    @app.route('/edit-image', methods=['POST'])
    def edit_image():
        """
        Edit an existing image based on user-requested changes using Google Gemini.
        """
        request_start = time.time()
        logger.info("=" * 50)
        logger.info("[/edit-image] Request received")

        try:
            data = request.get_json()
            logger.info(
                "Request data keys: %s",
                list(data.keys()) if data else 'None',
            )
            filename = data.get('filename', '')
            changes = data.get('changes', '')
            image_data_url = data.get('image_data_url', '')

            if not filename and not image_data_url:
                logger.warning("Request missing filename and image_data_url")
                return jsonify({
                    'error': 'Either filename or image_data_url is required'
                }), 400

            if not changes:
                logger.warning("Request missing changes")
                return jsonify({'error': 'Changes are required'}), 400

            logger.info(
                "Filename: %s, Changes: %s...",
                filename,
                changes[:100],
            )

            if not config.GOOGLE_API_KEY:
                logger.error("Google API key not configured")
                return jsonify({
                    'error': 'Google Generative AI API key not configured. Please set GOOGLE_GENERATIVE_AI_API_KEY environment variable.'
                }), 500

            if not state.gemini_client:
                logger.error("Gemini client not initialized")
                return jsonify({
                    'error': 'Gemini client not initialized'
                }), 500

            logger.info("Calling Gemini API for image editing...")
            api_start = time.time()
            try:
                new_filename, edited_bytes, edited_image_data_url, gemini_usage = (
                    image_service.edit_image(
                        filename, changes, image_data_url or None
                    )
                )
            except ValueError as e:
                msg = str(e)
                if "File not found" in msg:
                    return jsonify({'error': msg}), 404
                if "No edited image" in msg and "Error processing" not in msg:
                    return jsonify({
                        'error': 'No edited image generated in response. Check server logs for details.'
                    }), 500
                return jsonify({'error': msg}), 500
            api_time = time.time() - api_start
            logger.info("Gemini API response received in %.2fs", api_time)

            image_url = (
                edited_image_data_url
                if config.IS_SERVERLESS and edited_image_data_url
                else f'{request.host_url}images/{new_filename}'
            )

            request_time = time.time() - request_start
            logger.info("[/edit-image] Success in %.2fs", request_time)
            logger.info("Edited Image URL: %s", image_url)
            logger.info("=" * 50)

            return jsonify({
                'image_url': image_url,
                'filename': new_filename,
                'image_data_url': edited_image_data_url,
                'success': True,
                'usage': {'gemini': gemini_usage or {}},
            })

        except Exception as e:
            request_time = time.time() - request_start
            logger.error(
                "[/edit-image] Error after %.2fs: %s",
                request_time,
                e,
            )
            logger.error(traceback.format_exc())
            logger.info("=" * 50)
            return jsonify({
                'error': f'Error editing image: {str(e)}'
            }), 500

    @app.route('/get-accurate', methods=['POST'])
    def get_accurate():
        """
        Detect label/arrow flaws in an image with GPT-4o vision, then fix them
        iteratively using Gemini (max 3 flaws per pass, max 5 passes).
        """
        request_start = time.time()
        logger.info("=" * 50)
        logger.info("[/get-accurate] Request received")

        try:
            data = request.get_json()
            logger.info(
                "Request data keys: %s",
                list(data.keys()) if data else 'None',
            )
            filename = data.get('filename', '')
            image_data_url = data.get('image_data_url', '')
            original_prompt = data.get('original_prompt', '') or data.get('prompt', '')
            include_trace = bool(data.get('include_trace'))

            if not filename and not image_data_url:
                logger.warning("Request missing filename and image_data_url")
                return jsonify({
                    'error': 'Either filename or image_data_url is required'
                }), 400

            if not config.GOOGLE_API_KEY:
                logger.error("Google API key not configured")
                return jsonify({
                    'error': 'Google Generative AI API key not configured.'
                }), 500

            if not state.gemini_client:
                logger.error("Gemini client not initialized")
                return jsonify({'error': 'Gemini client not initialized'}), 500

            logger.info("Running accuracy refinement for: %s", filename)
            api_start = time.time()

            try:
                (
                    final_filename,
                    _,
                    final_data_url,
                    flaws_count,
                    iterations,
                    accuracy_trace,
                    accurate_usage,
                ) = image_service.get_accurate_image(
                    filename,
                    image_data_url or None,
                    original_prompt or None,
                    collect_trace=include_trace,
                )
            except ValueError as e:
                msg = str(e)
                if "File not found" in msg:
                    return jsonify({'error': msg}), 404
                return jsonify({'error': msg}), 500

            api_time = time.time() - api_start
            logger.info(
                "Accuracy refinement done in %.2fs — %d flaw(s), %d pass(es)",
                api_time, flaws_count, iterations,
            )

            image_url = (
                final_data_url
                if config.IS_SERVERLESS and final_data_url
                else f'{request.host_url}images/{final_filename}'
            )

            request_time = time.time() - request_start
            logger.info("[/get-accurate] Success in %.2fs", request_time)
            logger.info("=" * 50)

            payload = {
                'image_url': image_url,
                'filename': final_filename,
                'image_data_url': final_data_url,
                'flaws_detected': flaws_count,
                'iterations': iterations,
                'success': True,
                'usage': accurate_usage or {},
            }
            if include_trace and accuracy_trace is not None:
                payload['accuracy_trace'] = accuracy_trace
            return jsonify(payload)

        except Exception as e:
            request_time = time.time() - request_start
            logger.error(
                "[/get-accurate] Error after %.2fs: %s",
                request_time,
                e,
            )
            logger.error(traceback.format_exc())
            logger.info("=" * 50)
            return jsonify({
                'error': f'Error during accuracy refinement: {str(e)}'
            }), 500

    @app.route('/refined-prompt-image', methods=['POST'])
    def refined_prompt_image():
        """
        Vision QA on the current image vs. prompt, GPT refines the full generation
        prompt, then Gemini generates a new image from scratch.
        """
        request_start = time.time()
        logger.info("=" * 50)
        logger.info("[/refined-prompt-image] Request received")

        try:
            data = request.get_json()
            filename = (data or {}).get('filename', '')
            image_data_url = (data or {}).get('image_data_url', '')
            original_prompt = (data or {}).get('original_prompt', '') or (data or {}).get('prompt', '')
            include_trace = bool((data or {}).get('include_trace'))

            if not filename and not image_data_url:
                return jsonify({
                    'error': 'Either filename or image_data_url is required'
                }), 400

            if not config.GOOGLE_API_KEY:
                return jsonify({
                    'error': 'Google Generative AI API key not configured.'
                }), 500

            if not state.gemini_client:
                return jsonify({'error': 'Gemini client not initialized'}), 500

            if not state.openai_api_key:
                return jsonify({'error': 'OpenAI API key not configured'}), 500

            api_start = time.time()
            try:
                (
                    final_filename,
                    _,
                    final_data_url,
                    refined_prompt,
                    vision_analysis,
                    regen_trace,
                    regen_usage,
                ) = image_service.refined_prompt_regenerate_image(
                    filename,
                    image_data_url or None,
                    original_prompt or None,
                    collect_trace=include_trace,
                )
            except ValueError as e:
                msg = str(e)
                if "File not found" in msg:
                    return jsonify({'error': msg}), 404
                return jsonify({'error': msg}), 500

            api_time = time.time() - api_start
            logger.info(
                "[/refined-prompt-image] Done in %.2fs (refined prompt length=%d)",
                api_time,
                len(refined_prompt or ''),
            )

            image_url = (
                final_data_url
                if config.IS_SERVERLESS and final_data_url
                else f'{request.host_url}images/{final_filename}'
            )

            request_time = time.time() - request_start
            logger.info("[/refined-prompt-image] Success in %.2fs", request_time)
            logger.info("=" * 50)

            payload = {
                'image_url': image_url,
                'filename': final_filename,
                'image_data_url': final_data_url,
                'refined_prompt': refined_prompt,
                'vision_analysis': vision_analysis,
                'success': True,
                'usage': regen_usage or {},
            }
            if include_trace and regen_trace is not None:
                payload['refined_regen_trace'] = regen_trace
            return jsonify(payload)

        except Exception as e:
            request_time = time.time() - request_start
            logger.error(
                "[/refined-prompt-image] Error after %.2fs: %s",
                request_time,
                e,
            )
            logger.error(traceback.format_exc())
            logger.info("=" * 50)
            return jsonify({
                'error': f'Error during refined prompt regeneration: {str(e)}'
            }), 500
