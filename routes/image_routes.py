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
            filename, image_bytes, image_data_url = image_service.generate_image(
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
                new_filename, edited_bytes, edited_image_data_url = (
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
