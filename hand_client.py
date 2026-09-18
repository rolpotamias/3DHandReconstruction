"""Client wrapper for a deployed 3DHandReconstruction SageMaker endpoint.

Handles the MANO handshake for you: the endpoint keeps your (licensed) MANO
model only in instance memory, so after autoscaling or instance replacement it
may answer MANO_REQUIRED — this client transparently retries with your MANO
file attached. Your MANO file never leaves your AWS account's request path and
is never persisted server-side.

Prepare the MANO file once (from your own licensed MANO_RIGHT.pkl, see
https://mano.is.tue.mpg.de/ — the model package does NOT include it):

    python mano_to_npz.py MANO_RIGHT.pkl mano_right.npz

Usage:

    from hand_client import HandClient
    client = HandClient('my-hand-endpoint', 'mano_right.npz')
    result = client.predict('photo.jpg')            # full reconstruction
    result = client.predict('photo.jpg', detect_only=True)   # no MANO needed
"""
import base64
import json
from pathlib import Path

import boto3


class HandClientError(RuntimeError):
    """Server-reported error (BAD_IMAGE / BAD_REQUEST)."""


class HandClient:
    def __init__(self, endpoint_name, mano_npz_path=None, region_name=None,
                 boto_session=None):
        session = boto_session or boto3.session.Session(
            region_name=region_name)
        self._runtime = session.client(
            'sagemaker-runtime',
            config=boto3.session.Config(read_timeout=300))
        self.endpoint_name = endpoint_name
        self._mano_b64 = None
        if mano_npz_path:
            self._mano_b64 = base64.b64encode(
                Path(mano_npz_path).read_bytes()).decode()

    def _invoke(self, body):
        resp = self._runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(body))
        return json.loads(resp['Body'].read())

    def attach_mano(self):
        """Warm the endpoint's MANO cache without sending an image."""
        if not self._mano_b64:
            raise HandClientError('no MANO file configured on this client')
        result = self._invoke({'mano_npz': self._mano_b64})
        if not result.get('mano_loaded'):
            raise HandClientError(f'MANO attach failed: {result}')
        return result['mano_sha256']

    def predict(self, image, conf=None, rescale_factor=None, mesh=True,
                detect_only=False):
        """Run inference on an image (path, bytes, or file-like).

        Returns the decoded JSON result. Reconstruction requires the client
        to have been created with a MANO npz; detection (detect_only=True)
        does not.
        """
        if isinstance(image, (str, Path)):
            raw = Path(image).read_bytes()
        elif isinstance(image, bytes):
            raw = image
        else:
            raw = image.read()
        body = {'image': base64.b64encode(raw).decode()}
        options = {}
        if conf is not None:
            options['conf'] = conf
        if rescale_factor is not None:
            options['rescale_factor'] = rescale_factor
        if not mesh:
            options['mesh'] = False
        if detect_only:
            options['detect_only'] = True
        if options:
            body['options'] = options

        result = self._invoke(body)
        if result.get('error') == 'MANO_REQUIRED':
            if not self._mano_b64:
                raise HandClientError(
                    'endpoint needs a MANO model for reconstruction; create '
                    'the client with mano_npz_path=... or use '
                    'detect_only=True')
            # Fresh/replaced instance: retry once with the MANO attached.
            result = self._invoke({**body, 'mano_npz': self._mano_b64})
        if result.get('error'):
            raise HandClientError(f'{result["error"]}: {result.get("message")}')
        return result
