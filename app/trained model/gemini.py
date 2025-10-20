import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import requests
from urllib.parse import quote_plus
import json as _json
import base64
import time
from typing import Optional

# Load environment variables from the module's directory (ensures .env is found
# whether the process CWD is project root or another folder). Falls back to
# automatic search if the file is not present.
env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    # fallback to default search locations
    load_dotenv()

# Configure logging for helpful debug output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log whether a GEMINI_API_KEY was found at import time (mask actual key)
if os.getenv('GEMINI_API_KEY'):
    logger.info('GEMINI_API_KEY found in environment (length=%d)', len(os.getenv('GEMINI_API_KEY')))
else:
    logger.info('GEMINI_API_KEY not found in environment at import time')

# Optionally disable REST fallback in environments where only the client should be used
# Set GEMINI_DISABLE_REST_FALLBACK=1 or true in the environment to prevent REST calls.
DISABLE_REST_FALLBACK = str(os.getenv('GEMINI_DISABLE_REST_FALLBACK', '')).lower() in ('1', 'true', 'yes')

# We purposefully avoid importing google.generativeai at module import time
# because that package may not be installed in every environment. Instead
# we import it lazily inside functions and return helpful messages when
# the dependency or API key is missing.
genai = None

def _rest_generate(api_key: str, prompt: str, model_name: str = "models/gemini-2.5-flash", max_output_tokens: int = 1024, temperature: float = 0.7, full: bool = False):
    """Call the Generative Language REST endpoint as a fallback.

    Returns the generated text or an error string beginning with ❌.
    """
    try:
        if not api_key:
            # Try to obtain a bearer token from a service account file
            bearer = _get_service_account_bearer_token()
            if not bearer:
                return "❌ No API key provided for REST request and no service-account credentials found."
            auth_query = None
            headers = {"Content-Type": "application/json", "Accept": "application/json", "Authorization": f"Bearer {bearer}"}
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generate"
        else:
            auth_query = f"?key={quote_plus(api_key)}"
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generate{auth_query}"
        payload = {
            "prompt": {"text": prompt},
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        # If caller requested full response, return status/headers/text for diagnostics
        if full:
            try:
                return {"status": resp.status_code, "headers": dict(resp.headers), "text": resp.text}
            except Exception:
                return {"status": resp.status_code, "text": resp.text}

        try:
            j = resp.json()
        except Exception:
            return f"❌ REST: non-JSON response (status {resp.status_code})"

        # Handle API-level error
        if resp.status_code >= 400:
            # Try to extract message
            msg = j.get("error", {}).get("message") if isinstance(j, dict) else None
            if msg:
                return f"❌ REST error: {msg}"
            return f"❌ REST error: status={resp.status_code}"

        # Recent API returns {'candidates': [...]} or {'output': {'candidates': [...]}}
        candidates = None
        if isinstance(j, dict):
            candidates = j.get('candidates') or j.get('output', {}).get('candidates')

        if not candidates:
            # Maybe the text is in 'content' or 'text' top-level
            if isinstance(j, dict) and 'text' in j:
                return j['text']
            return "❌ REST: no candidates in response"

        first = candidates[0]
        # Candidate may have structure: {'content':[{'text': '...'}]} or {'content':{'parts':[{'text':...}]}}
        content = first.get('content') if isinstance(first, dict) else None
        if content:
            # handle list-of-parts
            if isinstance(content, list) and len(content) > 0:
                part = content[0]
                if isinstance(part, dict) and 'text' in part:
                    return part['text']
                if isinstance(part, str):
                    return part

            # handle object with parts
            parts = content.get('parts') if isinstance(content, dict) else None
            if parts and isinstance(parts, list) and len(parts) > 0:
                p0 = parts[0]
                if isinstance(p0, dict) and 'text' in p0:
                    return p0['text']
                if isinstance(p0, str):
                    return p0

        # fallback: try candidate['text']
        if isinstance(first, dict) and 'text' in first:
            return first['text']

        return "❌ REST: unable to extract text from response"
    except Exception as e:
        logging.exception('REST request failed')
        return f"❌ REST exception: {str(e)[:200]}"


def _get_service_account_bearer_token() -> Optional[str]:
    """If GOOGLE_APPLICATION_CREDENTIALS points to a service account JSON, obtain an OAuth2 access token.

    This avoids requiring an API key and will work when a service account with the proper scope is available.
    Returns bearer token string or None.
    """
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds_path:
        return None
    try:
        with open(creds_path, 'r', encoding='utf-8') as f:
            sa = _json.load(f)
        # Build JWT for OAuth2 token exchange
        header = {'alg': 'RS256', 'typ': 'JWT'}
        now = int(time.time())
        payload = {
            'iss': sa.get('client_email'),
            'scope': 'https://www.googleapis.com/auth/cloud-platform',
            'aud': 'https://oauth2.googleapis.com/token',
            'iat': now,
            'exp': now + 3600,
        }
        def _b64(u: bytes) -> bytes:
            return base64.urlsafe_b64encode(u).replace(b'=', b'')

        header_b = _b64(_json.dumps(header, separators=(',', ':')).encode('utf-8'))
        payload_b = _b64(_json.dumps(payload, separators=(',', ':')).encode('utf-8'))
        signing_input = header_b + b'.' + payload_b

        # Sign using the private key from the service account
        from cryptography.hazmat.primitives import serialization, hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.backends import default_backend

        key = sa.get('private_key').encode('utf-8')
        priv = serialization.load_pem_private_key(key, password=None, backend=default_backend())
        signature = priv.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
        signature_b = _b64(signature)
        jwt = (signing_input + b'.' + signature_b).decode('utf-8')

        # Exchange JWT for access token
        token_url = 'https://oauth2.googleapis.com/token'
        data = {
            'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
            'assertion': jwt,
        }
        r = requests.post(token_url, data=data, timeout=15)
        rj = r.json()
        access_token = rj.get('access_token')
        return access_token
    except Exception as e:
        logging.exception('Service account token exchange failed')
        return None

def get_gemini_solution(disease_name):
    """
    Get plant care advice from Gemini AI
    
    Args:
        disease_name (str): The predicted disease/condition name
        
    Returns:
        str: Formatted advice from Gemini AI
    """
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        msg = "❌ API key not found. Please set GEMINI_API_KEY in your environment or .env file."
        logger.warning(msg)
        return msg

    # Lazy import of the Google Generative AI client library. We prefer the
    # official client when available, but fall back to a REST adapter using
    # requests so the code works on Python versions where the client isn't
    # installable (for example older/newer Python mismatches).
    global genai
    genai_missing = False
    if genai is None:
        try:
            import google.generativeai as _genai
            genai = _genai
        except Exception:
            genai_missing = True
            logger.info('google.generativeai client not available; will try REST fallback')

    # Outer try: configure, prompt and call the model
    try:
        # Prefer to use the official client when present
        if not genai_missing:
            try:
                # Configure Gemini
                genai.configure(api_key=api_key)

                # Create model with proper generation config
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=2000,
                    temperature=0.7,
                )

                model = genai.GenerativeModel(
                    model_name="models/gemini-2.5-flash",
                    generation_config=generation_config
                )

                # Parse disease name for better prompt
                using_rest = False
            except Exception as e_client:
                logger.exception('Error using google.generativeai client, will fallback to REST: %s', e_client)
                genai_missing = True

        # If client is missing or errored, we'll use REST fallback
        if genai_missing:
            using_rest = True

        # Parse disease name for better prompt
        if '__' in disease_name:
            parts = disease_name.split('__')
            plant_name = parts[0].replace('_', ' ').strip() if len(parts) > 0 else "plant"
            condition = parts[1].replace('_', ' ').strip() if len(parts) > 1 else "condition"
        else:
            plant_name = "plant"
            condition = disease_name.replace('_', ' ').strip()

        is_healthy = "healthy" in condition.lower()

        # Create concise but detailed prompt to avoid token limits
        if is_healthy:
            prompt = f"""Give specific care plan for healthy {plant_name}:

FERTILIZER: NPK ratio, application rate, timing
WATERING: Frequency, amount, best schedule  
CARE: 2 key maintenance practices with success rates

Keep response under 300 words with specific numbers and percentages."""
        else:
            prompt = f"""Give treatment plan for {plant_name} with {condition}:

TREATMENT: Primary method with timing and 70-90% effectiveness rate
FERTILIZER: Specific NPK ratio and application schedule
WATERING: Exact frequency and amounts
PREVENTION: Top 2 prevention methods

Keep response under 300 words with specific numbers and percentages."""

        # Generate response with error handling
        text_result = None

        if not using_rest:
            try:
                response = model.generate_content(prompt)

                # Handle different response scenarios with better error handling
                if not response:
                    msg = "❌ No response received from Gemini AI."
                    logger.warning(msg)
                    return msg

                # Try direct text access first (simpler approach)
                try:
                    if hasattr(response, 'text') and response.text:
                        text_result = response.text
                except Exception:
                    pass

                # Try candidate approach
                try:
                    if getattr(response, 'candidates', None) and len(response.candidates) > 0:
                        candidate = response.candidates[0]

                        if hasattr(candidate, 'content') and candidate.content and getattr(candidate.content, 'parts', None):
                            if candidate.content.parts and len(candidate.content.parts) > 0:
                                maybe = candidate.content.parts[0]
                                # candidate content part may be a simple string or object
                                if isinstance(maybe, str):
                                    text_result = maybe
                                else:
                                    text_result = getattr(maybe, 'text', None) or getattr(maybe, 'content', None)

                        # Check for specific finish reasons (numeric codes observed)
                        finish = getattr(candidate, 'finish_reason', None)

                        # Helper to extract text from a candidate object/dict
                        def _extract_candidate_text(cand):
                            try:
                                # cand may be a dict-like or an object with attributes
                                if isinstance(cand, dict):
                                    content = cand.get('content')
                                    if content:
                                        # list-of-parts
                                        if isinstance(content, list) and len(content) > 0:
                                            part = content[0]
                                            if isinstance(part, dict) and 'text' in part:
                                                return part['text']
                                            if isinstance(part, str):
                                                return part
                                        # object with parts
                                        parts = content.get('parts') if isinstance(content, dict) else None
                                        if parts and isinstance(parts, list) and len(parts) > 0:
                                            p0 = parts[0]
                                            if isinstance(p0, dict) and 'text' in p0:
                                                return p0['text']
                                            if isinstance(p0, str):
                                                return p0
                                    if 'text' in cand:
                                        return cand['text']

                                # object-like candidate
                                try:
                                    if hasattr(cand, 'content') and getattr(cand.content, 'parts', None):
                                        p = cand.content.parts[0]
                                        if isinstance(p, str):
                                            return p
                                        return getattr(p, 'text', None) or getattr(p, 'content', None)
                                except Exception:
                                    pass

                                if hasattr(cand, 'text') and getattr(cand, 'text', None):
                                    return getattr(cand, 'text')
                            except Exception:
                                return None
                            return None

                        # If the response was cut off, try to return any partial text first,
                        # otherwise attempt a retry (client -> REST) with a larger token budget.
                        if finish == 2:
                            partial = None
                            try:
                                partial = _extract_candidate_text(candidate)
                            except Exception:
                                partial = None

                            if partial:
                                note = "\n\n[Note: response was truncated by the model; some content may be missing.]"
                                return partial + note

                            logger.warning('Response was cut off; attempting retry with larger max tokens')
                            # Try client-side retry first (if available)
                            try:
                                if not genai_missing:
                                    try:
                                        # Increase token limit conservatively
                                        current = getattr(generation_config, 'max_output_tokens', 2000)
                                        new_max = min(4000, int(current) * 2)
                                        generation_config.max_output_tokens = int(new_max)
                                        response2 = model.generate_content(prompt)
                                        # Try extract
                                        if getattr(response2, 'text', None):
                                            return response2.text
                                        if getattr(response2, 'candidates', None) and len(response2.candidates) > 0:
                                            cand2 = response2.candidates[0]
                                            txt2 = _extract_candidate_text(cand2)
                                            if txt2:
                                                return txt2
                                    except Exception as e_retry:
                                        logger.exception('Client retry failed: %s', e_retry)

                                # REST retry as a final attempt
                                try:
                                    rest_text = _rest_generate(api_key=api_key, prompt=prompt, model_name="models/gemini-2.5-flash", max_output_tokens=2048, temperature=0.7)
                                    if rest_text and not rest_text.startswith('❌'):
                                        return rest_text
                                except Exception as e_rest:
                                    logger.exception('REST retry failed: %s', e_rest)
                            except Exception:
                                logger.exception('Retries for truncated response encountered an error')

                            logger.warning('❌ Response was cut off and retries failed. Ask the user to shorten the prompt or increase token limits.')
                            return "❌ Response was cut off due to length and retries failed. Please try again with a shorter prompt."

                        if finish == 3:
                            msg = "❌ Response blocked for safety reasons. Please try a different request."
                            logger.warning(msg)
                            return msg
                        if finish == 4:
                            msg = "❌ Response blocked due to recitation. Please rephrase your request."
                            logger.warning(msg)
                            return msg
                except Exception as candidate_error:
                    msg = f"❌ Error processing response: {str(candidate_error)[:200]}"
                    logger.exception(msg)
                    return msg
            except Exception as e:
                logger.exception('Error while using client generate_content: %s', e)
                # fall through to REST fallback
                using_rest = True

        # If we still don't have a text_result and REST is available, try REST
        if (not text_result) and using_rest:
            rest_text = _rest_generate(api_key=api_key, prompt=prompt, model_name="models/gemini-2.5-flash", max_output_tokens=1024, temperature=0.7)
            if rest_text and not rest_text.startswith('❌'):
                return rest_text
            if rest_text:
                return rest_text

        if text_result:
            return text_result

        # If we reached here, response did not contain usable text
        msg = "❌ Empty response received. Please try again with a simpler request."
        logger.warning(msg)
        return msg

    except Exception as e:
        msg = f"❌ Error: {str(e)}"
        logger.exception(msg)
        return msg

def test_gemini_connection():
    """Test Gemini API connection.

    This wrapper uses the new debug_gemini helper which will try the client
    library first and then the REST fallback. It returns (ok: bool, message: str)
    for backward-compatible consumers.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False, "No API key found"

    try:
        dbg = debug_gemini(prompt="Say hello in 5 words")
        ok = dbg.get('ok', False)
        used = 'rest' if dbg.get('used_rest') else 'client' if dbg.get('used_client') else 'none'
        msg = f"ok={ok}, used={used}, message={str(dbg.get('message'))[:200]}"
        return ok, msg
    except Exception as e:
        return False, f"Error running debug_gemini: {e}"


def debug_gemini(prompt: str = "Say hello in 5 words") -> dict:
    """Return verbose debug information when calling Gemini.

    The returned dict contains keys:
      - ok: bool
      - used_client: bool
      - used_rest: bool
      - message: string (text or error)
      - raw: raw response object or dict (when available)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    result = {
        'ok': False,
        'used_client': False,
        'used_rest': False,
        'message': None,
        'raw': None,
    }

    if not api_key:
        result['message'] = '❌ No GEMINI_API_KEY found in environment.'
        return result

    # Try client library first
    try:
        import google.generativeai as _genai
        try:
            _genai.configure(api_key=api_key)
            gen = _genai.GenerativeModel(model_name="models/gemini-2.5-flash", generation_config=_genai.types.GenerationConfig(max_output_tokens=64, temperature=0.3))
            resp = gen.generate_content(prompt)
            result['used_client'] = True
            result['raw'] = resp
            # attempt to extract text
            text = None
            if hasattr(resp, 'text') and resp.text:
                text = resp.text
            else:
                try:
                    if getattr(resp, 'candidates', None) and len(resp.candidates) > 0:
                        cand = resp.candidates[0]
                        if hasattr(cand, 'content') and getattr(cand.content, 'parts', None):
                            p = cand.content.parts[0]
                            text = getattr(p, 'text', None) or (p if isinstance(p, str) else None)
                        else:
                            text = getattr(cand, 'text', None)
                except Exception:
                    text = None

            if text:
                result['ok'] = True
                result['message'] = text
            else:
                result['message'] = 'Client responded but text extraction failed or response empty.'
            return result
        except Exception as e:
            result['message'] = f'Client invocation error: {e}'
            # fall through to REST
    except Exception:
        result['message'] = 'Client library not available; will try REST fallback.'

    # REST fallback (only if not disabled)
    if DISABLE_REST_FALLBACK:
        result['used_rest'] = False
        result['message'] = result.get('message') or 'REST fallback disabled by environment.'
        return result

    try:
        rest_resp = _rest_generate(api_key=api_key, prompt=prompt, model_name="models/gemini-2.5-flash", max_output_tokens=256, temperature=0.3, full=True)
        result['used_rest'] = True
        result['raw'] = rest_resp
        # rest_resp is a dict with status/headers/text when full=True
        if isinstance(rest_resp, dict):
            status = rest_resp.get('status')
            text = rest_resp.get('text')
            result['rest_status'] = status
            result['rest_body'] = (text[:2000] if isinstance(text, str) else text)
            if status and status < 400 and text:
                result['ok'] = True
                result['message'] = text
            else:
                result['message'] = f'❌ REST: status={status}'
        else:
            result['message'] = str(rest_resp)
        return result
    except Exception as e:
        result['message'] = f'REST fallback error: {e}'
        return result