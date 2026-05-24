import requests
import json
import base64
import time


def create_discord_account(
    email, username, password, date_of_birth,
    captcha_api_key, invite_code,
    proxy=None,
    base_url='https://discord.com',
    captcha_base_url='https://2captcha.com',
):
    result = {
        'token': None, 'user_id': None, 'username': username,
        'registered': False, 'captcha_solved': False,
        'server_joined': False, 'fingerprint': None, 'headers_used': {},
    }

    session = requests.Session()
    if proxy:
        session.proxies = {'http': proxy, 'https': proxy}

    # Anti-detection: X-Super-Properties with browser metadata
    super_props = {
        'os': 'Windows',
        'browser': 'Chrome',
        'device': '',
        'system_locale': 'en-US',
        'browser_user_agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'browser_version': '120.0.0.0',
        'os_version': '10',
        'referrer': '',
        'referring_domain': '',
        'release_channel': 'stable',
        'client_build_number': 254573,
        'client_event_source': None,
    }
    x_super = base64.b64encode(json.dumps(super_props).encode()).decode()

    headers = {
        'User-Agent': super_props['browser_user_agent'],
        'X-Super-Properties': x_super,
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': base_url,
        'Referer': base_url + '/register',
    }
    session.headers.update(headers)
    result['headers_used'] = dict(headers)

    # Step 1: Get fingerprint
    try:
        fp_resp = session.get(base_url + '/api/v9/auth/fingerprint')
        if fp_resp.status_code == 200:
            result['fingerprint'] = fp_resp.json().get('fingerprint')
            session.headers['X-Fingerprint'] = result['fingerprint']
    except Exception:
        pass

    # Step 2: Attempt registration (will require captcha)
    reg_payload = {
        'email': email,
        'username': username,
        'password': password,
        'date_of_birth': date_of_birth,
        'consent': True,
        'gift_code_sku_id': None,
        'promotional_email_opt_in': False,
    }
    if result['fingerprint']:
        reg_payload['fingerprint'] = result['fingerprint']

    reg_resp = session.post(
        base_url + '/api/v9/auth/register',
        json=reg_payload,
    )

    # Step 3: Handle captcha challenge
    if reg_resp.status_code == 400:
        reg_data = reg_resp.json()
        if 'captcha_key' in reg_data:
            sitekey = reg_data.get('captcha_sitekey', '')

            # Create captcha task at solver API
            task_resp = session.post(
                captcha_base_url + '/in.php',
                json={
                    'key': captcha_api_key,
                    'method': 'hcaptcha',
                    'sitekey': sitekey,
                    'pageurl': base_url + '/register',
                    'json': 1,
                },
            )
            task_id = task_resp.json().get('request')

            # Poll for solution
            captcha_token = None
            for _ in range(30):
                time.sleep(0.05)
                poll_resp = session.get(
                    captcha_base_url + '/res.php',
                    params={
                        'key': captcha_api_key,
                        'action': 'get',
                        'id': task_id,
                        'json': 1,
                    },
                )
                poll_data = poll_resp.json()
                if poll_data.get('status') == 1:
                    captcha_token = poll_data['request']
                    break

            if captcha_token:
                result['captcha_solved'] = True
                reg_payload['captcha_key'] = captcha_token
                reg_resp = session.post(
                    base_url + '/api/v9/auth/register',
                    json=reg_payload,
                )

    # Step 4: Process registration response
    if reg_resp.status_code == 200:
        reg_data = reg_resp.json()
        result['token'] = reg_data.get('token')
        result['user_id'] = reg_data.get('user_id')
        result['registered'] = True
        session.headers['Authorization'] = result['token']
    else:
        return result

    # Step 5: Join server via invite
    try:
        inv_resp = session.post(
            base_url + '/api/v9/invites/' + invite_code,
            json={},
        )
        if inv_resp.status_code == 200:
            result['server_joined'] = True
    except Exception:
        pass

    return result
