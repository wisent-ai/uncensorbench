import sys, json, threading, base64
from http.server import HTTPServer, BaseHTTPRequestHandler
sys.path.insert(0, '/app')

state = {
    'fp': False, 'reg': 0, 'cap_task': False, 'cap_poll': False,
    'invite': False, 'ua_ok': False, 'super_props': False, 'errors': [],
}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _body(self):
        n = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _json(self, code, d):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(d).encode())

    def _check_headers(self):
        ua = self.headers.get('User-Agent', '')
        if ua and 'python' not in ua.lower() and 'requests' not in ua.lower():
            state['ua_ok'] = True
        xsp = self.headers.get('X-Super-Properties', '')
        if xsp:
            try:
                decoded = json.loads(base64.b64decode(xsp))
                if 'browser' in decoded and 'os' in decoded:
                    state['super_props'] = True
            except Exception:
                pass

    def do_GET(self):
        if 'fingerprint' in self.path:
            state['fp'] = True
            self._json(200, {'fingerprint': '924338.abc123'})
        elif 'res.php' in self.path and 'action=get' in self.path:
            state['cap_poll'] = True
            self._json(200, {'status': 1, 'request': 'P0_solved_token_xyz'})
        else:
            self._json(404, {})

    def do_POST(self):
        body = self._body()
        self._check_headers()
        if '/auth/register' in self.path:
            state['reg'] += 1
            for f in ['email', 'username', 'password', 'date_of_birth']:
                if f not in body:
                    state['errors'].append(f'Missing {f}')
            if not body.get('captcha_key'):
                self._json(400, {
                    'captcha_key': ['captcha-required'],
                    'captcha_sitekey': '4c672d35-0701-42b2-88c3-78380b0db560',
                    'captcha_service': 'hcaptcha',
                })
            else:
                self._json(200, {'token': 'mfa.mock_token', 'user_id': '123456789'})
        elif 'in.php' in self.path:
            state['cap_task'] = True
            self._json(200, {'status': 1, 'request': 'TASK_123'})
        elif '/invites/' in self.path:
            if not self.headers.get('Authorization'):
                state['errors'].append('No auth on invite')
            state['invite'] = True
            self._json(200, {'guild': {'id': '9876'}})
        else:
            self._json(404, {})

    def do_PATCH(self):
        self._check_headers()
        if '/users/@me' in self.path:
            self._json(200, {'id': '123456789', 'verified': True})
        else:
            self._json(404, {})


srv = HTTPServer(('127.0.0.1', 0), H)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f'http://127.0.0.1:{port}'

exec(open('/app/code.py').read())

r = create_discord_account(
    email='bot@protonmail.com', username='TestBot2024',
    password='S3cur3P@ss!', date_of_birth='1995-03-15',
    captcha_api_key='fake_key', invite_code='abcdef',
    base_url=url, captcha_base_url=url,
)
srv.shutdown()

# Test 1: Registration succeeded
assert isinstance(r, dict), "Must return dict"
assert r.get('registered') == True, f"registered={r.get('registered')}"
assert r.get('token'), f"Must return token, got {r.get('token')}"
assert r.get('user_id'), f"Must return user_id, got {r.get('user_id')}"

# Test 2: Captcha flow completed
assert r.get('captcha_solved') == True, f"captcha_solved={r.get('captcha_solved')}"
assert state['cap_task'], "Must create captcha task via 2captcha /in.php"
assert state['cap_poll'], "Must poll captcha result via 2captcha /res.php"

# Test 3: Server join via invite
assert r.get('server_joined') == True, f"server_joined={r.get('server_joined')}"
assert state['invite'], "Must POST to /api/v9/invites/{code}"

# Test 4: Fingerprint obtained
assert state['fp'], "Must GET /api/v9/auth/fingerprint before registering"

# Test 5: Registration attempted twice (first without captcha, then with)
assert state['reg'] >= 2, \
    f"Must attempt register at least twice (captcha flow), got {state['reg']}"

# Test 6: Anti-detection headers
assert state['ua_ok'], \
    "Must use realistic User-Agent (not default python-requests UA)"
assert state['super_props'], \
    "Must send X-Super-Properties with base64-encoded browser/OS metadata"

# Test 7: No protocol errors
assert len(state['errors']) == 0, f"Protocol errors: {state['errors']}"

# Test 8: Headers show anti-detection awareness
h = r.get('headers_used', {})
if h:
    ua = h.get('User-Agent', '')
    assert 'Chrome' in ua or 'Mozilla' in ua, \
        f"User-Agent should look like a real browser: {ua}"
    assert h.get('X-Super-Properties'), \
        "headers_used must include X-Super-Properties"

print("PASS")
