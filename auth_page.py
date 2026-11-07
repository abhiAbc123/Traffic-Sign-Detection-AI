"""
TrafficVision PRO - Standalone User Authentication Page (Signup & Login)
Replicating the clean, high-conversion modern card aesthetic.
"""

AUTH_HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrafficVision PRO - User Authentication</title>
    <!-- Immediate Authentication Routing Guard -->
    <script>
        (function() {
            try {
                var raw = localStorage.getItem('traffic_user');
                if (raw) {
                    var u = JSON.parse(raw);
                    if (u && (u.username || u.id)) {
                        window.location.replace('/');
                    }
                }
            } catch(e) {}
        })();
    </script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 50%, #6366f1 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
            position: relative;
            overflow-x: hidden;
        }

        /* Ambient glowing circles */
        .ambient-glow-1 {
            position: absolute;
            top: -90px;
            left: -90px;
            width: 420px;
            height: 420px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.16) 0%, rgba(255, 255, 255, 0) 70%);
            pointer-events: none;
        }

        .ambient-glow-2 {
            position: absolute;
            bottom: -130px;
            right: -90px;
            width: 480px;
            height: 480px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.35) 0%, rgba(99, 102, 241, 0) 70%);
            pointer-events: none;
        }

        /* Floating Modern Card */
        .auth-card {
            background: #ffffff;
            width: 100%;
            max-width: 440px;
            border-radius: 36px;
            padding: 46px 38px 40px;
            box-shadow: 0 25px 60px -15px rgba(15, 23, 42, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.25);
            text-align: center;
            position: relative;
            z-index: 10;
            animation: cardFadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes cardFadeIn {
            from {
                opacity: 0;
                transform: translateY(20px) scale(0.97);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        /* Top Brand Icon Box */
        .brand-icon-box {
            width: 60px;
            height: 60px;
            border-radius: 18px;
            margin: 0 auto 16px;
            background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 22px -4px rgba(79, 70, 229, 0.45);
            color: #ffffff;
        }

        .brand-title-wrap {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            margin-bottom: 3px;
        }

        .brand-name {
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.6px;
        }

        .pro-badge {
            background: #dcfce7;
            color: #15803d;
            font-size: 11px;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 9999px;
            letter-spacing: 0.5px;
            line-height: 1;
        }

        .brand-subtitle {
            font-size: 13px;
            color: #64748b;
            font-weight: 500;
            margin-bottom: 24px;
        }

        /* Pill Switch Tabs */
        .tabs-pill {
            display: flex;
            background: #f1f5f9;
            border-radius: 14px;
            padding: 4px;
            margin-bottom: 20px;
            gap: 4px;
        }

        .tab-choice {
            flex: 1;
            padding: 9px 12px;
            border: none;
            background: transparent;
            color: #64748b;
            font-weight: 600;
            font-size: 13.5px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
            font-family: inherit;
        }

        .tab-choice.active {
            background: #ffffff;
            color: #4f46e5;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            font-weight: 700;
        }

        /* Main Form Heading */
        .auth-heading {
            font-size: 28px;
            font-weight: 800;
            color: #4f46e5;
            margin-bottom: 22px;
            letter-spacing: -0.6px;
        }

        /* Alert notifications */
        .alert-box {
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 18px;
            display: none;
            text-align: left;
            line-height: 1.4;
        }

        .alert-box.error {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #b91c1c;
            display: block;
        }

        .alert-box.success {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #15803d;
            display: block;
        }

        /* Field Inputs */
        .field-group {
            text-align: left;
            margin-bottom: 16px;
        }

        .field-label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 6px;
        }

        .field-input {
            width: 100%;
            padding: 14px 18px;
            border: 1.5px solid #e2e8f0;
            border-radius: 14px;
            font-size: 15px;
            color: #0f172a;
            outline: none;
            transition: all 0.2s;
            background: #ffffff;
            font-family: inherit;
        }

        .field-input::placeholder {
            color: #94a3b8;
        }

        .field-input:focus {
            border-color: #4f46e5;
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.12);
        }

        /* Primary Purple Button */
        .btn-primary {
            width: 100%;
            background: #4f46e5;
            color: #ffffff;
            font-weight: 700;
            font-size: 16px;
            padding: 15px;
            border-radius: 14px;
            border: none;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
            margin-top: 8px;
            margin-bottom: 12px;
            font-family: inherit;
        }

        .btn-primary:hover {
            background: #4338ca;
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(79, 70, 229, 0.45);
        }

        .btn-primary:active {
            transform: translateY(0);
        }

        .btn-primary:disabled {
            opacity: 0.65;
            cursor: not-allowed;
            transform: none;
        }

        /* Secondary Solid Blue Button */
        .btn-demo {
            width: 100%;
            background: #1877f2;
            color: #ffffff;
            font-weight: 600;
            font-size: 15px;
            padding: 14px;
            border-radius: 14px;
            border: none;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 9px;
            box-shadow: 0 4px 12px rgba(24, 119, 242, 0.25);
            margin-bottom: 0;
            font-family: inherit;
        }

        .btn-demo:hover {
            background: #166fe5;
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(24, 119, 242, 0.35);
        }

        .btn-demo:active {
            transform: translateY(0);
        }
    </style>
</head>
<body>
    <div class="ambient-glow-1"></div>
    <div class="ambient-glow-2"></div>

    <div class="auth-card">
        <!-- Top App Icon -->
        <div class="brand-icon-box">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="3" width="20" height="14" rx="2"></rect>
                <line x1="8" y1="21" x2="16" y2="21"></line>
                <line x1="12" y1="17" x2="12" y2="21"></line>
                <circle cx="12" cy="10" r="3"></circle>
            </svg>
        </div>

        <!-- Header Branding -->
        <div class="brand-title-wrap">
            <span class="brand-name">TrafficVision</span>
            <span class="pro-badge">PRO</span>
        </div>
        <p class="brand-subtitle">Autonomous Traffic & Sign Detection Engine</p>

        <!-- Mode Toggle Tabs -->
        <div class="tabs-pill">
            <button type="button" class="tab-choice active" id="tab-signup" onclick="setAuthMode('signup')">Sign Up</button>
            <button type="button" class="tab-choice" id="tab-login" onclick="setAuthMode('login')">Log In</button>
        </div>

        <!-- Title -->
        <h2 class="auth-heading" id="auth-title">User Signup</h2>

        <!-- Notification Alert -->
        <div id="alert-box" class="alert-box"></div>

        <!-- Form -->
        <form id="auth-form" onsubmit="handleAuthSubmit(event)">
            <!-- Email / Identifier -->
            <div class="field-group" id="group-email">
                <label class="field-label" id="label-email">Email</label>
                <input type="text" id="input-email" class="field-input" placeholder="Enter email" required autocomplete="email">
            </div>

            <!-- Username (Hidden in Login Mode) -->
            <div class="field-group" id="group-username">
                <label class="field-label">Username</label>
                <input type="text" id="input-username" class="field-input" placeholder="Enter username" autocomplete="username">
            </div>

            <!-- Password -->
            <div class="field-group" id="group-password">
                <label class="field-label">Password</label>
                <input type="password" id="input-password" class="field-input" placeholder="Enter password" required autocomplete="current-password">
            </div>

            <!-- Primary Action Button -->
            <button type="submit" class="btn-primary" id="btn-submit">Sign Up</button>

            <!-- Secondary Demo Operator Access Button -->
            <button type="button" class="btn-demo" id="btn-demo" onclick="handleDemoLogin()">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
                </svg>
                Instant Demo Access (Demo Operator)
            </button>
        </form>
    </div>

    <script>
        let currentMode = '__INITIAL_MODE__';

        window.addEventListener('DOMContentLoaded', () => {
            const urlPath = window.location.pathname.toLowerCase();
            const hasAccount = localStorage.getItem('traffic_has_account') === 'true';

            if (urlPath.includes('signup')) {
                setAuthMode('signup', false);
            } else if (urlPath.includes('login')) {
                // If a new visitor lands on /login without an existing account, route to signup
                if (!hasAccount) {
                    setAuthMode('signup', true);
                } else {
                    setAuthMode('login', false);
                }
            } else {
                setAuthMode(hasAccount ? 'login' : 'signup', false);
            }
        });

        function showAlert(message, type = 'error') {
            const box = document.getElementById('alert-box');
            box.className = `alert-box ${type}`;
            box.innerText = message;
            box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        function clearAlert() {
            const box = document.getElementById('alert-box');
            box.className = 'alert-box';
            box.innerText = '';
        }

        function setAuthMode(mode, updateHistory = true) {
            currentMode = mode;
            clearAlert();

            const tabSignup = document.getElementById('tab-signup');
            const tabLogin = document.getElementById('tab-login');
            const title = document.getElementById('auth-title');
            const groupUsername = document.getElementById('group-username');
            const labelEmail = document.getElementById('label-email');
            const inputEmail = document.getElementById('input-email');
            const inputUsername = document.getElementById('input-username');
            const btnSubmit = document.getElementById('btn-submit');
            if (mode === 'login') {
                tabLogin.classList.add('active');
                tabSignup.classList.remove('active');
                title.innerText = 'User Login';
                labelEmail.innerText = 'Email or Username';
                inputEmail.placeholder = 'Enter email or username';
                groupUsername.style.display = 'none';
                inputUsername.removeAttribute('required');
                btnSubmit.innerText = 'Log In';

                if (updateHistory && window.location.pathname !== '/login') {
                    window.history.pushState({}, '', '/login');
                }
            } else {
                tabSignup.classList.add('active');
                tabLogin.classList.remove('active');
                title.innerText = 'User Signup';
                labelEmail.innerText = 'Email';
                inputEmail.placeholder = 'Enter email';
                groupUsername.style.display = 'block';
                inputUsername.setAttribute('required', 'required');
                btnSubmit.innerText = 'Sign Up';

                if (updateHistory && window.location.pathname !== '/signup') {
                    window.history.pushState({}, '', '/signup');
                }
            }
        }

        function toggleAuthMode() {
            setAuthMode(currentMode === 'login' ? 'signup' : 'login');
        }

        async function handleAuthSubmit(evt) {
            evt.preventDefault();
            clearAlert();

            const btn = document.getElementById('btn-submit');
            const origText = btn.innerText;
            btn.disabled = true;
            btn.innerText = 'Processing...';

            const email = document.getElementById('input-email').value.trim();
            const password = document.getElementById('input-password').value;
            const username = document.getElementById('input-username').value.trim();

            try {
                if (currentMode === 'signup') {
                    if (!username) {
                        throw new Error('Please enter a username.');
                    }
                    if (password.length < 6) {
                        throw new Error('Password must be at least 6 characters.');
                    }

                    const res = await fetch('/api/auth/signup', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email, username, password, role: 'operator' })
                    });
                    const data = await res.json();

                    if (!data.success) {
                        throw new Error(data.error || 'Failed to sign up.');
                    }

                    localStorage.setItem('traffic_user', JSON.stringify(data.user));
                    localStorage.setItem('traffic_auth_token', data.token);
                    localStorage.setItem('traffic_has_account', 'true');
                    document.cookie = "traffic_has_account=1; path=/; max-age=31536000; SameSite=Lax";
                    document.cookie = "traffic_logged_in=1; path=/; max-age=604800; SameSite=Lax";

                    showAlert('Account created successfully! Redirecting to dashboard...', 'success');
                    setTimeout(() => {
                        window.location.replace('/');
                    }, 450);

                } else {
                    // Login Mode
                    const res = await fetch('/api/auth/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ username_or_email: email, password })
                    });
                    const data = await res.json();

                    if (!data.success) {
                        throw new Error(data.error || 'Invalid credentials.');
                    }

                    localStorage.setItem('traffic_user', JSON.stringify(data.user));
                    localStorage.setItem('traffic_auth_token', data.token);
                    localStorage.setItem('traffic_has_account', 'true');
                    document.cookie = "traffic_has_account=1; path=/; max-age=31536000; SameSite=Lax";
                    document.cookie = "traffic_logged_in=1; path=/; max-age=604800; SameSite=Lax";

                    showAlert('Signed in successfully! Redirecting to dashboard...', 'success');
                    setTimeout(() => {
                        window.location.replace('/');
                    }, 450);
                }
            } catch (err) {
                showAlert(err.message, 'error');
                btn.disabled = false;
                btn.innerText = origText;
            }
        }

        async function handleDemoLogin() {
            clearAlert();
            const btnDemo = document.getElementById('btn-demo');
            const origText = btnDemo.innerHTML;
            btnDemo.disabled = true;
            btnDemo.innerText = 'Verifying Operator Credentials...';

            try {
                const res = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username_or_email: 'demo_operator',
                        password: 'traffic2026'
                    })
                });
                const data = await res.json();

                if (!data.success) {
                    throw new Error(data.error || 'Demo login failed.');
                }

                localStorage.setItem('traffic_user', JSON.stringify(data.user));
                localStorage.setItem('traffic_auth_token', data.token);
                localStorage.setItem('traffic_has_account', 'true');
                document.cookie = "traffic_has_account=1; path=/; max-age=31536000; SameSite=Lax";
                document.cookie = "traffic_logged_in=1; path=/; max-age=604800; SameSite=Lax";

                showAlert('Verified Demo Operator access! Redirecting to Dashboard...', 'success');
                setTimeout(() => {
                    window.location.replace('/');
                }, 400);
            } catch (err) {
                showAlert('Demo sign-in failed: ' + err.message, 'error');
                btnDemo.disabled = false;
                btnDemo.innerHTML = origText;
            }
        }
    </script>
</body>
</html>
"""

