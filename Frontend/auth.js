// Auth Functions
function showTab(evt, tabName) {
    // Hide all forms
    document.querySelectorAll('.auth-form').forEach(form => {
        form.classList.remove('active');
    });

    // Show selected form
    const form = document.getElementById(`${tabName}Form`);
    if (form) form.classList.add('active');

    // Update tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Mark clicked tab active
    if (evt && evt.target) evt.target.classList.add('active');
}

function togglePassword(button, inputId) {
    const input = document.getElementById(inputId);
    const icon = button.querySelector('i');
    if (!input || !icon) return;

    if (input.type === 'password') {
        input.type = 'text';
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
    } else {
        input.type = 'password';
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
    }
}

async function handleLogin(event) {
    event.preventDefault();
    
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;
    
    // Basic validation
    if (!username || !password) {
        API.showMessage('Please fill in all fields', 'error');
        return;
    }
    
    // Show loading
    const submitBtn = event.target.querySelector('.btn-primary');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Logging in...';
    submitBtn.disabled = true;
    
    // Call API
    const result = await API.loginUser(username, password);
    
    if (result.success) {
        API.showMessage('Login successful! Redirecting...', 'success');
        
        // Redirect to dashboard after 1 second
        setTimeout(() => {
            window.location.href = 'dashboard.html';
        }, 1000);
    } else {
        API.showMessage(result.error, 'error');
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
}

async function handleRegister(event) {
    event.preventDefault();
    
    const username = document.getElementById('regUsername').value.trim();
    const email = document.getElementById('regEmail').value.trim();
    const fullName = document.getElementById('regFullName').value.trim();
    const password = document.getElementById('regPassword').value;
    const confirmPassword = document.getElementById('regConfirmPassword').value;
    
    // Validation
    if (!username || !email || !password || !confirmPassword) {
        API.showMessage('Please fill in all required fields', 'error');
        return;
    }
    
    if (!API.validateEmail(email)) {
        API.showMessage('Please enter a valid email address', 'error');
        return;
    }
    
    if (!API.validatePassword(password)) {
        API.showMessage('Password must be at least 6 characters long', 'error');
        return;
    }
    
    if (password !== confirmPassword) {
        API.showMessage('Passwords do not match', 'error');
        return;
    }
    
    // Show loading
    const submitBtn = event.target.querySelector('.btn-primary');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating account...';
    submitBtn.disabled = true;
    
    // Prepare user data
    const userData = {
        username: username,
        email: email,
        password: password,
        full_name: fullName
    };
    
    // Call API
    const result = await API.registerUser(userData);
    
    if (result.success) {
        API.showMessage('Registration successful! Please login.', 'success');
        
        // Switch to login tab
        setTimeout(() => {
            showTab('login');
            // Auto-fill username
            document.getElementById('loginUsername').value = username;
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }, 1500);
    } else {
        API.showMessage(result.error, 'error');
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', async function() {
    // Check if user is already logged in
    if (API.isLoggedIn() && window.location.pathname.endsWith('index.html')) {
        window.location.href = 'dashboard.html';
    }
    
    // Check backend status
    const isBackendHealthy = await API.checkBackendStatus();
    const statusElement = document.getElementById('backendStatus');
    
    if (statusElement) {
        if (isBackendHealthy) {
            statusElement.textContent = 'Connected';
            statusElement.className = 'status-indicator active';
        } else {
            statusElement.textContent = 'Disconnected';
            statusElement.className = 'status-indicator inactive';
            API.showMessage('Cannot connect to backend server. Please ensure it is running on http://localhost:8000', 'error', 10000);
        }
    }
    
    // Attach form handlers
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }
    
    if (registerForm) {
        registerForm.addEventListener('submit', handleRegister);
    }
    
    // Demo credentials button
    const demoBtn = document.createElement('button');
    demoBtn.type = 'button';
    demoBtn.className = 'btn btn-secondary';
    demoBtn.style.marginTop = '10px';
    demoBtn.style.width = '100%';
    demoBtn.innerHTML = '<i class="fas fa-magic"></i> Use Demo Credentials';
    demoBtn.onclick = () => {
        document.getElementById('loginUsername').value = 'admin';
        document.getElementById('loginPassword').value = 'password123';
        API.showMessage('Demo credentials filled. Click Login to continue.', 'info');
    };
    
    const formActions = document.querySelector('#loginForm .form-actions');
    if (formActions) {
        formActions.appendChild(demoBtn);
    }
});