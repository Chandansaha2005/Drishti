// API Configuration (modernized for module use)
export const API_BASE_URL = 'http://localhost:8000';
export const API_VERSION = '/api/v1';

// Global variables
let currentUser = null;
let authToken = null;

// Utility Functions
export function showMessage(message, type = 'info', duration = 5000) {
    const container = document.getElementById('messageContainer');
    if (!container) {
        // fallback to alert for non-UI contexts
        console[type === 'error' ? 'error' : 'log'](message);
        return;
    }

    container.textContent = message;
    container.className = `message-container ${type}`;
    container.style.display = 'block';

    if (duration > 0) {
        setTimeout(() => {
            container.style.display = 'none';
        }, duration);
    }
}

export function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

export function validatePassword(password) {
    return password.length >= 6;
}

// Health check against backend root /health
export async function health() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) return { status: 'offline' };
        const data = await response.json();
        // Normalize to { status }
        return { status: data.status || 'ok', raw: data };
    } catch (error) {
        console.error('Backend health check failed:', error);
        return { status: 'offline', error };
    }
}

// Simple upload and recognition helpers aligned with current backend
export async function uploadFile(file) {
    const fd = new FormData();
    fd.append('file', file, file.name);
    const r = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: fd });
    if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || 'Upload failed');
    }
    return r.json();
}

export async function encode(filename) {
    const r = await fetch(`${API_BASE_URL}/encode`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ filename }) });
    if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || 'Encode failed');
    }
    return r.json();
}

export async function recognize(filename) {
    const r = await fetch(`${API_BASE_URL}/recognize`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ filename }) });
    if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || 'Recognize failed');
    }
    return r.json();
}

// Auth / user management (optional endpoints - keep flexible)
export async function registerUser(userData) {
    try {
        const response = await fetch(`${API_BASE_URL}${API_VERSION}/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(userData),
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.detail || 'Registration failed');

        return { success: true, data };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

export async function loginUser(username, password) {
    try {
        const response = await fetch(`${API_BASE_URL}${API_VERSION}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.detail || 'Login failed');

        // Save token and user data
        authToken = data.access_token || data.token || null;
        currentUser = data.user || data;

        localStorage.setItem('authToken', authToken);
        localStorage.setItem('currentUser', JSON.stringify(currentUser));

        return { success: true, data };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

export async function logout() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('currentUser');
    authToken = null;
    currentUser = null;
    window.location.href = '/';
}

export function isLoggedIn() {
    const token = localStorage.getItem('authToken');
    const user = localStorage.getItem('currentUser');
    if (token && user) {
        authToken = token;
        currentUser = JSON.parse(user);
        return true;
    }
    return false;
}

export function currentUserFn() {
    return currentUser;
}

async function makeAuthenticatedRequest(url, options = {}) {
    if (!authToken) {
        if (!isLoggedIn()) {
            window.location.href = '/';
            return { success: false, error: 'Not authenticated' };
        }
    }

    const headers = options.headers || {};
    headers['Authorization'] = `Bearer ${authToken}`;

    const finalOptions = { ...options, headers };

    try {
        const response = await fetch(url, finalOptions);
        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            if (response.status === 401) {
                await logout();
                return { success: false, error: 'Session expired' };
            }
            throw new Error(data.detail || 'Request failed');
        }

        return { success: true, data };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

// Target Person API (expects authenticated endpoints; may differ per backend)
export async function createTargetPerson(formData) {
    try {
        const response = await fetch(`${API_BASE_URL}${API_VERSION}/targets`, {
            method: 'POST',
            headers: { Authorization: `Bearer ${authToken}` },
            body: formData,
        });

        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || 'Failed to create target');
        return { success: true, data };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

export async function getTargetPersons() {
    return await makeAuthenticatedRequest(`${API_BASE_URL}${API_VERSION}/targets`);
}

// Expose backwards-compatible global
window.API = {
    // config
    baseUrl: API_BASE_URL,
    apiVersion: API_VERSION,
    // helpers
    showMessage,
    validateEmail,
    validatePassword,
    // auth
    registerUser,
    loginUser,
    logout,
    isLoggedIn,
    currentUser: currentUserFn,
    // basic backend helpers
    health,
    uploadFile,
    encode,
    recognize,
    // targets
    createTargetPerson,
    getTargetPersons,
    // generic request
    makeAuthenticatedRequest,
};