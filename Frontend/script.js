// Main JavaScript file

// Navigation
function navigateTo(page) {
    window.location.href = page;
}

// File Upload Handling
function handleFileUpload(inputId, previewId = null) {
    const input = document.getElementById(inputId);
    const file = input.files[0];
    
    if (!file) return null;
    
    // Check file type
    const validImageTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    const validVideoTypes = ['video/mp4', 'video/avi', 'video/mov'];
    
    if (inputId.includes('photo') && !validImageTypes.includes(file.type)) {
        API.showMessage('Please upload a valid image file (JPEG, PNG)', 'error');
        input.value = '';
        return null;
    }
    
    if (inputId.includes('video') && !validVideoTypes.includes(file.type)) {
        API.showMessage('Please upload a valid video file (MP4, AVI, MOV)', 'error');
        input.value = '';
        return null;
    }
    
    // Show preview if preview element exists
    if (previewId) {
        const preview = document.getElementById(previewId);
        if (inputId.includes('photo')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.innerHTML = `<img src="${e.target.result}" alt="Preview" style="max-width: 100%; border-radius: 4px;">`;
            };
            reader.readAsDataURL(file);
        } else {
            preview.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        }
    }
    
    return file;
}

// Create FormData from form
function createFormData(formId) {
    const form = document.getElementById(formId);
    const formData = new FormData();
    
    const inputs = form.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        if (input.type === 'file') {
            if (input.files[0]) {
                formData.append(input.name || input.id, input.files[0]);
            }
        } else if (input.value) {
            formData.append(input.name || input.id, input.value);
        }
    });
    
    return formData;
}

// Load Targets
async function loadTargets() {
    const result = await API.getTargetPersons();
    
    if (result.success) {
        const targets = result.data;
        const container = document.getElementById('targetsContainer');
        
        if (!container) return;
        
        if (targets.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-user-slash"></i>
                    <h3>No Target Persons</h3>
                    <p>Create your first target person to start searching</p>
                    <button class="btn btn-primary" onclick="showCreateTargetModal()">
                        <i class="fas fa-plus"></i> Add Target Person
                    </button>
                </div>
            `;
            return;
        }
        
        let html = '<div class="targets-grid">';
        
        targets.forEach(target => {
            html += `
                <div class="target-card">
                    <div class="target-image">
                        <img src="${target.photo_path}" alt="${target.name}" onerror="this.src='https://via.placeholder.com/300x300?text=No+Image'">
                    </div>
                    <div class="target-info">
                        <h3>${target.name}</h3>
                        ${target.description ? `<p>${target.description}</p>` : ''}
                        <div class="target-meta">
                            <span><i class="fas fa-calendar"></i> ${new Date(target.created_at).toLocaleDateString()}</span>
                        </div>
                    </div>
                    <div class="target-actions">
                        <button class="btn btn-sm btn-primary" onclick="useTargetForSearch('${target.id}')">
                            <i class="fas fa-search"></i> Search
                        </button>
                        <button class="btn btn-sm btn-secondary" onclick="viewTargetDetails('${target.id}')">
                            <i class="fas fa-eye"></i> View
                        </button>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
    } else {
        API.showMessage('Failed to load targets: ' + result.error, 'error');
    }
}

// Dashboard Stats
async function loadDashboardStats() {
    // This would load actual stats from API
    const stats = {
        totalSearches: 0,
        matchesFound: 0,
        activeJobs: 0,
        recentMatches: []
    };
    
    // Update UI
    document.getElementById('totalSearches').textContent = stats.totalSearches;
    document.getElementById('matchesFound').textContent = stats.matchesFound;
    document.getElementById('activeJobs').textContent = stats.activeJobs;
}

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    // Check authentication on all pages except login
    if (!window.location.pathname.endsWith('index.html')) {
        if (!API.isLoggedIn()) {
            window.location.href = 'index.html';
            return;
        }
        
        // Update user info
        const user = API.currentUser();
        if (user) {
            const userElements = document.querySelectorAll('.user-name, .user-email');
            userElements.forEach(el => {
                if (el.classList.contains('user-name')) {
                    el.textContent = user.full_name || user.username;
                }
                if (el.classList.contains('user-email')) {
                    el.textContent = user.email;
                }
            });
        }
        
        // Load appropriate data based on page
        const path = window.location.pathname;
        
        if (path.includes('dashboard.html')) {
            loadDashboardStats();
        } else if (path.includes('targets.html')) {
            loadTargets();
        } else if (path.includes('cctv.html')) {
            // Load CCTV list
        } else if (path.includes('search.html')) {
            // Load search jobs
        } else if (path.includes('results.html')) {
            // Load results
        }
    }
    
    // Initialize tooltips
    const tooltips = document.querySelectorAll('[title]');
    tooltips.forEach(el => {
        el.addEventListener('mouseenter', showTooltip);
        el.addEventListener('mouseleave', hideTooltip);
    });
});

// Tooltip functions
function showTooltip(event) {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = event.target.title;
    document.body.appendChild(tooltip);
    
    const rect = event.target.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 5 + 'px';
    
    event.target._tooltip = tooltip;
}

function hideTooltip(event) {
    if (event.target._tooltip) {
        event.target._tooltip.remove();
        delete event.target._tooltip;
    }
}

// Logout function
function logout() {
    if (confirm('Are you sure you want to logout?')) {
        API.logout();
    }
}

// Export to window
window.navigateTo = navigateTo;
window.handleFileUpload = handleFileUpload;
window.logout = logout;