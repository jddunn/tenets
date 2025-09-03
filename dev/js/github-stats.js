// GitHub Stats Fetcher with Codecov Integration
// Fetches repository statistics and CI/CD badges

(function() {
    'use strict';
    
    // Configuration
    const CONFIG = {
        owner: 'jddunn',
        repo: 'tenets',
        updateInterval: 300000, // 5 minutes
        animationDuration: 1500,
        githubAPI: 'https://api.github.com',
        codecovAPI: 'https://codecov.io/api/gh',
        cacheKey: 'tenets_github_stats',
        cacheExpiry: 300000 // 5 minutes
    };
    
    // ==============================================
    // STATS MANAGER CLASS
    // ==============================================
    
    class GitHubStatsManager {
        constructor() {
            this.stats = {
                stars: 0,
                forks: 0,
                issues: 0,
                watchers: 0,
                contributors: 0,
                commits: 0,
                releases: 0,
                language: '',
                coverage: 0,
                build: 'passing'
            };
            
            this.elements = {
                stars: document.getElementById('stars-count'),
                forks: document.getElementById('forks-count'),
                issues: document.getElementById('issues-count'),
                watchers: document.getElementById('watchers-count'),
                contributors: document.getElementById('contributors-count'),
                commits: document.getElementById('commits-count'),
                releases: document.getElementById('releases-count'),
                language: document.getElementById('main-language'),
                coverage: document.getElementById('coverage-badge'),
                build: document.getElementById('build-status')
            };
            
            this.init();
        }
        
        async init() {
            // Check cache first
            const cached = this.getCachedStats();
            if (cached) {
                this.stats = cached;
                this.updateUI(false); // No animation for cached data
            }
            
            // Fetch fresh data
            await this.fetchAllStats();
            
            // Set up periodic updates
            setInterval(() => this.fetchAllStats(), CONFIG.updateInterval);
            
            // Add hover effects
            this.setupInteractions();
        }
        
        // ==============================================
        // DATA FETCHING
        // ==============================================
        
        async fetchAllStats() {
            try {
                // Parallel fetch all data
                const [repoData, contributorsData, releasesData, commitsData] = await Promise.all([
                    this.fetchRepoStats(),
                    this.fetchContributors(),
                    this.fetchReleases(),
                    this.fetchCommits()
                ]);
                
                // Fetch coverage data separately (might fail)
                const coverageData = await this.fetchCoverage().catch(() => null);
                
                // Process and update stats
                if (repoData) {
                    this.stats.stars = repoData.stargazers_count;
                    this.stats.forks = repoData.forks_count;
                    this.stats.issues = repoData.open_issues_count;
                    this.stats.watchers = repoData.subscribers_count;
                    this.stats.language = repoData.language;
                }
                
                if (contributorsData) {
                    this.stats.contributors = contributorsData.length;
                }
                
                if (releasesData && releasesData.length > 0) {
                    this.stats.releases = releasesData.length;
                    this.stats.latestRelease = releasesData[0].tag_name;
                }
                
                if (commitsData) {
                    // Get total commits (approximation from last page number)
                    const linkHeader = commitsData.headers?.link || '';
                    const match = linkHeader.match(/page=(\d+)>; rel="last"/);
                    this.stats.commits = match ? parseInt(match[1]) * 30 : commitsData.length;
                }
                
                if (coverageData) {
                    this.stats.coverage = coverageData.coverage;
                }
                
                // Update UI with animations
                this.updateUI(true);
                
                // Cache the stats
                this.cacheStats();
                
            } catch (error) {
                console.error('Error fetching GitHub stats:', error);
                this.handleError(error);
            }
        }
        
        async fetchRepoStats() {
            const response = await fetch(`${CONFIG.githubAPI}/repos/${CONFIG.owner}/${CONFIG.repo}`);
            if (!response.ok) throw new Error('Failed to fetch repo stats');
            return response.json();
        }
        
        async fetchContributors() {
            const response = await fetch(`${CONFIG.githubAPI}/repos/${CONFIG.owner}/${CONFIG.repo}/contributors?per_page=100`);
            if (!response.ok) return [];
            return response.json();
        }
        
        async fetchReleases() {
            const response = await fetch(`${CONFIG.githubAPI}/repos/${CONFIG.owner}/${CONFIG.repo}/releases?per_page=10`);
            if (!response.ok) return [];
            return response.json();
        }
        
        async fetchCommits() {
            const response = await fetch(`${CONFIG.githubAPI}/repos/${CONFIG.owner}/${CONFIG.repo}/commits?per_page=1`);
            if (!response.ok) return [];
            
            // Get link header for pagination info
            return {
                data: await response.json(),
                headers: {
                    link: response.headers.get('link')
                }
            };
        }
        
        async fetchCoverage() {
            // Try multiple coverage services
            const codecovUrl = `https://codecov.io/gh/${CONFIG.owner}/${CONFIG.repo}/branch/main/graph/badge.svg`;
            
            // For now, we'll just set the badge image
            // In production, you'd want to parse the actual coverage percentage
            if (this.elements.coverage) {
                const img = document.createElement('img');
                img.src = codecovUrl;
                img.alt = 'Code Coverage';
                img.style.height = '20px';
                this.elements.coverage.innerHTML = '';
                this.elements.coverage.appendChild(img);
            }
            
            return { coverage: 'N/A' };
        }
        
        // ==============================================
        // UI UPDATES
        // ==============================================
        
        updateUI(animate = true) {
            Object.keys(this.elements).forEach(key => {
                const element = this.elements[key];
                if (!element) return;
                
                const value = this.stats[key];
                
                if (key === 'coverage' || key === 'build') {
                    // These are handled separately
                    return;
                }
                
                if (animate && typeof value === 'number') {
                    this.animateNumber(element, value);
                } else {
                    element.textContent = this.formatValue(key, value);
                }
                
                // Remove loading state
                element.classList.remove('loading');
            });
            
            // Update build status if element exists
            if (this.elements.build) {
                this.updateBuildStatus();
            }
            
            // Show stats container
            const statsContainers = document.querySelectorAll('.github-stats-container, .stats-card');
            statsContainers.forEach(container => {
                container.style.opacity = '1';
                container.style.transform = 'translateY(0)';
            });
        }
        
        animateNumber(element, target) {
            const start = parseInt(element.textContent) || 0;
            const duration = CONFIG.animationDuration;
            const startTime = performance.now();
            
            const animate = (currentTime) => {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                
                // Easing function (ease-out-cubic)
                const easeProgress = 1 - Math.pow(1 - progress, 3);
                const current = Math.floor(start + (target - start) * easeProgress);
                
                element.textContent = this.formatNumber(current);
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    element.textContent = this.formatNumber(target);
                    
                    // Add pulse effect on completion
                    element.classList.add('pulse');
                    setTimeout(() => element.classList.remove('pulse'), 600);
                }
            };
            
            requestAnimationFrame(animate);
        }
        
        formatValue(key, value) {
            if (typeof value === 'number') {
                return this.formatNumber(value);
            }
            return value || 'N/A';
        }
        
        formatNumber(num) {
            if (num >= 1000000) {
                return (num / 1000000).toFixed(1) + 'M';
            } else if (num >= 10000) {
                return (num / 1000).toFixed(0) + 'k';
            } else if (num >= 1000) {
                return (num / 1000).toFixed(1) + 'k';
            }
            return num.toString();
        }
        
        updateBuildStatus() {
            // Fetch GitHub Actions status
            fetch(`${CONFIG.githubAPI}/repos/${CONFIG.owner}/${CONFIG.repo}/actions/workflows`)
                .then(res => res.json())
                .then(data => {
                    if (data.workflows && data.workflows.length > 0) {
                        // Get the main workflow
                        const mainWorkflow = data.workflows.find(w => w.name === 'CI') || data.workflows[0];
                        
                        // Fetch latest run
                        return fetch(`${CONFIG.githubAPI}/repos/${CONFIG.owner}/${CONFIG.repo}/actions/workflows/${mainWorkflow.id}/runs?per_page=1`);
                    }
                })
                .then(res => res && res.json())
                .then(data => {
                    if (data && data.workflow_runs && data.workflow_runs.length > 0) {
                        const latestRun = data.workflow_runs[0];
                        const status = latestRun.conclusion || latestRun.status;
                        
                        if (this.elements.build) {
                            const badge = this.createStatusBadge(status);
                            this.elements.build.innerHTML = '';
                            this.elements.build.appendChild(badge);
                        }
                    }
                })
                .catch(err => console.warn('Could not fetch build status:', err));
        }
        
        createStatusBadge(status) {
            const badge = document.createElement('span');
            badge.className = `status-badge status-${status}`;
            badge.style.cssText = `
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 12px;
                font-size: 0.875rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            `;
            
            switch(status) {
                case 'success':
                case 'completed':
                    badge.textContent = 'Passing';
                    badge.style.background = 'linear-gradient(135deg, #059669 0%, #10b981 100%)';
                    badge.style.color = 'white';
                    break;
                case 'failure':
                    badge.textContent = 'Failing';
                    badge.style.background = 'linear-gradient(135deg, #dc2626 0%, #ef4444 100%)';
                    badge.style.color = 'white';
                    break;
                case 'in_progress':
                case 'queued':
                    badge.textContent = 'Running';
                    badge.style.background = 'linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)';
                    badge.style.color = '#1a2332';
                    break;
                default:
                    badge.textContent = 'Unknown';
                    badge.style.background = '#6b5d4f';
                    badge.style.color = 'white';
            }
            
            return badge;
        }
        
        // ==============================================
        // CACHING
        // ==============================================
        
        getCachedStats() {
            try {
                const cached = localStorage.getItem(CONFIG.cacheKey);
                if (!cached) return null;
                
                const data = JSON.parse(cached);
                const age = Date.now() - data.timestamp;
                
                if (age > CONFIG.cacheExpiry) {
                    localStorage.removeItem(CONFIG.cacheKey);
                    return null;
                }
                
                return data.stats;
            } catch (error) {
                console.warn('Cache read error:', error);
                return null;
            }
        }
        
        cacheStats() {
            try {
                const data = {
                    stats: this.stats,
                    timestamp: Date.now()
                };
                localStorage.setItem(CONFIG.cacheKey, JSON.stringify(data));
            } catch (error) {
                console.warn('Cache write error:', error);
            }
        }
        
        // ==============================================
        // INTERACTIONS
        // ==============================================
        
        setupInteractions() {
            // Add click handlers for stats
            Object.keys(this.elements).forEach(key => {
                const element = this.elements[key];
                if (!element) return;
                
                element.style.cursor = 'pointer';
                element.addEventListener('click', () => {
                    this.handleStatClick(key);
                });
                
                // Add hover effect
                element.addEventListener('mouseenter', () => {
                    element.style.transform = 'scale(1.1)';
                    element.style.transition = 'transform 0.3s ease';
                });
                
                element.addEventListener('mouseleave', () => {
                    element.style.transform = 'scale(1)';
                });
            });
            
            // Add refresh button if it exists
            const refreshBtn = document.getElementById('refresh-stats');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => {
                    refreshBtn.classList.add('spinning');
                    this.fetchAllStats().then(() => {
                        refreshBtn.classList.remove('spinning');
                    });
                });
            }
        }
        
        handleStatClick(stat) {
            const urls = {
                stars: `https://github.com/${CONFIG.owner}/${CONFIG.repo}/stargazers`,
                forks: `https://github.com/${CONFIG.owner}/${CONFIG.repo}/network/members`,
                issues: `https://github.com/${CONFIG.owner}/${CONFIG.repo}/issues`,
                watchers: `https://github.com/${CONFIG.owner}/${CONFIG.repo}/watchers`,
                contributors: `https://github.com/${CONFIG.owner}/${CONFIG.repo}/graphs/contributors`,
                commits: `https://github.com/${CONFIG.owner}/${CONFIG.repo}/commits`,
                releases: `https://github.com/${CONFIG.owner}/${CONFIG.repo}/releases`,
                coverage: `https://codecov.io/gh/${CONFIG.owner}/${CONFIG.repo}`
            };
            
            if (urls[stat]) {
                window.open(urls[stat], '_blank');
            }
        }
        
        // ==============================================
        // ERROR HANDLING
        // ==============================================
        
        handleError(error) {
            console.error('GitHub Stats Error:', error);
            
            // Show fallback UI
            Object.keys(this.elements).forEach(key => {
                const element = this.elements[key];
                if (!element) return;
                
                element.classList.remove('loading');
                element.classList.add('error');
                
                // Show appropriate fallback
                if (key === 'coverage') {
                    element.innerHTML = '<span style="color: #6b5d4f;">N/A</span>';
                } else {
                    element.textContent = '—';
                }
            });
            
            // Show error message if container exists
            const errorContainer = document.getElementById('stats-error');
            if (errorContainer) {
                errorContainer.style.display = 'block';
                errorContainer.textContent = 'Unable to load GitHub stats. Please try again later.';
            }
        }
    }
    
    // ==============================================
    // INITIALIZATION
    // ==============================================
    
    document.addEventListener('DOMContentLoaded', () => {
        // Initialize stats manager
        const statsManager = new GitHubStatsManager();
        
        // Expose to global scope for debugging
        window.tenetsStats = statsManager;
    });
    
    // ==============================================
    // CSS INJECTION
    // ==============================================
    
    const styles = `
        .github-stats-container {
            transition: all 0.5s ease;
            opacity: 0;
            transform: translateY(10px);
        }
        
        .stat-value.loading {
            opacity: 0.5;
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        .stat-value.pulse {
            animation: statPulse 0.6s ease;
        }
        
        .stat-value.error {
            color: #dc2626 !important;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
        
        @keyframes statPulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
        
        @keyframes spinning {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .spinning {
            animation: spinning 1s linear infinite;
        }
    `;
    
    const styleSheet = document.createElement('style');
    styleSheet.textContent = styles;
    document.head.appendChild(styleSheet);
    
})();