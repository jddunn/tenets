// Enhanced search functionality
document.addEventListener('DOMContentLoaded', function() {
    // Add keyboard shortcut for search (/)
    document.addEventListener('keydown', function(e) {
        // Check if not in input/textarea
        if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
            e.preventDefault();
            const searchInput = document.querySelector('.md-search__input');
            if (searchInput) {
                searchInput.focus();
                searchInput.select();
            }
        }
        
        // ESC to close search
        if (e.key === 'Escape') {
            const searchReset = document.querySelector('[for="__search"]');
            if (searchReset && document.querySelector('.md-search__input:focus')) {
                searchReset.click();
            }
        }
    });
    
    // Add search hints
    const searchInput = document.querySelector('.md-search__input');
    if (searchInput) {
        // Dynamic placeholder
        const placeholders = [
            'Search for "make-context"...',
            'Try "session management"...',
            'Search "API reference"...',
            'Find "installation"...',
            'Look for "configuration"...'
        ];
        
        let placeholderIndex = 0;
        
        // Only show hints when not focused
        searchInput.addEventListener('blur', function() {
            if (!this.value) {
                placeholderIndex = (placeholderIndex + 1) % placeholders.length;
                this.placeholder = placeholders[placeholderIndex];
            }
        });
        
        searchInput.addEventListener('focus', function() {
            this.placeholder = 'Search documentation...';
        });
        
        // Search statistics
        let searchCount = 0;
        searchInput.addEventListener('input', function() {
            if (this.value.length >= 2) {
                searchCount++;
                // You could send this to analytics
                console.log(`Search #${searchCount}: ${this.value}`);
            }
        });
    }
    
    // Enhance search results with icons
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                const results = document.querySelectorAll('.md-search-result__link');
                results.forEach(result => {
                    if (!result.dataset.enhanced) {
                        const title = result.querySelector('.md-search-result__title');
                        if (title) {
                            const text = title.textContent.toLowerCase();
                            let icon = '📄'; // Default
                            
                            // Add icons based on content type
                            if (text.includes('install')) icon = '📦';
                            else if (text.includes('api')) icon = '🔌';
                            else if (text.includes('cli')) icon = '⌨️';
                            else if (text.includes('guide')) icon = '📖';
                            else if (text.includes('config')) icon = '⚙️';
                            else if (text.includes('session')) icon = '🔄';
                            else if (text.includes('quick')) icon = '🚀';
                            else if (text.includes('example')) icon = '💡';
                            
                            title.innerHTML = `<span style="margin-right: 0.5rem">${icon}</span>${title.innerHTML}`;
                        }
                        result.dataset.enhanced = 'true';
                    }
                });
            }
        });
    });
    
    // Observe search results container
    const searchOutput = document.querySelector('.md-search__output');
    if (searchOutput) {
        observer.observe(searchOutput, { childList: true, subtree: true });
    }
    
    // Add search history (localStorage)
    const SEARCH_HISTORY_KEY = 'tenets-search-history';
    const MAX_HISTORY = 10;
    
    function getSearchHistory() {
        const history = localStorage.getItem(SEARCH_HISTORY_KEY);
        return history ? JSON.parse(history) : [];
    }
    
    function addToSearchHistory(query) {
        if (query.length < 3) return;
        
        let history = getSearchHistory();
        // Remove if already exists
        history = history.filter(h => h !== query);
        // Add to beginning
        history.unshift(query);
        // Limit size
        history = history.slice(0, MAX_HISTORY);
        
        localStorage.setItem(SEARCH_HISTORY_KEY, JSON.stringify(history));
    }
    
    // Save searches
    if (searchInput) {
        searchInput.addEventListener('change', function() {
            if (this.value) {
                addToSearchHistory(this.value);
            }
        });
    }
    
    // Add Command Palette style search (Ctrl/Cmd + K)
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchToggle = document.querySelector('[for="__search"]');
            if (searchToggle) {
                searchToggle.click();
                setTimeout(() => {
                    const searchInput = document.querySelector('.md-search__input');
                    if (searchInput) {
                        searchInput.focus();
                    }
                }, 100);
            }
        }
    });
});

// Add this CSS dynamically for search hints
const style = document.createElement('style');
style.textContent = `
    /* Search hints tooltip */
    .search-hint {
        position: absolute;
        right: 3rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 0.75rem;
        opacity: 0.5;
        pointer-events: none;
        font-family: 'JetBrains Mono', monospace;
        background: var(--md-default-bg-color--lighter);
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
    }
    
    .md-search__input:focus ~ .search-hint {
        display: none;
    }
`;
document.head.appendChild(style);