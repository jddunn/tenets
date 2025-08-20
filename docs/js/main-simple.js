// Simplified main.js - Essential functionality only, no heavy animations
(function() {
    'use strict';
    
    // Simple utility functions
    const TenetsUtils = {
        debounce(func, wait = 150) {
            let timeout;
            return function(...args) {
                clearTimeout(timeout);
                timeout = setTimeout(() => func.apply(this, args), wait);
            };
        },
        
        storage: {
            get(key) {
                try {
                    const item = localStorage.getItem(key);
                    return item ? JSON.parse(item) : null;
                } catch (e) {
                    return null;
                }
            },
            set(key, value) {
                try {
                    localStorage.setItem(key, JSON.stringify(value));
                    return true;
                } catch (e) {
                    return false;
                }
            }
        }
    };
    
    // Simple smooth scroll for anchor links
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                const href = anchor.getAttribute('href');
                if (!href || href === '#') return;
                
                const target = document.querySelector(href);
                if (!target) return;
                
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                
                if (history.pushState) {
                    history.pushState(null, null, href);
                }
            });
        });
    }
    
    // Simple header shadow on scroll
    function initHeaderEffects() {
        const header = document.querySelector('.md-header');
        if (!header) return;
        
        const updateHeader = TenetsUtils.debounce(() => {
            if (window.pageYOffset > 10) {
                header.classList.add('md-header--shadow');
            } else {
                header.classList.remove('md-header--shadow');
            }
        }, 100);
        
        window.addEventListener('scroll', updateHeader, { passive: true });
    }
    
    // Initialize when DOM is ready
    function init() {
        initSmoothScroll();
        initHeaderEffects();
        
        // Mark as initialized
        document.body.classList.add('tenets-initialized');
        
        // Export utils for other scripts
        window.TenetsUtils = TenetsUtils;
        
        console.log('Tenets docs initialized (simplified)');
    }
    
    // Start initialization
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();