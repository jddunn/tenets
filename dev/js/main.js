// docs/js/main.js
// Core initialization and site-wide functionality for Tenets Documentation
// ==========================================================================

(function() {
    'use strict';

    // ==========================================================================
    // Global Configuration
    // ==========================================================================

    const TENETS_CONFIG = {
        // Animation settings
        animation: {
            duration: 300,
            easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
            delayIncrement: 50,
            observerThreshold: 0.1,
            observerRootMargin: '0px 0px -50px 0px'
        },

        // Scroll settings
        scroll: {
            smoothDuration: 800,
            headerOffset: 126, // 72px header + 54px tabs
            mobileOffset: 64,
            hideHeaderThreshold: 100,
            showBackToTopThreshold: 300
        },

        // Breakpoints
        breakpoints: {
            mobile: 480,
            tablet: 768,
            desktop: 1024,
            wide: 1440
        },

        // Performance
        performance: {
            debounceDelay: 150,
            throttleDelay: 100,
            lazyLoadOffset: 200
        },

        // Storage keys
        storage: {
            theme: 'tenets-theme',
            searchHistory: 'tenets-search-history',
            sessionId: 'tenets-session-id',
            preferences: 'tenets-preferences'
        }
    };

    // ==========================================================================
    // Utility Functions
    // ==========================================================================

    const TenetsUtils = {
        // Debounce function for performance
        debounce(func, wait = TENETS_CONFIG.performance.debounceDelay) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func.apply(this, args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        },

        // Throttle function for scroll/resize events
        throttle(func, limit = TENETS_CONFIG.performance.throttleDelay) {
            let inThrottle;
            let lastFunc;
            let lastTime;

            return function(...args) {
                if (!inThrottle) {
                    func.apply(this, args);
                    lastTime = Date.now();
                    inThrottle = true;
                } else {
                    clearTimeout(lastFunc);
                    lastFunc = setTimeout(() => {
                        if ((Date.now() - lastTime) >= limit) {
                            func.apply(this, args);
                            lastTime = Date.now();
                        }
                    }, Math.max(limit - (Date.now() - lastTime), 0));
                }
            };
        },

        // Check if element is in viewport
        isInViewport(element, offset = 0) {
            const rect = element.getBoundingClientRect();
            const windowHeight = window.innerHeight || document.documentElement.clientHeight;
            const windowWidth = window.innerWidth || document.documentElement.clientWidth;

            return (
                rect.top <= (windowHeight - offset) &&
                rect.bottom >= offset &&
                rect.left <= (windowWidth - offset) &&
                rect.right >= offset
            );
        },

        // Smooth scroll to element
        scrollToElement(element, options = {}) {
            const defaults = {
                offset: window.innerWidth <= TENETS_CONFIG.breakpoints.tablet
                    ? TENETS_CONFIG.scroll.mobileOffset
                    : TENETS_CONFIG.scroll.headerOffset,
                behavior: 'smooth',
                block: 'start'
            };

            const settings = { ...defaults, ...options };
            const targetPosition = element.getBoundingClientRect().top + window.pageYOffset - settings.offset;

            window.scrollTo({
                top: targetPosition,
                behavior: settings.behavior
            });
        },

        // Get current breakpoint
        getCurrentBreakpoint() {
            const width = window.innerWidth;

            if (width <= TENETS_CONFIG.breakpoints.mobile) return 'mobile';
            if (width <= TENETS_CONFIG.breakpoints.tablet) return 'tablet';
            if (width <= TENETS_CONFIG.breakpoints.desktop) return 'desktop';
            return 'wide';
        },

        // Parse JSON safely
        parseJSON(str) {
            try {
                return JSON.parse(str);
            } catch (e) {
                console.error('JSON parse error:', e);
                return null;
            }
        },

        // Format number with commas
        formatNumber(num) {
            return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        },

        // Get CSS custom property value
        getCSSVariable(name) {
            return getComputedStyle(document.documentElement)
                .getPropertyValue(name)
                .trim();
        },

        // Set CSS custom property
        setCSSVariable(name, value) {
            document.documentElement.style.setProperty(name, value);
        },

        // Generate unique ID
        generateId(prefix = 'tenets') {
            return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        },

        // Local storage wrapper with error handling
        storage: {
            get(key) {
                try {
                    const item = localStorage.getItem(key);
                    return item ? JSON.parse(item) : null;
                } catch (e) {
                    console.error('Storage get error:', e);
                    return null;
                }
            },

            set(key, value) {
                try {
                    localStorage.setItem(key, JSON.stringify(value));
                    return true;
                } catch (e) {
                    console.error('Storage set error:', e);
                    return false;
                }
            },

            remove(key) {
                try {
                    localStorage.removeItem(key);
                    return true;
                } catch (e) {
                    console.error('Storage remove error:', e);
                    return false;
                }
            },

            clear() {
                try {
                    localStorage.clear();
                    return true;
                } catch (e) {
                    console.error('Storage clear error:', e);
                    return false;
                }
            }
        },

        // Add event listener with automatic cleanup
        addEvent(element, event, handler, options = {}) {
            element.addEventListener(event, handler, options);

            // Return cleanup function
            return () => {
                element.removeEventListener(event, handler, options);
            };
        },

        // Wait/sleep function
        wait(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        },

        // Check if mobile device
        isMobile() {
            return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        },

        // Check if touch device
        isTouchDevice() {
            return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        }
    };

    // ==========================================================================
    // Module Loader
    // ==========================================================================

    class ModuleLoader {
        constructor() {
            this.modules = new Map();
            this.loadOrder = [];
            this.initialized = false;
        }

        register(name, module, dependencies = []) {
            this.modules.set(name, {
                module,
                dependencies,
                loaded: false
            });
        }

        async loadModule(name) {
            const moduleInfo = this.modules.get(name);
            if (!moduleInfo) {
                console.error(`Module ${name} not found`);
                return null;
            }

            if (moduleInfo.loaded) {
                return moduleInfo.module;
            }

            // Load dependencies first
            for (const dep of moduleInfo.dependencies) {
                await this.loadModule(dep);
            }

            // Initialize module
            if (typeof moduleInfo.module.init === 'function') {
                await moduleInfo.module.init();
            }

            moduleInfo.loaded = true;
            this.loadOrder.push(name);

            console.log(`✅ Module loaded: ${name}`);
            return moduleInfo.module;
        }

        async loadAll() {
            for (const [name] of this.modules) {
                await this.loadModule(name);
            }
            this.initialized = true;
            console.log('🔥 All modules loaded successfully');
        }

        get(name) {
            const moduleInfo = this.modules.get(name);
            return moduleInfo ? moduleInfo.module : null;
        }

        isLoaded(name) {
            const moduleInfo = this.modules.get(name);
            return moduleInfo ? moduleInfo.loaded : false;
        }
    }

    // ==========================================================================
    // Smooth Scroll Handler
    // ==========================================================================

    class SmoothScrollHandler {
        constructor() {
            this.isScrolling = false;
            this.lastScrollY = 0;
            this.ticking = false;
        }

        init() {
            // Avoid heavy anchor binding on large API reference pages
            const isApi = document.body && document.body.classList.contains('is-api');
            if (!isApi) {
                this.bindAnchorLinks();
                this.setupScrollEffects();
            }
            // Back-to-top handled by dedicated module (js/back-to-top.js)
            console.log('✅ Smooth scroll initialized');
        }

        bindAnchorLinks() {
            // Handle all anchor links
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', (e) => {
                    const href = anchor.getAttribute('href');

                    // Skip if just "#" or no href
                    if (!href || href === '#') return;

                    const target = document.querySelector(href);
                    if (!target) return;

                    e.preventDefault();

                    // Smooth scroll to target
                    TenetsUtils.scrollToElement(target);

                    // Update URL without triggering scroll
                    if (history.pushState) {
                        history.pushState(null, null, href);
                    }

                    // Trigger focus for accessibility
                    target.setAttribute('tabindex', '-1');
                    target.focus();
                });
            });
        }

        setupScrollEffects() {
            const header = document.querySelector('.md-header');
            if (!header) return;

            let lastScrollY = window.pageYOffset;
            let ticking = false;

            const updateScrollEffects = () => {
                const currentScrollY = window.pageYOffset;

                // Add shadow on scroll
                if (currentScrollY > 10) {
                    header.classList.add('md-header--shadow');
                } else {
                    header.classList.remove('md-header--shadow');
                }

                // Hide/show header on scroll
                if (currentScrollY > lastScrollY && currentScrollY > TENETS_CONFIG.scroll.hideHeaderThreshold) {
                    header.classList.add('md-header--hidden');
                } else {
                    header.classList.remove('md-header--hidden');
                }

                lastScrollY = currentScrollY;
                ticking = false;
            };

            const requestTick = () => {
                if (!ticking) {
                    window.requestAnimationFrame(updateScrollEffects);
                    ticking = true;
                }
            };

            // Use throttled scroll event
            window.addEventListener('scroll', TenetsUtils.throttle(requestTick, 100));
        }

    // setupBackToTop removed (superseded by js/back-to-top.js)
    }

    // ==========================================================================
    // Animation Observer
    // ==========================================================================

    class AnimationObserver {
        constructor() {
            this.observers = new Map();
            this.animatedElements = new Set();
        }

        init() {
            this.setupIntersectionObserver();
            this.setupMutationObserver();
            this.observeElements();
            console.log('✅ Animation observer initialized');
        }

        setupIntersectionObserver() {
            const options = {
                threshold: TENETS_CONFIG.animation.observerThreshold,
                rootMargin: TENETS_CONFIG.animation.observerRootMargin
            };

            this.intersectionObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting && !this.animatedElements.has(entry.target)) {
                        this.animateElement(entry.target);

                        // Unobserve if not persistent
                        if (!entry.target.hasAttribute('data-persist-observe')) {
                            this.intersectionObserver.unobserve(entry.target);
                            this.animatedElements.add(entry.target);
                        }
                    }
                });
            }, options);
        }

        setupMutationObserver() {
            // Watch for dynamically added content
            this.mutationObserver = new MutationObserver((mutations) => {
                mutations.forEach(mutation => {
                    mutation.addedNodes.forEach(node => {
                        if (node.nodeType === 1) { // Element node
                            this.checkAndObserveElement(node);

                            // Check children
                            node.querySelectorAll('[data-animate]').forEach(child => {
                                this.checkAndObserveElement(child);
                            });
                        }
                    });
                });
            });

            // Start observing document body
            this.mutationObserver.observe(document.body, {
                childList: true,
                subtree: true
            });
        }

        observeElements() {
            // Find all elements with animation classes or data attributes
            const selectors = [
                '[data-animate]',
                '.fade-in',
                '.slide-up',
                '.slide-in',
                '.scale-in',
                '.rotate-in',
                '.counter'
            ];

            document.querySelectorAll(selectors.join(', ')).forEach(element => {
                this.checkAndObserveElement(element);
            });
        }

        checkAndObserveElement(element) {
            if (!this.animatedElements.has(element)) {
                this.intersectionObserver.observe(element);
            }
        }

        animateElement(element) {
            // Add animation class
            element.classList.add('in-view');

            // Handle counter animations
            if (element.classList.contains('counter') || element.hasAttribute('data-counter')) {
                this.animateCounter(element);
            }

            // Handle typewriter animations
            if (element.classList.contains('typewriter') || element.hasAttribute('data-typewriter')) {
                this.animateTypewriter(element);
            }

            // Handle stagger animations
            if (element.hasAttribute('data-stagger')) {
                this.animateStagger(element);
            }

            // Trigger custom event
            element.dispatchEvent(new CustomEvent('tenets:animated', {
                bubbles: true,
                detail: { element }
            }));
        }

        animateCounter(element) {
            const target = parseInt(element.dataset.target || element.textContent);
            const duration = parseInt(element.dataset.duration || 2000);
            const prefix = element.dataset.prefix || '';
            const suffix = element.dataset.suffix || '';

            let start = 0;
            const increment = target / (duration / 16);

            const timer = setInterval(() => {
                start += increment;
                if (start >= target) {
                    start = target;
                    clearInterval(timer);
                }

                element.textContent = prefix + TenetsUtils.formatNumber(Math.floor(start)) + suffix;
            }, 16);
        }

        animateTypewriter(element) {
            const text = element.dataset.text || element.textContent;
            const speed = parseInt(element.dataset.speed || 50);

            element.textContent = '';
            let index = 0;

            const timer = setInterval(() => {
                if (index < text.length) {
                    element.textContent += text.charAt(index);
                    index++;
                } else {
                    clearInterval(timer);
                    element.classList.add('typewriter-complete');
                }
            }, speed);
        }

        animateStagger(element) {
            const children = element.children;
            const delay = parseInt(element.dataset.staggerDelay || 100);

            Array.from(children).forEach((child, index) => {
                setTimeout(() => {
                    child.classList.add('stagger-in');
                }, index * delay);
            });
        }

        destroy() {
            if (this.intersectionObserver) {
                this.intersectionObserver.disconnect();
            }
            if (this.mutationObserver) {
                this.mutationObserver.disconnect();
            }
            this.observers.clear();
            this.animatedElements.clear();
        }
    }

    // ==========================================================================
    // Performance Monitor
    // ==========================================================================

    class PerformanceMonitor {
        constructor() {
            this.metrics = {};
            this.enabled = window.location.hostname === 'localhost' ||
                          window.location.hostname === '127.0.0.1';
        }

        init() {
            if (!this.enabled) return;

            this.measurePageLoad();
            this.measureResources();
            this.logMetrics();

            console.log('✅ Performance monitor initialized (dev mode)');
        }

        measurePageLoad() {
            if (window.performance && window.performance.timing) {
                const timing = window.performance.timing;
                const loadTime = timing.loadEventEnd - timing.navigationStart;
                const domReady = timing.domContentLoadedEventEnd - timing.navigationStart;
                const firstPaint = performance.getEntriesByType('paint')[0]?.startTime || 0;

                this.metrics.pageLoad = {
                    total: loadTime,
                    domReady: domReady,
                    firstPaint: Math.round(firstPaint)
                };
            }
        }

        measureResources() {
            if (window.performance && window.performance.getEntriesByType) {
                const resources = window.performance.getEntriesByType('resource');

                this.metrics.resources = {
                    total: resources.length,
                    scripts: resources.filter(r => r.initiatorType === 'script').length,
                    styles: resources.filter(r => r.initiatorType === 'css').length,
                    images: resources.filter(r => r.initiatorType === 'img').length
                };
            }
        }

        logMetrics() {
            if (!this.enabled) return;

            console.group('📊 Performance Metrics');
            console.table(this.metrics);
            console.groupEnd();
        }

        mark(name) {
            if (!this.enabled) return;

            if (window.performance && window.performance.mark) {
                window.performance.mark(name);
            }
        }

        measure(name, startMark, endMark) {
            if (!this.enabled) return;

            if (window.performance && window.performance.measure) {
                window.performance.measure(name, startMark, endMark);
                const measure = window.performance.getEntriesByName(name)[0];
                console.log(`⏱️ ${name}: ${Math.round(measure.duration)}ms`);
            }
        }
    }

    // ==========================================================================
    // Main Application
    // ==========================================================================

    class TenetsApp {
        constructor() {
            this.config = TENETS_CONFIG;
            this.utils = TenetsUtils;
            this.moduleLoader = new ModuleLoader();
            this.performanceMonitor = new PerformanceMonitor();
            this.initialized = false;
        }

        async init() {
            try {
                // Mark init start
                this.performanceMonitor.mark('tenets-init-start');

                // Wait for DOM ready
                await this.domReady();

                // Initialize core modules
                this.initializeCoreModules();

                // Load all modules
                await this.moduleLoader.loadAll();

                // Setup global event handlers
                this.setupGlobalHandlers();

                // Mark as initialized
                this.initialized = true;
                document.body.classList.add('tenets-initialized');

                // Mark init end and measure
                this.performanceMonitor.mark('tenets-init-end');
                this.performanceMonitor.measure('tenets-init', 'tenets-init-start', 'tenets-init-end');

                // Log success
                console.log('🔥 Tenets documentation initialized successfully');

                // Dispatch custom event
                document.dispatchEvent(new CustomEvent('tenets:initialized', {
                    detail: { app: this }
                }));

            } catch (error) {
                console.error('Failed to initialize Tenets app:', error);
                this.handleInitError(error);
            }
        }

        domReady() {
            return new Promise(resolve => {
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', resolve);
                } else {
                    resolve();
                }
            });
        }

        initializeCoreModules() {
            // Register core modules
            this.moduleLoader.register('smooth-scroll', new SmoothScrollHandler());
            // Skip heavy intersection observers on API pages
            const isApi = document.body && document.body.classList.contains('is-api');
            if (!isApi) {
                this.moduleLoader.register('animations', new AnimationObserver());
            } else {
                // Ensure any existing observers are not running when arriving via instant nav
                try { this.destroyAnimations && this.destroyAnimations(); } catch(_) {}
            }
            this.moduleLoader.register('performance', this.performanceMonitor);

            // These will be loaded from external files
            // Just register them as placeholders that will be replaced
            this.moduleLoader.register('theme-toggle', { init: () => {} });
            this.moduleLoader.register('terminal', { init: () => {} });
            this.moduleLoader.register('mobile-menu', { init: () => {} });
            this.moduleLoader.register('copy-buttons', { init: () => {} });
            this.moduleLoader.register('search', { init: () => {} });
        }

    setupGlobalHandlers() {
            // Ensure search input always has placeholder
            const ensureSearchPlaceholder = () => {
                const searchInputs = document.querySelectorAll('.md-search__input');
                searchInputs.forEach(input => {
                    input.setAttribute('placeholder', 'Search...');
                });
            };

            // Set placeholder initially
            ensureSearchPlaceholder();

            // Re-set after any navigation or dynamic content loading
            document.addEventListener('DOMContentLoaded', ensureSearchPlaceholder);
            window.addEventListener('load', ensureSearchPlaceholder);

            // Create and inject close button for search
            const createSearchCloseButton = () => {
                const searchForm = document.querySelector('.md-search__form');
                if (!searchForm || document.querySelector('.search-close-btn')) return;

                const closeBtn = document.createElement('button');
                closeBtn.className = 'search-close-btn';
                closeBtn.innerHTML = '✕';
                closeBtn.style.cssText = `
                    position: absolute;
                    right: 10px;
                    top: 50%;
                    transform: translateY(-50%);
                    background: #f59e0b;
                    color: white;
                    border: none;
                    border-radius: 50%;
                    width: 24px;
                    height: 24px;
                    font-size: 16px;
                    cursor: pointer;
                    display: none !important;
                    z-index: 1000;
                    line-height: 1;
                    padding: 0;
                    transition: all 0.2s ease;
                `;

                closeBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    closeSearch();
                });

                closeBtn.addEventListener('mouseenter', () => {
                    closeBtn.style.background = '#dc2626';
                    closeBtn.style.transform = 'translateY(-50%) scale(1.1)';
                });

                closeBtn.addEventListener('mouseleave', () => {
                    closeBtn.style.background = '#f59e0b';
                    closeBtn.style.transform = 'translateY(-50%)';
                });

                searchForm.style.position = 'relative';
                searchForm.appendChild(closeBtn);
            };

            // Show/hide close button based on search state
            const updateSearchCloseButton = () => {
                const searchContainer = document.querySelector('.md-search');
                const searchInput = document.querySelector('.md-search__input');
                const searchOutput = document.querySelector('.md-search__output');
                const searchResults = document.querySelector('.md-search-result__list');
                const closeBtn = document.querySelector('.search-close-btn');

                if (!closeBtn || !searchContainer) return;

                // Check if we have actual search results visible
                const hasResults = searchResults &&
                                 searchResults.children.length > 0 &&
                                 !searchResults.hasAttribute('hidden');

                // Check if search output is visible with content
                const hasVisibleOutput = searchOutput &&
                                       searchOutput.offsetParent !== null &&
                                       hasResults;

                // Check if input has text
                const hasText = searchInput && searchInput.value.trim() !== '';

                // Only show when we have actual results or text being searched
                const shouldShow = hasVisibleOutput && hasText;

                // Use setProperty to override with !important
                if (shouldShow) {
                    closeBtn.style.setProperty('display', 'block', 'important');
                } else {
                    closeBtn.style.setProperty('display', 'none', 'important');
                }
            };

            // Force close search
            const closeSearch = () => {
                const searchContainer = document.querySelector('.md-search');
                const searchInput = document.querySelector('.md-search__input');
                const searchOutput = document.querySelector('.md-search__output');

                if (!searchContainer || !searchInput) return;

                // Clear and blur input
                searchInput.blur();
                searchInput.value = '';

                // Hide output
                if (searchOutput) {
                    searchOutput.style.opacity = '0';
                    setTimeout(() => {
                        searchOutput.style.opacity = '';
                    }, 300);
                }

                // Remove ALL active states
                searchContainer.removeAttribute('aria-expanded');
                searchContainer.removeAttribute('data-md-state');
                searchContainer.setAttribute('aria-expanded', 'false');
                searchContainer.classList.remove('md-search--active');

                // Clear body states
                document.body.removeAttribute('data-md-state');
                document.body.classList.remove('md-search--active');

                // Trigger reset
                const resetBtn = searchContainer.querySelector('[type="reset"]');
                if (resetBtn) {
                    resetBtn.click();
                }

                // Hide close button
                updateSearchCloseButton();
            };

            // Initialize close button
            setTimeout(createSearchCloseButton, 100);

            // Monitor search state changes
            const searchObserver = new MutationObserver(updateSearchCloseButton);
            const searchContainer = document.querySelector('.md-search');
            if (searchContainer) {
                searchObserver.observe(searchContainer, {
                    attributes: true,
                    attributeFilter: ['aria-expanded', 'data-md-state']
                });
            }

            // Monitor search input and output changes
            const searchInput = document.querySelector('.md-search__input');
            if (searchInput) {
                searchInput.addEventListener('input', updateSearchCloseButton);
                searchInput.addEventListener('focus', () => setTimeout(updateSearchCloseButton, 100));
                searchInput.addEventListener('blur', () => setTimeout(updateSearchCloseButton, 100));
            }

            // Monitor search output for changes
            const searchOutput = document.querySelector('.md-search__output');
            if (searchOutput) {
                const outputObserver = new MutationObserver(updateSearchCloseButton);
                outputObserver.observe(searchOutput, {
                    childList: true,
                    subtree: true
                });
            }

            // Handle clicks anywhere on page
            document.addEventListener('mousedown', (e) => {
                const searchContainer = document.querySelector('.md-search');
                const searchOutput = document.querySelector('.md-search__output');

                if (!searchContainer) return;

                // Check if search is actually active
                const isActive = (searchContainer.hasAttribute('aria-expanded') &&
                                 searchContainer.getAttribute('aria-expanded') !== 'false') ||
                                searchContainer.hasAttribute('data-md-state');

                const hasVisibleOutput = searchOutput &&
                                        searchOutput.offsetParent !== null &&
                                        searchOutput.querySelector('.md-search-result');

                // If active and click is outside
                if ((isActive || hasVisibleOutput) &&
                    !searchContainer.contains(e.target) &&
                    !e.target.closest('.md-search')) {

                    e.preventDefault();
                    e.stopPropagation();
                    closeSearch();
                }
            }, true);

            // Also handle overlay if it exists
            const searchOverlay = document.querySelector('.md-search__overlay');
            if (searchOverlay) {
                searchOverlay.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    closeSearch();
                });
            }

            // Prevent search from staying active on ESC key
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    const searchInput = document.querySelector('.md-search__input');
                    if (searchInput && document.activeElement === searchInput) {
                        searchInput.blur();

                        const searchContainer = document.querySelector('.md-search');
                        if (searchContainer) {
                            searchContainer.removeAttribute('aria-expanded');
                            searchContainer.removeAttribute('data-md-state');
                        }
                    }
                }
            });

            // Handle resize events
            let resizeTimer;
            window.addEventListener('resize', () => {
                document.body.classList.add('is-resizing');

                clearTimeout(resizeTimer);
                resizeTimer = setTimeout(() => {
                    document.body.classList.remove('is-resizing');

                    // Dispatch custom resize end event
                    document.dispatchEvent(new CustomEvent('tenets:resizeEnd', {
                        detail: {
                            breakpoint: this.utils.getCurrentBreakpoint(),
                            width: window.innerWidth,
                            height: window.innerHeight
                        }
                    }));
                }, 250);
            });

            // Handle visibility change
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    document.body.classList.add('is-hidden');
                } else {
                    document.body.classList.remove('is-hidden');
                }
            });

            // Handle online/offline
            window.addEventListener('online', () => {
                document.body.classList.remove('is-offline');
                document.body.classList.add('is-online');
            });

            window.addEventListener('offline', () => {
                document.body.classList.remove('is-online');
                document.body.classList.add('is-offline');
            });

            // Handle print
            window.addEventListener('beforeprint', () => {
                document.body.classList.add('is-printing');
            });

            window.addEventListener('afterprint', () => {
                document.body.classList.remove('is-printing');
            });

            // Listen for page flag changes (e.g., API docs via navigation.instant)
            document.addEventListener('tenets:page-flags', (e) => {
                const isApi = !!(e && e.detail && e.detail.isApi);
                if (isApi) {
                    this.destroyAnimations();
                }
            });
        }

        destroyAnimations() {
            try {
                const mod = this.moduleLoader && this.moduleLoader.get('animations');
                if (mod && typeof mod.destroy === 'function') {
                    mod.destroy();
                }
            } catch (_) {}
        }

        handleInitError(error) {
            // Create error message
            const errorBanner = document.createElement('div');
            errorBanner.className = 'tenets-error-banner';
            errorBanner.innerHTML = `
                <div class="error-content">
                    <strong>Initialization Error:</strong>
                    <span>${error.message}</span>
                    <button onclick="location.reload()">Reload Page</button>
                </div>
            `;

            document.body.insertBefore(errorBanner, document.body.firstChild);
        }

        // Public API
        getModule(name) {
            return this.moduleLoader.get(name);
        }

        registerModule(name, module, dependencies) {
            return this.moduleLoader.register(name, module, dependencies);
        }

        isInitialized() {
            return this.initialized;
        }
    }

    // ==========================================================================
    // Initialize Application
    // ==========================================================================

    // Create global instance
    window.Tenets = new TenetsApp();

    // Initialize
    window.Tenets.init();

    // Export utilities for external use
    window.TenetsUtils = TenetsUtils;

})();
