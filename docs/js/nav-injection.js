// Navigation Link Injection for Desktop and Mobile
// Path: docs/js/nav-injection.js
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    'use strict';
    
    // Navigation links configuration
    const navLinks = [
        { text: 'Home', href: '/', icon: '🏠' },
        { text: 'Features', href: '/features/', icon: '✨' },
        { text: 'Quick Start', href: '/quickstart/', icon: '🚀' },
        { text: 'Installation', href: '/installation/', icon: '📦' },
        { text: 'CLI Reference', href: '/CLI/', icon: '⌨️' },
        { text: 'Configuration', href: '/CONFIG/', icon: '⚙️' },
        { text: 'Architecture', href: '/DEEP-DIVE/', icon: '🏗️' },
        { text: 'Blog', href: 'https://manic.agency/blog', external: true, icon: '📝' },
        { text: 'Contact', href: 'https://manic.agency/contact', external: true, icon: '💬' }
    ];
    
    // Inject desktop navigation links
    function injectDesktopNav() {
        const navContainer = document.querySelector('.md-header-nav');
        if (!navContainer) return;
        
        // Clear existing content
        navContainer.innerHTML = '';
        
        // Create navigation links
        navLinks.forEach((link, index) => {
            const navLink = document.createElement('a');
            navLink.className = 'md-header-nav__link';
            navLink.href = link.href;
            navLink.textContent = link.text;
            
            // Add external link indicator
            if (link.external) {
                navLink.setAttribute('target', '_blank');
                navLink.setAttribute('rel', 'noopener noreferrer');
                navLink.classList.add('external-link');
            }
            
            // Mark active link
            const currentPath = window.location.pathname;
            if (currentPath === link.href || 
                (link.href !== '/' && currentPath.startsWith(link.href))) {
                navLink.classList.add('md-header-nav__link--active');
            }
            
            navContainer.appendChild(navLink);
        });
    }
    
    // Inject mobile navigation links
    function injectMobileNav() {
        const mobileNav = document.querySelector('.md-nav--primary .md-nav__list');
        if (!mobileNav) return;
        
        // Clear existing content except title
        mobileNav.innerHTML = '';
        
        // Create navigation items
        navLinks.forEach((link, index) => {
            const navItem = document.createElement('li');
            navItem.className = 'md-nav__item';
            
            const navLink = document.createElement('a');
            navLink.className = 'md-nav__link';
            navLink.href = link.href;
            
            // Create link content with icon
            const linkContent = document.createElement('span');
            linkContent.style.display = 'flex';
            linkContent.style.alignItems = 'center';
            linkContent.style.gap = '0.75rem';
            
            const icon = document.createElement('span');
            icon.textContent = link.icon;
            icon.style.fontSize = '1.25rem';
            icon.style.opacity = '0.8';
            
            const text = document.createElement('span');
            text.textContent = link.text;
            
            linkContent.appendChild(icon);
            linkContent.appendChild(text);
            navLink.appendChild(linkContent);
            
            // Add external link indicator
            if (link.external) {
                navLink.setAttribute('target', '_blank');
                navLink.setAttribute('rel', 'noopener noreferrer');
                navLink.classList.add('external-link');
                
                const externalIcon = document.createElement('span');
                externalIcon.textContent = ' ↗';
                externalIcon.style.opacity = '0.6';
                externalIcon.style.fontSize = '0.875rem';
                linkContent.appendChild(externalIcon);
            }
            
            // Mark active link
            const currentPath = window.location.pathname;
            if (currentPath === link.href || 
                (link.href !== '/' && currentPath.startsWith(link.href))) {
                navLink.classList.add('md-nav__link--active');
            }
            
            // Add click handler to close drawer
            navLink.addEventListener('click', function(e) {
                if (!link.external) {
                    const drawer = document.getElementById('__drawer');
                    if (drawer && drawer.checked) {
                        drawer.checked = false;
                    }
                }
            });
            
            navItem.appendChild(navLink);
            mobileNav.appendChild(navItem);
        });
    }
    
    // Fix hamburger menu toggle
    function fixHamburgerMenu() {
        const hamburgerButton = document.querySelector('.md-header__button[for="__drawer"]');
        const drawer = document.getElementById('__drawer');
        const overlay = document.querySelector('.md-overlay');
        
        if (!hamburgerButton || !drawer) {
            console.warn('Hamburger menu elements not found, creating fallback');
            
            // Create drawer checkbox if it doesn't exist
            if (!drawer) {
                const drawerInput = document.createElement('input');
                drawerInput.type = 'checkbox';
                drawerInput.id = '__drawer';
                drawerInput.className = 'md-toggle md-toggle--drawer';
                drawerInput.style.display = 'none';
                document.body.insertBefore(drawerInput, document.body.firstChild);
            }
        }
        
        // Add click handler to hamburger button
        if (hamburgerButton) {
            hamburgerButton.addEventListener('click', function(e) {
                e.preventDefault();
                const drawer = document.getElementById('__drawer');
                if (drawer) {
                    drawer.checked = !drawer.checked;
                    document.body.classList.toggle('drawer-open', drawer.checked);
                }
            });
        }
        
        // Add click handler to overlay to close drawer
        if (overlay) {
            overlay.addEventListener('click', function() {
                const drawer = document.getElementById('__drawer');
                if (drawer) {
                    drawer.checked = false;
                    document.body.classList.remove('drawer-open');
                }
            });
        }
    }
    
    // Initialize navigation
    function initNavigation() {
        // Check viewport width
        const isMobile = window.innerWidth < 1400;
        
        if (!isMobile) {
            injectDesktopNav();
        }
        
        // Always inject mobile nav (it's hidden on desktop anyway)
        injectMobileNav();
        
        // Fix hamburger menu
        fixHamburgerMenu();
    }
    
    // Run on load
    initNavigation();
    
    // Re-initialize on resize
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            initNavigation();
        }, 250);
    });
    
    // Handle navigation state changes
    document.addEventListener('DOMContentLoaded', function() {
        // Check for MkDocs navigation updates
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList') {
                    // Re-inject navigation if container is replaced
                    const navContainer = document.querySelector('.md-header-nav');
                    const mobileNav = document.querySelector('.md-nav--primary .md-nav__list');
                    
                    if (navContainer && navContainer.children.length === 0) {
                        injectDesktopNav();
                    }
                    
                    if (mobileNav && mobileNav.children.length === 0) {
                        injectMobileNav();
                    }
                }
            });
        });
        
        // Observe header for changes
        const header = document.querySelector('.md-header');
        if (header) {
            observer.observe(header, { childList: true, subtree: true });
        }
    });
});