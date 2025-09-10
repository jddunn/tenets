/**
 * Lazy loading for API documentation pages
 * Improves performance by loading content only when needed
 */

document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on an API page
    if (!window.location.pathname.includes('/api/')) {
        return;
    }

    // Add loading states to code blocks
    const codeBlocks = document.querySelectorAll('.doc-object');
    codeBlocks.forEach(block => {
        // Add lazy loading class
        block.classList.add('lazy-load');
        
        // Create intersection observer for lazy loading
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.remove('lazy-load');
                    entry.target.classList.add('loaded');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            rootMargin: '50px'
        });
        
        observer.observe(block);
    });

    // Collapse large docstrings by default
    const docstrings = document.querySelectorAll('.doc-contents');
    docstrings.forEach(docstring => {
        if (docstring.textContent.length > 500) {
            docstring.classList.add('collapsed');
            
            // Add expand button
            const expandBtn = document.createElement('button');
            expandBtn.className = 'expand-docstring';
            expandBtn.textContent = 'Show more...';
            expandBtn.onclick = function() {
                docstring.classList.toggle('collapsed');
                this.textContent = docstring.classList.contains('collapsed') 
                    ? 'Show more...' 
                    : 'Show less';
            };
            docstring.parentNode.insertBefore(expandBtn, docstring.nextSibling);
        }
    });

    // Add search filtering for API members
    const apiContent = document.querySelector('.md-content__inner');
    if (apiContent) {
        // Add filter input
        const filterInput = document.createElement('input');
        filterInput.type = 'text';
        filterInput.placeholder = 'Filter API members...';
        filterInput.className = 'api-filter';
        filterInput.style.cssText = `
            width: 100%;
            padding: 8px 12px;
            margin: 16px 0;
            border: 1px solid var(--md-default-fg-color--lightest);
            border-radius: 4px;
            font-size: 14px;
        `;
        
        // Insert after h1
        const h1 = apiContent.querySelector('h1');
        if (h1) {
            h1.parentNode.insertBefore(filterInput, h1.nextSibling);
        }
        
        // Add filtering logic
        filterInput.addEventListener('input', function(e) {
            const filter = e.target.value.toLowerCase();
            const sections = apiContent.querySelectorAll('.doc-object');
            
            sections.forEach(section => {
                const name = section.querySelector('.doc-object-name');
                if (name) {
                    const text = name.textContent.toLowerCase();
                    section.style.display = text.includes(filter) ? '' : 'none';
                }
            });
        });
    }
});