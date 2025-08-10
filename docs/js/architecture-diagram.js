// Interactive Architecture Diagram
// Path: docs/js/architecture-diagram.js
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    'use strict';
    
    // Architecture components
    const architectureData = {
        components: [
            {
                id: 'input',
                name: 'Input',
                description: 'Query or prompt from user',
                details: 'Accepts natural language queries, file paths, and configuration options',
                color: '#f59e0b',
                x: 10,
                y: 50
            },
            {
                id: 'scanner',
                name: 'Scanner',
                description: 'File discovery and filtering',
                details: 'Respects .gitignore, applies include/exclude patterns, discovers all relevant files',
                color: '#fbbf24',
                x: 25,
                y: 50
            },
            {
                id: 'analyzer',
                name: 'Analyzer',
                description: 'Code structure analysis',
                details: 'Extracts imports, dependencies, functions, classes, and complexity metrics',
                color: '#fcd34d',
                x: 40,
                y: 50
            },
            {
                id: 'ranker',
                name: 'Ranker',
                description: 'Multi-factor relevance scoring',
                details: 'TF-IDF, git activity, import graph, path relevance, semantic similarity',
                color: '#fbbf24',
                x: 55,
                y: 50
            },
            {
                id: 'mlpipeline',
                name: 'ML Pipeline',
                description: 'Optional deep learning models',
                details: 'Semantic embeddings, code understanding, advanced ranking (optional)',
                color: '#60a5fa',
                x: 55,
                y: 30,
                optional: true
            },
            {
                id: 'aggregator',
                name: 'Aggregator',
                description: 'Context optimization',
                details: 'Token budgeting, smart summarization, format optimization',
                color: '#f59e0b',
                x: 70,
                y: 50
            },
            {
                id: 'output',
                name: 'Output',
                description: 'Formatted context',
                details: 'Markdown, JSON, or XML format ready for LLMs',
                color: '#10b981',
                x: 85,
                y: 50
            }
        ],
        connections: [
            { from: 'input', to: 'scanner' },
            { from: 'scanner', to: 'analyzer' },
            { from: 'analyzer', to: 'ranker' },
            { from: 'ranker', to: 'aggregator' },
            { from: 'mlpipeline', to: 'ranker', optional: true },
            { from: 'aggregator', to: 'output' }
        ]
    };
    
    // Create interactive SVG diagram
    function createArchitectureDiagram() {
        const container = document.querySelector('.architecture-wrapper');
        if (!container) return;
        
        // Clear existing content
        container.innerHTML = '';
        
        // Create SVG
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 100 80');
        svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        svg.classList.add('architecture-svg');
        svg.style.width = '100%';
        svg.style.height = 'auto';
        svg.style.maxHeight = '500px';
        
        // Add definitions for arrows and gradients
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        
        // Arrow marker
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id', 'arrow');
        marker.setAttribute('viewBox', '0 0 10 10');
        marker.setAttribute('refX', '8');
        marker.setAttribute('refY', '5');
        marker.setAttribute('markerWidth', '6');
        marker.setAttribute('markerHeight', '6');
        marker.setAttribute('orient', 'auto');
        
        const arrowPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        arrowPath.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
        arrowPath.setAttribute('fill', '#f59e0b');
        marker.appendChild(arrowPath);
        defs.appendChild(marker);
        
        // Add gradients for each component
        architectureData.components.forEach(comp => {
            const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            gradient.setAttribute('id', `gradient-${comp.id}`);
            gradient.setAttribute('x1', '0%');
            gradient.setAttribute('y1', '0%');
            gradient.setAttribute('x2', '100%');
            gradient.setAttribute('y2', '100%');
            
            const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop1.setAttribute('offset', '0%');
            stop1.setAttribute('stop-color', comp.color);
            stop1.setAttribute('stop-opacity', '0.8');
            
            const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop2.setAttribute('offset', '100%');
            stop2.setAttribute('stop-color', comp.color);
            stop2.setAttribute('stop-opacity', '1');
            
            gradient.appendChild(stop1);
            gradient.appendChild(stop2);
            defs.appendChild(gradient);
        });
        
        svg.appendChild(defs);
        
        // Draw connections first (so they appear behind components)
        const connectionsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        connectionsGroup.classList.add('connections');
        
        architectureData.connections.forEach(conn => {
            const fromComp = architectureData.components.find(c => c.id === conn.from);
            const toComp = architectureData.components.find(c => c.id === conn.to);
            
            if (fromComp && toComp) {
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', fromComp.x + 7);
                line.setAttribute('y1', fromComp.y);
                line.setAttribute('x2', toComp.x - 7);
                line.setAttribute('y2', toComp.y);
                line.setAttribute('stroke', conn.optional ? '#60a5fa' : '#f59e0b');
                line.setAttribute('stroke-width', '0.5');
                line.setAttribute('stroke-dasharray', conn.optional ? '2,1' : 'none');
                line.setAttribute('marker-end', 'url(#arrow)');
                line.setAttribute('opacity', conn.optional ? '0.5' : '0.7');
                line.classList.add('connection-line');
                line.dataset.from = conn.from;
                line.dataset.to = conn.to;
                
                connectionsGroup.appendChild(line);
            }
        });
        
        svg.appendChild(connectionsGroup);
        
        // Draw components
        const componentsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        componentsGroup.classList.add('components');
        
        architectureData.components.forEach(comp => {
            const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.classList.add('component');
            g.dataset.id = comp.id;
            g.style.cursor = 'pointer';
            
            // Component box
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', comp.x - 7);
            rect.setAttribute('y', comp.y - 5);
            rect.setAttribute('width', '14');
            rect.setAttribute('height', '10');
            rect.setAttribute('rx', '2');
            rect.setAttribute('fill', `url(#gradient-${comp.id})`);
            rect.setAttribute('stroke', comp.color);
            rect.setAttribute('stroke-width', '0.3');
            rect.setAttribute('opacity', comp.optional ? '0.7' : '1');
            rect.classList.add('component-box');
            
            // Component text
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', comp.x);
            text.setAttribute('y', comp.y);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('dominant-baseline', 'middle');
            text.setAttribute('font-size', '2.5');
            text.setAttribute('font-weight', '600');
            text.setAttribute('fill', '#1a2332');
            text.textContent = comp.name;
            
            // Optional indicator
            if (comp.optional) {
                const optionalText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                optionalText.setAttribute('x', comp.x);
                optionalText.setAttribute('y', comp.y + 7);
                optionalText.setAttribute('text-anchor', 'middle');
                optionalText.setAttribute('font-size', '1.5');
                optionalText.setAttribute('fill', '#60a5fa');
                optionalText.setAttribute('font-style', 'italic');
                optionalText.textContent = '(optional)';
                g.appendChild(optionalText);
            }
            
            g.appendChild(rect);
            g.appendChild(text);
            
            // Add hover effect
            g.addEventListener('mouseenter', function() {
                rect.setAttribute('transform', 'translate(0, -1)');
                rect.style.filter = 'drop-shadow(0 4px 8px rgba(245, 158, 11, 0.3))';
                showTooltip(comp, rect);
            });
            
            g.addEventListener('mouseleave', function() {
                rect.setAttribute('transform', '');
                rect.style.filter = '';
                hideTooltip();
            });
            
            // Add click interaction
            g.addEventListener('click', function() {
                highlightComponent(comp.id);
                showDetails(comp);
            });
            
            componentsGroup.appendChild(g);
        });
        
        svg.appendChild(componentsGroup);
        container.appendChild(svg);
        
        // Add details panel
        createDetailsPanel(container);
    }
    
    // Create details panel
    function createDetailsPanel(container) {
        const panel = document.createElement('div');
        panel.className = 'architecture-details';
        panel.style.cssText = `
            margin-top: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.05) 0%, rgba(251, 191, 36, 0.02) 100%);
            border: 1px solid rgba(245, 158, 11, 0.2);
            border-radius: 12px;
            min-height: 100px;
            transition: all 0.3s ease;
        `;
        
        const title = document.createElement('h4');
        title.style.cssText = `
            font-size: 1.25rem;
            color: #f59e0b;
            margin: 0 0 0.75rem;
            font-family: 'Playfair Display', serif;
        `;
        title.textContent = 'Click on any component to see details';
        
        const description = document.createElement('p');
        description.style.cssText = `
            color: #6b5d4f;
            line-height: 1.6;
            margin: 0;
        `;
        description.textContent = 'The Tenets architecture processes your query through a pipeline of intelligent components, each optimizing the context for your specific needs.';
        
        panel.appendChild(title);
        panel.appendChild(description);
        container.appendChild(panel);
    }
    
    // Show component details
    function showDetails(component) {
        const panel = document.querySelector('.architecture-details');
        if (!panel) return;
        
        panel.innerHTML = `
            <h4 style="font-size: 1.25rem; color: #f59e0b; margin: 0 0 0.75rem; font-family: 'Playfair Display', serif;">
                ${component.name}
            </h4>
            <p style="color: #1a2332; font-weight: 600; margin: 0 0 0.5rem;">
                ${component.description}
            </p>
            <p style="color: #6b5d4f; line-height: 1.6; margin: 0;">
                ${component.details}
            </p>
            ${component.optional ? '<p style="color: #60a5fa; font-style: italic; margin: 0.5rem 0 0; font-size: 0.875rem;">This component is optional and can be enabled with the ML extra.</p>' : ''}
        `;
        
        // Add animation
        panel.style.animation = 'pulse 0.5s ease';
        setTimeout(() => {
            panel.style.animation = '';
        }, 500);
    }
    
    // Highlight component and its connections
    function highlightComponent(componentId) {
        // Reset all highlights
        document.querySelectorAll('.component-box').forEach(box => {
            box.style.opacity = '0.5';
            box.style.transition = 'all 0.3s ease';
        });
        
        document.querySelectorAll('.connection-line').forEach(line => {
            line.style.opacity = '0.2';
            line.style.transition = 'all 0.3s ease';
        });
        
        // Highlight selected component
        const selectedComponent = document.querySelector(`[data-id="${componentId}"] .component-box`);
        if (selectedComponent) {
            selectedComponent.style.opacity = '1';
            selectedComponent.style.filter = 'drop-shadow(0 0 10px rgba(245, 158, 11, 0.5))';
        }
        
        // Highlight connected lines
        document.querySelectorAll('.connection-line').forEach(line => {
            if (line.dataset.from === componentId || line.dataset.to === componentId) {
                line.style.opacity = '1';
                line.style.strokeWidth = '0.8';
            }
        });
        
        // Reset after 3 seconds
        setTimeout(() => {
            document.querySelectorAll('.component-box').forEach(box => {
                box.style.opacity = '';
                box.style.filter = '';
            });
            document.querySelectorAll('.connection-line').forEach(line => {
                line.style.opacity = '';
                line.style.strokeWidth = '';
            });
        }, 3000);
    }
    
    // Tooltip functionality
    let tooltip = null;
    
    function showTooltip(component, element) {
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.style.cssText = `
                position: absolute;
                background: #1a2332;
                color: #fdfdf9;
                padding: 0.5rem 0.75rem;
                border-radius: 6px;
                font-size: 0.875rem;
                pointer-events: none;
                z-index: 1000;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                transition: all 0.2s ease;
                opacity: 0;
            `;
            document.body.appendChild(tooltip);
        }
        
        tooltip.textContent = component.description;
        
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2 + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
        tooltip.style.opacity = '1';
    }
    
    function hideTooltip() {
        if (tooltip) {
            tooltip.style.opacity = '0';
        }
    }
    
    // Initialize
    createArchitectureDiagram();
    
    // Recreate on resize
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(createArchitectureDiagram, 250);
    });
    
    // Add CSS animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        .architecture-svg .component {
            transition: all 0.3s ease;
        }
        
        .architecture-svg .component:hover {
            transform: scale(1.05);
        }
        
        .architecture-svg .connection-line {
            transition: all 0.3s ease;
        }
    `;
    document.head.appendChild(style);
});