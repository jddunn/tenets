// Terminal Examples and Animations
// Path: docs/js/terminal.js
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    'use strict';
    
    // Terminal output examples
    const terminalExamples = {
        distill: {
            command: 'tenets distill "implement OAuth2 authentication" ./src',
            outputs: [
                { text: '🔍 Scanning codebase...', delay: 100, class: 'info' },
                { text: '📁 Found 1,247 files in 89 directories', delay: 200 },
                { text: '', delay: 50 },
                { text: '🎯 Analyzing relevance...', delay: 150, class: 'info' },
                { text: '  • Keywords: oauth2, authentication, auth, token, provider', delay: 100 },
                { text: '  • Git activity: 23 recent changes in auth/', delay: 100 },
                { text: '  • Import graph: 7 core authentication modules', delay: 100 },
                { text: '', delay: 50 },
                { text: '📊 Ranking files by relevance...', delay: 200, class: 'info' },
                { text: '  [████████████████████] 100%', delay: 300, class: 'progress' },
                { text: '', delay: 50 },
                { text: '📦 Building context (GPT-4 format)...', delay: 150, class: 'info' },
                { text: '  Token budget: 100,000', delay: 100 },
                { text: '  Files selected: 23', delay: 100 },
                { text: '  Files summarized: 7', delay: 100 },
                { text: '  Total tokens: 45,231', delay: 100 },
                { text: '', delay: 50 },
                { text: '✅ Context ready!', delay: 200, class: 'success' },
                { text: '', delay: 50 },
                { text: 'Top files included:', delay: 100, class: 'header' },
                { text: '  98% auth/oauth2_handler.py', delay: 80, class: 'file high' },
                { text: '  95% auth/providers/google.py', delay: 80, class: 'file high' },
                { text: '  93% auth/token_manager.py', delay: 80, class: 'file high' },
                { text: '  89% models/user.py', delay: 80, class: 'file medium' },
                { text: '  87% config/oauth_settings.yaml', delay: 80, class: 'file medium' },
                { text: '  84% api/auth_endpoints.py', delay: 80, class: 'file medium' },
                { text: '  ... and 17 more files', delay: 100, class: 'dim' },
                { text: '', delay: 50 },
                { text: '💾 Output saved to context_oauth2.md', delay: 200, class: 'success' }
            ]
        },
        
        instill: {
            command: 'tenets tenet add "Use type hints for all functions"',
            outputs: [
                { text: '✨ Adding guiding principle...', delay: 100, class: 'info' },
                { text: '', delay: 50 },
                { text: 'Tenet #1 added:', delay: 150, class: 'success' },
                { text: '"Use type hints for all functions"', delay: 100, class: 'quote' },
                { text: '', delay: 50 },
                { text: 'This principle will be included in all future context generation.', delay: 150 },
                { text: '', delay: 100 },
                { text: 'Current tenets (3):', delay: 100, class: 'header' },
                { text: '  1. Use type hints for all functions', delay: 80 },
                { text: '  2. Follow PEP 8 style guidelines', delay: 80 },
                { text: '  3. Write docstrings for public APIs', delay: 80 },
                { text: '', delay: 50 },
                { text: '💡 These principles help maintain consistency when working with AI assistants.', delay: 200, class: 'tip' }
            ]
        },
        
        examine: {
            command: 'tenets examine --complexity --hotspots',
            outputs: [
                { text: '📊 Analyzing codebase...', delay: 100, class: 'info' },
                { text: '', delay: 50 },
                { text: 'Project Statistics:', delay: 150, class: 'header' },
                { text: '  Total files: 1,247', delay: 80 },
                { text: '  Lines of code: 45,678', delay: 80 },
                { text: '  Languages: Python (72%), JavaScript (18%), YAML (10%)', delay: 100 },
                { text: '  Test coverage: 78%', delay: 80, class: 'good' },
                { text: '', delay: 50 },
                { text: 'Complexity Hotspots:', delay: 150, class: 'header' },
                { text: '  🔴 payment/processor.py - Cyclomatic: 42', delay: 100, class: 'high' },
                { text: '  🟡 auth/oauth_handler.py - Cyclomatic: 38', delay: 100, class: 'medium' },
                { text: '  🟡 api/routes.py - Cyclomatic: 35', delay: 100, class: 'medium' },
                { text: '  🟢 utils/validators.py - Cyclomatic: 12', delay: 100, class: 'good' },
                { text: '', delay: 50 },
                { text: 'Frequently Changed Files:', delay: 150, class: 'header' },
                { text: '  models/user.py - 67 changes', delay: 80 },
                { text: '  api/endpoints.py - 45 changes', delay: 80 },
                { text: '  tests/test_auth.py - 38 changes', delay: 80 },
                { text: '', delay: 50 },
                { text: '💡 Consider refactoring payment/processor.py to reduce complexity', delay: 200, class: 'tip' }
            ]
        },
        
        session: {
            command: 'tenets session create payment-feature',
            outputs: [
                { text: '📂 Creating new session: payment-feature', delay: 100, class: 'info' },
                { text: 'Session ID: a3f4b2c1-8d9e-4f5a-b6c7', delay: 150 },
                { text: '', delay: 50 },
                { text: '✅ Session created and activated', delay: 150, class: 'success' },
                { text: '', delay: 50 },
                { text: 'Now you can build context iteratively:', delay: 100 },
                { text: '  tenets distill "design payment flow" --session payment-feature', delay: 150, class: 'code' },
                { text: '  tenets distill "add Stripe integration" --session payment-feature', delay: 150, class: 'code' },
                { text: '', delay: 50 },
                { text: 'Each command builds on previous context.', delay: 100 },
                { text: 'Session state is preserved across runs.', delay: 100 }
            ]
        }
    };
    
    // Terminal animation class
    class TerminalAnimation {
        constructor(element) {
            this.element = element;
            this.outputElement = element.querySelector('.terminal-output');
            this.isRunning = false;
            this.currentAnimation = null;
            
            this.init();
        }
        
        init() {
            // Find run button
            const runBtn = this.element.querySelector('.run-btn');
            if (runBtn) {
                runBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    const example = runBtn.dataset.example || 'distill';
                    this.runExample(example);
                });
            }
            
            // Add static demo notice
            this.addDemoNotice();
        }
        
        addDemoNotice() {
            const notice = document.createElement('div');
            notice.className = 'terminal-notice';
            notice.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
                <span>Static demo - shows example output only</span>
            `;
            notice.style.cssText = `
                position: absolute;
                top: 8px;
                right: 8px;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.25rem 0.75rem;
                background: rgba(245, 158, 11, 0.1);
                border: 1px solid rgba(245, 158, 11, 0.3);
                border-radius: 4px;
                font-size: 0.75rem;
                color: #fbbf24;
                opacity: 0.8;
                z-index: 10;
            `;
            this.element.style.position = 'relative';
            this.element.appendChild(notice);
        }
        
        async runExample(exampleName) {
            if (this.isRunning) return;
            
            const example = terminalExamples[exampleName];
            if (!example) return;
            
            this.isRunning = true;
            
            // Clear output
            this.outputElement.innerHTML = '';
            
            // Show command
            await this.typeCommand(example.command);
            await this.wait(500);
            
            // Show outputs
            for (const output of example.outputs) {
                if (!this.isRunning) break;
                await this.showOutput(output);
                await this.wait(output.delay || 100);
            }
            
            this.isRunning = false;
        }
        
        async typeCommand(command) {
            const line = document.createElement('div');
            line.className = 'terminal-line command-line';
            line.innerHTML = '<span class="prompt">$</span> <span class="command"></span>';
            this.outputElement.appendChild(line);
            
            const cmdSpan = line.querySelector('.command');
            
            for (let i = 0; i < command.length; i++) {
                cmdSpan.textContent += command[i];
                await this.wait(30);
            }
        }
        
        async showOutput(output) {
            const line = document.createElement('div');
            line.className = `terminal-line output-line ${output.class || ''}`;
            
            // Handle different output types
            if (output.class === 'progress') {
                line.innerHTML = output.text;
                line.style.color = '#fbbf24';
            } else if (output.class === 'file') {
                const match = output.text.match(/(\d+%)\s+(.+)/);
                if (match) {
                    const [_, percent, file] = match;
                    const percentNum = parseInt(percent);
                    let colorClass = 'low';
                    if (percentNum >= 90) colorClass = 'high';
                    else if (percentNum >= 70) colorClass = 'medium';
                    
                    line.innerHTML = `<span class="relevance ${colorClass}">${percent}</span> ${file}`;
                } else {
                    line.textContent = output.text;
                }
            } else {
                line.textContent = output.text;
            }
            
            // Add classes for styling
            if (output.class === 'success') line.style.color = '#10b981';
            else if (output.class === 'info') line.style.color = '#60a5fa';
            else if (output.class === 'header') line.style.color = '#fbbf24';
            else if (output.class === 'high') line.style.color = '#ef4444';
            else if (output.class === 'medium') line.style.color = '#fbbf24';
            else if (output.class === 'good') line.style.color = '#10b981';
            else if (output.class === 'dim') line.style.opacity = '0.6';
            else if (output.class === 'quote') {
                line.style.color = '#fbbf24';
                line.style.fontStyle = 'italic';
            }
            else if (output.class === 'code') {
                line.style.fontFamily = 'JetBrains Mono, monospace';
                line.style.color = '#fbbf24';
                line.style.marginLeft = '2rem';
            }
            else if (output.class === 'tip') {
                line.style.color = '#60a5fa';
                line.style.fontStyle = 'italic';
            }
            
            // Animate in
            line.style.opacity = '0';
            line.style.transform = 'translateX(-10px)';
            this.outputElement.appendChild(line);
            
            await this.wait(10);
            line.style.transition = 'all 0.3s ease';
            line.style.opacity = '1';
            line.style.transform = 'translateX(0)';
            
            // Scroll to bottom
            this.outputElement.scrollTop = this.outputElement.scrollHeight;
        }
        
        wait(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }
        
        stop() {
            this.isRunning = false;
        }
    }
    
    // Initialize all terminal examples
    document.querySelectorAll('.terminal-example').forEach(terminal => {
        new TerminalAnimation(terminal);
    });
});