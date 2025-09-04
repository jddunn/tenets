/**
 * Lean Search Fix for Material for MkDocs
 * --------------------------------------
 * Goals:
 *  - Use native Material search lifecycle (checkbox + reset button)
 *  - Avoid fighting built‑in class management
 *  - Provide reliable close/clear UX (ESC, click outside, visible clear button)
 *  - Survive instant navigation (navigation.instant)
 */
(function(){
    'use strict';

    const CONFIG = { debug: true, retryLimit: 20, retryDelay: 120 };
    const STATE = { wired: false, tries: 0 };

    function log(...a){ if(CONFIG.debug) console.log('[SearchFix]', ...a); }

    function isActive(){ return document.body.classList.contains('md-search--active'); }

    function getCheckbox(){ return document.getElementById('__search'); }

    function closeSearch(){
        const cb = getCheckbox();
        const input = document.querySelector('.md-search__input');
        const reset = document.querySelector('.md-search__form [type="reset"]');
        if (input) {
            input.value='';
            input.dispatchEvent(new Event('input', {bubbles:true}));
        }
        if (reset) reset.click();
        if (cb) {
            cb.checked = false; // lets Material remove classes & focus
            cb.dispatchEvent(new Event('change', {bubbles:true}));
        }
        if (input) input.blur();
        log('Closed search via native mechanisms');
    }

    function styleResetButton(reset){
        if (!reset) return;
        if (!reset.classList.contains('sf-reset')) {
            reset.classList.add('sf-reset');
            // Ensure accessible label
            if(!reset.getAttribute('aria-label')) reset.setAttribute('aria-label','Clear search');
            // Replace any SVG with text × (we keep SVG hidden for a11y if present)
            const hasIcon = reset.querySelector('svg');
            if (!hasIcon) reset.textContent = '×';
        }
    }

    function updateResetVisibility(){
        const input = document.querySelector('.md-search__input');
        const reset = document.querySelector('.md-search__form .sf-reset');
        if (!input || !reset) return;
        const show = isActive() && input.value.trim().length > 0;
        reset.dataset.hasValue = show ? 'true' : 'false';
    }

    function wireInput(){
        const input = document.querySelector('.md-search__input');
        if(!input) return false;
        ['input','focus','blur'].forEach(ev=>input.addEventListener(ev, updateResetVisibility,{passive:true}));
        return true;
    }

    function wireReset(){
        const reset = document.querySelector('.md-search__form [type="reset"]');
        if(!reset) return false;
        styleResetButton(reset);
        reset.addEventListener('click', ()=>{
            requestAnimationFrame(()=>{ updateResetVisibility(); });
        });
        return true;
    }

    function wireEsc(){
        document.addEventListener('keydown', (e)=>{
            if (e.key === 'Escape' && isActive()) {
                e.stopPropagation();
                closeSearch();
            }
        }, true); // capture: run before app handlers
    }

    function wireClickOutside(){
        document.addEventListener('pointerdown', (e)=>{
            if(!isActive()) return;
            const root = document.querySelector('.md-search');
            if(root && !root.contains(e.target)) {
                closeSearch();
            }
        });
    }

    function observeBodyClass(){
        const obs = new MutationObserver(muts=>{
            for(const m of muts){
                if(m.type==='attributes' && m.attributeName==='class') {
                    updateResetVisibility();
                }
            }
        });
        obs.observe(document.body,{attributes:true, attributeFilter:['class']});
    }

    function attemptWire(){
        const okInput = wireInput();
        const okReset = wireReset();
        if (okInput && okReset) {
            STATE.wired = true;
            updateResetVisibility();
            log('Search wired');
            return true;
        }
        return false;
    }

    function delayedInit(){
        if (STATE.wired) return;
        if (attemptWire()) return;
        if (STATE.tries++ < CONFIG.retryLimit) {
            setTimeout(delayedInit, CONFIG.retryDelay);
        } else {
            log('Stopped retrying search wiring');
        }
    }

    function onPage(){
        STATE.wired = false; // rewire after instant navigation
        STATE.tries = 0;
        delayedInit();
    }

    function bootstrap(){
        wireEsc();
        wireClickOutside();
        observeBodyClass();
        onPage();
        // Instant navigation hook (Material exposes Rx Subject document$)
        if (window.document$ && typeof window.document$.subscribe === 'function') {
            window.document$.subscribe(()=>{ onPage(); });
            log('Subscribed to document$ navigation events');
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bootstrap);
    } else {
        bootstrap();
    }

    window.SearchFix = { close: closeSearch, rewire: onPage, config: CONFIG };
})();
