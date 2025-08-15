// Mobile menu: replace drawer contents with main nav links and close on click
(function(){
  const ROOT = '/';
  const MAIN_NAV = [
    { label: 'Home', href: ROOT },
    { label: 'Quick Start', href: ROOT + 'quickstart/' },
    { label: 'Features', href: ROOT + 'features/' },
    { label: 'Supported Languages', href: ROOT + 'supported-languages/' },
    { label: 'Documentation', href: ROOT + 'docs/' },
    { label: 'Reference', children: [
      { label: 'CLI Reference', href: ROOT + 'CLI/' },
      { label: 'Configuration', href: ROOT + 'CONFIG/' },
      { label: 'Architecture', href: ROOT + 'ARCHITECTURE/' },
    ]},
    { label: 'API Reference', href: ROOT + 'api/' },
    { label: 'Blog', href: 'https://manic.agency/blog', external: true },
    { label: 'Contact', href: 'https://manic.agency/contact', external: true }
  ];

  function buildList(items){
    const ul = document.createElement('ul');
    ul.className = 'md-nav__list tenets-mobile-nav';
    items.forEach(item => {
      const li = document.createElement('li'); li.className = 'md-nav__item';
      const a = document.createElement('a');
      a.className = 'md-nav__link';
      a.textContent = item.label;
      a.href = item.href;
      if(item.external){ a.target = '_blank'; a.rel = 'noopener'; }
      li.appendChild(a);
      if(item.children && item.children.length){
        li.classList.add('md-nav__item--nested');
        li.appendChild(buildList(item.children));
      }
      ul.appendChild(li);
    });
    return ul;
  }

  function replaceDrawer(){
    const drawer = document.querySelector('.md-sidebar--primary .md-nav');
    if(!drawer) return;
    // Clear existing
    drawer.innerHTML = '';
    // Insert header
    const hdr = document.createElement('div'); hdr.className = 'tenets-drawer-header'; hdr.textContent = 'Navigation';
    drawer.appendChild(hdr);
    // Insert our main nav
    drawer.appendChild(buildList(MAIN_NAV));
  }

  function closeDrawerOnClick(){
    document.addEventListener('click', (e)=>{
      const link = e.target.closest('.tenets-mobile-nav a');
      if(!link) return;
      const drawerToggle = document.getElementById('__drawer');
      if(drawerToggle && drawerToggle.checked){ drawerToggle.checked = false; }
    });
  }

  function init(){
    replaceDrawer();
    closeDrawerOnClick();
    // Re-apply after navigation.instant replaces content
    const mo = new MutationObserver(()=>{ replaceDrawer(); });
    mo.observe(document.documentElement, {childList:true, subtree:true});
  }

  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
