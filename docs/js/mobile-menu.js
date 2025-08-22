// Mobile menu: keep Material's built-in nav; just close drawer on link click
(function(){
  function closeDrawerOnClick(){
    document.addEventListener('click', (e)=>{
      const link = e.target.closest('.md-sidebar--primary a, .md-nav--primary a');
      if(!link) return;
      const drawerToggle = document.getElementById('__drawer');
      if(drawerToggle && drawerToggle.checked){ drawerToggle.checked = false; }
    });
  }

  function init(){
    closeDrawerOnClick();
  }

  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
