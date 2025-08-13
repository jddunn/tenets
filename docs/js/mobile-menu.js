// Mobile menu enhancements (stub)
// Ensures existence since referenced in mkdocs.yml
// You can extend this to add analytics or state handling for drawer open/close.
(function(){
  function onToggle(){
    // Placeholder for future logic
  }
  document.addEventListener('DOMContentLoaded', ()=>{
    const drawerToggle = document.querySelector('label.md-header__button[for="__drawer"]');
    if(drawerToggle){ drawerToggle.addEventListener('click', onToggle); }
  });
})();
