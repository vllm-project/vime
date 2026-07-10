// Inject a language toggle button into the topbar (sphinx-book-theme compatible)
(function(){
  const STORAGE_KEY = 'vime-doc-lang';
  const ZH_SEGMENT = 'zh-cn';
  // Default language EN has no URL prefix; Chinese uses 'zh-cn' inserted after optional repo root.
  function detectCurrent(){
    const { langIndex, parts } = analyzePath();
    if(langIndex === -1) return 'en';
    return parts[langIndex] === ZH_SEGMENT ? 'zh' : 'en';
  }
  function otherLang(lang){ return lang === 'zh' ? 'en' : 'zh'; }
  /**
   * Analyze current pathname to figure out repo root + language segment pattern.
   * Supports patterns:
   *  /en/…                (language as first segment)
   *  /vime/en/…          (GitHub Pages project site repo root, language second)
   *  /projects/vime/en/…  (Read the Docs project prefix, language third)
   *  /vime/ (no lang yet) -> insert /vime/zh-cn/
   *  /projects/vime/ (no lang yet) -> insert /projects/vime/zh-cn/
   *  / (no lang) -> insert /zh/
   */
  function analyzePath(){
    const rawParts = window.location.pathname.split('/').filter(Boolean);
    const parts = rawParts.slice();
    let repoRootLen = 0;
    if(parts[0] === 'projects' && parts[1] === 'vime'){
      repoRootLen = 2;
    } else if(parts.length > 0 && (window.location.host.endsWith('github.io') || parts[0] === 'vime')){
      repoRootLen = 1;
    }
    let langIndex = -1;
    for(let i = repoRootLen; i < Math.min(parts.length, repoRootLen + 3); i++){
      if(parts[i] === 'en' || parts[i] === ZH_SEGMENT){
        langIndex = i;
        break;
      }
    }
    return { parts, repoRootLen, langIndex };
  }

  function buildTargetUrl(target){
    const url = new URL(window.location.href);
    const trailingSlash = url.pathname.endsWith('/') || url.pathname === '/';
    const { parts, repoRootLen, langIndex } = analyzePath();
    if(langIndex !== -1){
      parts.splice(langIndex, 1);
    }
    if(repoRootLen > 0){
      parts.splice(repoRootLen, 0, target === 'zh' ? ZH_SEGMENT : 'en');
    } else if(target === 'zh'){
      parts.unshift(ZH_SEGMENT);
    } else if(parts[0] !== 'en'){
      parts.unshift('en');
    }
    let newPath = '/' + parts.join('/');
    if(newPath === '/') {
      // stay root
    } else if(trailingSlash && !/\.[a-zA-Z0-9]+$/.test(parts[parts.length-1] || '')) newPath += '/';
    url.pathname = newPath;
    return url.toString();
  }
  function createButton(){
    const current = detectCurrent();
    const tgt = otherLang(current);
    const btn = document.createElement('button');
    btn.className = 'btn btn-sm lang-toggle-btn';
    btn.type = 'button';
    btn.setAttribute('data-current', current);
    btn.title = current === 'en' ? '切换到中文 (当前 EN)' : 'Switch to English (当前 中文)';
    // Show two labels with active highlighted via CSS
    btn.innerHTML = `
      <span class="lang-seg" data-lang="en">EN</span>
      <span class="lang-sep">/</span>
      <span class="lang-seg" data-lang="zh">中</span>
    `;
    btn.addEventListener('click', ()=>{
      const targetUrl = buildTargetUrl(tgt);
      try{ localStorage.setItem(STORAGE_KEY, tgt);}catch(e){}
      window.location.href = targetUrl;
    });
    return btn;
  }
  function findContainer(){
    // Priority: sidebar end area or header button groups
    return document.querySelector(
      '.article-header-buttons, .header-article-items__end, .sidebar-header-items, .sidebar-primary-items__end, .bd-header, .bd-sidebar'
    );
  }
  function idealContainer(){
    return document.querySelector('.article-header-buttons');
  }
  function insert(attempt=0){
    // Prefer final header buttons group
    let c = idealContainer();
    if(!c) c = findContainer();
    if(!c){
      if(attempt < 40) return setTimeout(()=>insert(attempt+1), 125);
      return;
    }
    // If button exists elsewhere but not inside ideal container, move it
    const existing = document.querySelector('.lang-toggle-btn');
    if(existing && c !== existing.parentElement){
      c.appendChild(existing);
      return;
    }
    if(existing) return; // already good
    const btn = createButton();
    // Insert near theme switch button if present
    const themeBtn = c.querySelector('.theme-switch-button');
    // Insert just before theme switch if found, else at end
    if(themeBtn){
      const parent = themeBtn.parentElement;
      if(parent === c) c.insertBefore(btn, themeBtn);
      else c.appendChild(btn);
    } else c.appendChild(btn);
    // If current is zh ensure active highlighting reflects zh
    btn.setAttribute('data-current', detectCurrent());
  }
  document.addEventListener('DOMContentLoaded', ()=>{
    insert();
    // Observe for dynamic header injection
  const obs = new MutationObserver(()=>{ insert(); });
    obs.observe(document.body, {childList:true, subtree:true});
    // Stop observing after 5s
    setTimeout(()=>obs.disconnect(), 5000);
  });
})();

// Minimal styling; can be overridden in custom css
// Light structural styling now moved to CSS file; keep minimal runtime if CSS missing.
(function(){
  if(document.querySelector('style[data-lang-toggle-style]')) return;
  const style = document.createElement('style');
  style.setAttribute('data-lang-toggle-style','');
  style.textContent = `.lang-toggle-btn{display:inline-flex;align-items:center;gap:2px}`;
  document.head.appendChild(style);
})();
