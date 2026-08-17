// Charts inside <details> render while hidden and keep a stale size when
// opened — nudge Plotly with a resize whenever any details element toggles.
document.addEventListener("toggle", function () {
  setTimeout(function () { window.dispatchEvent(new Event("resize")); }, 60);
}, true);
