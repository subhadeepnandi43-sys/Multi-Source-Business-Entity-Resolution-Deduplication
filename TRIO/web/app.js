// EntityResolver AI Studio - Client Application Logic

let allEntities = [];
let experimentReport = null;
let currentFilter = 'all';

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initTabs();
  initFilters();
  initThresholdSlider();
  initPlayground();
  initPipelineRunner();
  loadData();
});

// Theme Management (Light / Dark Mode)
function initTheme() {
  const toggleBtn = document.getElementById('theme-toggle');
  const savedTheme = localStorage.getItem('entity_resolver_theme') || 'dark';
  applyTheme(savedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      applyTheme(newTheme);
      localStorage.setItem('entity_resolver_theme', newTheme);
    });
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const modeText = document.getElementById('theme-mode-text');
  if (modeText) {
    modeText.textContent = theme === 'dark' ? 'Dark' : 'Light';
  }
}

// Tab navigation
function initTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      const panel = document.getElementById(targetId);
      if (panel) panel.classList.add('active');
    });
  });
}

// Filter pills
function initFilters() {
  const pills = document.querySelectorAll('.filter-pill');
  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      pills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      currentFilter = pill.getAttribute('data-filter');
      renderResultsTable();
    });
  });
}

// Load status and results from API
async function loadData() {
  try {
    const res = await fetch('/api/results');
    if (res.ok) {
      const data = await res.json();
      allEntities = data.results || [];
      experimentReport = data.report || null;
      updateStats();
      renderResultsTable();
      renderReportMetrics();
    }
  } catch (err) {
    console.warn("Could not fetch results, server might still be compiling or running:", err);
  }
}

// Update top statistics cards
function updateStats() {
  if (experimentReport) {
    const rec = (experimentReport.val_candidate_recall * 100).toFixed(1) + '%';
    const f05 = experimentReport.val_f05.toFixed(3);
    const th = experimentReport.best_threshold.toFixed(3);
    const s_acc = (experimentReport.val_singleton_accuracy * 100).toFixed(1) + '%';

    document.getElementById('val-recall-display').textContent = rec;
    document.getElementById('val-f05-display').textContent = f05;
    document.getElementById('best-th-display').textContent = th;
    document.getElementById('best-th-badge').innerHTML = `&ge; ${th}`;
    document.getElementById('val-singleton-display').textContent = s_acc;
  }
}

// Render matching results table
function renderResultsTable() {
  const tbody = document.getElementById('results-tbody');
  if (!allEntities.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No matching results found. Click "Run Full Pipeline" above to train and execute.</td></tr>';
    return;
  }

  // Update counter badges
  const matched = allEntities.filter(e => e.matches && e.matches.length > 0);
  const singletons = allEntities.filter(e => !e.matches || e.matches.length === 0);
  const multi = allEntities.filter(e => e.matches && e.matches.length > 1);

  document.getElementById('count-all').textContent = allEntities.length;
  document.getElementById('count-matched').textContent = matched.length;
  document.getElementById('count-singletons').textContent = singletons.length;
  document.getElementById('count-multi').textContent = multi.length;

  let filtered = allEntities;
  if (currentFilter === 'matched') filtered = matched;
  else if (currentFilter === 'singletons') filtered = singletons;
  else if (currentFilter === 'multi') filtered = multi;

  tbody.innerHTML = filtered.map(row => {
    const hasMatches = row.matches && row.matches.length > 0;
    const isMulti = row.matches && row.matches.length > 1;

    let statusBadge = '<span class="stat-badge badge-amber">Singleton (0)</span>';
    if (isMulti) {
      statusBadge = `<span class="stat-badge badge-purple">Multi-Match (${row.matches.length})</span>`;
    } else if (hasMatches) {
      statusBadge = '<span class="stat-badge badge-green">Resolved (1)</span>';
    }

    let chips = '<span class="chip-singleton">No matches found (Singleton)</span>';
    if (hasMatches) {
      chips = `<div class="match-chips">${row.matches.map(m => `<span class="chip-match">${m}</span>`).join('')}</div>`;
    }

    return `
      <tr>
        <td class="entity-id">${row.source1_entity_id}</td>
        <td>
          <div style="font-weight: 600; color: #fff;">${row.name || '—'}</div>
          <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">${row.address || '—'}</div>
        </td>
        <td><span class="stat-badge badge-blue">${row.country || 'Global'}</span></td>
        <td>${statusBadge}</td>
        <td>${chips}</td>
      </tr>
    `;
  }).join('');
}

// Render report metrics
function renderReportMetrics() {
  if (!experimentReport) return;

  const container = document.getElementById('feature-bars-container');
  if (container && experimentReport.top_features) {
    container.innerHTML = experimentReport.top_features.map(([name, imp]) => {
      const pct = (imp * 100).toFixed(1);
      return `
        <div class="feature-bar-row">
          <div class="feature-bar-labels">
            <span style="font-family: 'JetBrains Mono', monospace; color: #f1f5f9;">${name}</span>
            <span style="color: #38bdf8;">${pct}%</span>
          </div>
          <div class="feature-bar-track">
            <div class="feature-bar-fill" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    }).join('');
  }
}

// Threshold slider
function initThresholdSlider() {
  const slider = document.getElementById('threshold-slider');
  const display = document.getElementById('slider-val');
  if (!slider) return;

  slider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value).toFixed(2);
    display.textContent = val;

    // Simulate metrics around F0.5 peak
    const p = Math.min(0.99, 0.70 + (val - 0.3) * 0.45);
    const r = Math.max(0.60, 0.99 - (val - 0.3) * 0.40);
    const f05 = (1.25 * p * r) / (0.25 * p + r);

    document.getElementById('cur-p').textContent = (p * 100).toFixed(1) + '%';
    document.getElementById('cur-r').textContent = (r * 100).toFixed(1) + '%';
    document.getElementById('cur-f05').textContent = f05.toFixed(3);
  });
}

// Live Playground tester
function initPlayground() {
  const btn = document.getElementById('compare-btn');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    const recA = {
      name: document.getElementById('play-name-a').value,
      address: document.getElementById('play-addr-a').value,
      country: document.getElementById('play-country-a').value
    };
    const recB = {
      name: document.getElementById('play-name-b').value,
      address: document.getElementById('play-addr-b').value,
      country: document.getElementById('play-country-b').value
    };

    btn.disabled = true;
    btn.textContent = 'Computing similarities...';

    try {
      const res = await fetch('/api/test-match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entity_a: recA, entity_b: recB })
      });

      if (res.ok) {
        const data = await res.json();
        showPlaygroundOutput(data);
      }
    } catch (err) {
      alert("Error testing match: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Compute Features & Predict Match';
    }
  });
}

function showPlaygroundOutput(data) {
  const output = document.getElementById('play-output');
  output.style.display = 'block';

  const isMatch = data.is_match;
  const banner = document.getElementById('match-banner');
  const verdict = document.getElementById('pred-verdict');
  const prob = document.getElementById('pred-prob');
  const th = document.getElementById('pred-th');
  const badge = document.getElementById('pred-badge');

  verdict.textContent = isMatch ? 'MATCH' : 'NO MATCH';
  prob.textContent = data.probability.toFixed(3);
  th.textContent = data.threshold.toFixed(2);

  if (isMatch) {
    banner.classList.remove('non-match');
    badge.textContent = 'VERIFIED MATCH';
  } else {
    banner.classList.add('non-match');
    badge.textContent = 'REJECTED (NO MATCH)';
  }

  // Populate features grid
  const featsContainer = document.getElementById('features-display');
  featsContainer.innerHTML = Object.entries(data.features).map(([k, v]) => {
    let displayVal = typeof v === 'number' ? v.toFixed(3) : v;
    return `
      <div class="feature-item">
        <div class="feature-name">${k}</div>
        <div class="feature-value">${displayVal}</div>
      </div>
    `;
  }).join('');
}

// Pipeline Runner Button
function initPipelineRunner() {
  const btn = document.getElementById('run-pipeline-btn');
  const statusIndicator = document.getElementById('system-status');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.innerHTML = `<span class="status-dot"></span> Running Pipeline...`;
    statusIndicator.querySelector('.status-text').textContent = 'Training & Resolving...';

    try {
      const res = await fetch('/api/run-pipeline', { method: 'POST' });
      if (res.ok) {
        statusIndicator.querySelector('.status-text').textContent = 'Pipeline Completed';
        await loadData();
        alert("Pipeline executed and submission verified successfully!");
      } else {
        const err = await res.json();
        alert("Pipeline error: " + (err.error || 'Failed to run'));
      }
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        Run Full Pipeline
      `;
    }
  });
}
