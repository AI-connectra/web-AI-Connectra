    let currentDeptKey = null;
    let currentProjIndex = 0;
    let currentCourseId = 'ml';
    let currentRelevanceDeptKey = 'eee';

    // DOM Views Map
    const allViews = [
      'viewSubjects',
      'viewDepartment',
      'viewProjectDetail',
      'viewCourse',
      'viewCompareTracks',
      'viewDisciplineRelevance'
    ];

    // ====================================================================
    // CANVAS BACKGROUND ANIMATION (SERENE BLUE PARTICLES)
    // ====================================================================
    let blueCanvas, blueCtx, blueAnimId;
    let blueParticles = [];

    function initBlueParticles() {
      blueCanvas = document.getElementById('heroBlueCanvas');
      if (!blueCanvas) return;
      blueCtx = blueCanvas.getContext('2d');
      resizeBlueCanvas();
      window.addEventListener('resize', resizeBlueCanvas);

      blueParticles = [];
      const count = Math.min(Math.floor(window.innerWidth / 22), 65);
      for (let i = 0; i < count; i++) {
        blueParticles.push({
          x: Math.random() * blueCanvas.width,
          y: Math.random() * blueCanvas.height,
          radius: Math.random() * 2.2 + 0.8,
          vx: (Math.random() - 0.5) * 0.45,
          vy: (Math.random() - 0.5) * 0.45,
          alpha: Math.random() * 0.55 + 0.25,
          color: Math.random() > 0.4 ? '#38bdf8' : '#2563eb'
        });
      }
      animateBlueHero();
    }

    function resizeBlueCanvas() {
      if (!blueCanvas) return;
      const hero = document.getElementById('heroStage');
      if (hero) {
        blueCanvas.width = hero.clientWidth;
        blueCanvas.height = hero.clientHeight;
      }
    }

    function animateBlueHero() {
      if (!blueCtx || !blueCanvas) return;
      blueCtx.clearRect(0, 0, blueCanvas.width, blueCanvas.height);

      for (let i = 0; i < blueParticles.length; i++) {
        for (let j = i + 1; j < blueParticles.length; j++) {
          const dx = blueParticles[i].x - blueParticles[j].x;
          const dy = blueParticles[i].y - blueParticles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 110) {
            blueCtx.beginPath();
            blueCtx.moveTo(blueParticles[i].x, blueParticles[i].y);
            blueCtx.lineTo(blueParticles[j].x, blueParticles[j].y);
            blueCtx.strokeStyle = `rgba(56, 189, 248, ${0.18 * (1 - dist / 110)})`;
            blueCtx.lineWidth = 0.8;
            blueCtx.stroke();
          }
        }
      }

      blueParticles.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x = blueCanvas.width;
        if (p.x > blueCanvas.width) p.x = 0;
        if (p.y < 0) p.y = blueCanvas.height;
        if (p.y > blueCanvas.height) p.y = 0;

        blueCtx.beginPath();
        blueCtx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        blueCtx.fillStyle = p.color;
        blueCtx.globalAlpha = p.alpha;
        blueCtx.fill();
        blueCtx.globalAlpha = 1.0;
      });

      blueAnimId = requestAnimationFrame(animateBlueHero);
    }

    // ====================================================================
    // VIEW SWITCHING & ROUTING (CLEAN ARCHITECTURE)
    // ====================================================================
    function switchView(targetViewId) {
      const isProjectView = targetViewId === 'viewProjectDetail';
      const projectView = document.getElementById('viewProjectDetail');
      if (projectView) {
        projectView.hidden = !isProjectView;
        // Inline display is intentional: it prevents any generic .active-stage
        // rule or cached stylesheet from leaking project-only content into
        // catalog, course, comparison, or relevance pages.
        projectView.style.display = isProjectView ? '' : 'none';
      }

      allViews.forEach(vId => {
        const el = document.getElementById(vId);
        if (el) el.classList.remove('active-stage');
      });

      // Project-only material (facts, roadmap, deliverables, and next/previous links)
      // must never be exposed by the catalog, course, or other non-project views.
      document.querySelectorAll('#viewProjectDetail .project-detail-hero, #viewProjectDetail .project-page-content')
        .forEach(section => { section.hidden = !isProjectView; });

      const target = document.getElementById(targetViewId);
      if (target) {
        target.classList.add('active-stage');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }

    function navigateToSubjects() {
      currentDeptKey = null;
      currentProjIndex = 0;
      safeSetHash('#subjects');
      switchView('viewSubjects');
    }

    function scrollToCourses() {
      if (!document.getElementById('viewSubjects').classList.contains('active-stage')) {
        navigateToSubjects();
        setTimeout(() => {
          const el = document.getElementById('coursesSectionHeading');
          if (el) el.scrollIntoView({ behavior: 'smooth' });
        }, 200);
      } else {
        const el = document.getElementById('coursesSectionHeading');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }
    }

    function scrollToSubjects() {
      if (!document.getElementById('viewSubjects').classList.contains('active-stage')) {
        navigateToSubjects();
        setTimeout(() => {
          const el = document.getElementById('subjectsSectionHeading');
          if (el) el.scrollIntoView({ behavior: 'smooth' });
        }, 200);
      } else {
        const el = document.getElementById('subjectsSectionHeading');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }
    }

    function scrollToVideoLecture() {
      const el = document.getElementById('videoLectureSection');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }

    function openCourseView(courseId = 'ml') {
      currentCourseId = courseId;
      safeSetHash(`#course-${courseId}`);
      switchCourseTrack(courseId);
      switchView('viewCourse');
    }

    function switchCourseTrack(courseId) {
      currentCourseId = courseId;
      const c = coursesData[courseId] || coursesData['ml'];

      ['python', 'ml', 'dl'].forEach(id => {
        const tabEl = document.getElementById(`tabCourse${id.toUpperCase()}`);
        if (tabEl) {
          if (id === courseId) tabEl.classList.add('active');
          else tabEl.classList.remove('active');
        }
      });

      const breadcrumb = document.getElementById('courseBreadcrumbTitle');
      if (breadcrumb) breadcrumb.textContent = `🎓 ${c.title}`;

      const heroBadge = document.getElementById('courseHeroBadge');
      if (heroBadge) heroBadge.textContent = c.badge || '✦ Flagship Interdisciplinary Masterclass ✦';

      const titleEl = document.getElementById('courseMainTitle');
      if (titleEl) titleEl.innerHTML = `${c.title} <br><span class="gradient-text-blue">${c.short} Curriculum</span>`;

      const descEl = document.getElementById('courseMainDesc');
      if (descEl) descEl.innerHTML = `${c.freeDesc} <strong>Paid Practical Track:</strong> ${c.paidDesc}`;

      const iframe = document.getElementById('courseVideoIframe');
      if (iframe && c.embedUrl) iframe.src = c.embedUrl;

      const plTitle = document.getElementById('playlistTitleText');
      if (plTitle) plTitle.textContent = c.playlistName;

      const plExt = document.getElementById('btnOpenPlaylistExternal');
      if (plExt) plExt.href = c.playlistUrl;

      const modulesList = document.getElementById('courseModulesList');
      if (modulesList && c.modules) {
        modulesList.innerHTML = c.modules.map(m => `
          <div class="module-item-row">
            <div class="module-left-matter">
              <span class="module-number-pill">${m.num}</span>
              <div class="module-text-block">
                <div class="module-heading-title">${m.title}</div>
                <div class="module-desc-topics">${m.desc}</div>
              </div>
            </div>
            <span class="module-tier-tag ${m.tier === 'free' ? 'tag-theory-free' : 'tag-practical-paid'}">
              ${m.tier === 'free' ? 'Free Theory' : 'Paid Practical'}
            </span>
          </div>
        `).join('');
      }
    }

    // ====================================================================
    // OPEN DEDICATED VIEW 5: COMPARE FREE VS PAID TRACKS PAGE
    // ====================================================================
    function safeSetHash(newHash) {
      try {
        if (window.location.protocol === 'file:') {
          window.location.hash = newHash;
        } else {
          history.pushState(null, '', newHash);
        }
      } catch (e) {
        try { window.location.hash = newHash; } catch (err) {}
      }
    }

    function openCompareTracksView() {
      safeSetHash('#compare-tracks');
      switchView('viewCompareTracks');
    }

    // ====================================================================
    // OPEN DEDICATED VIEW 6: HOW AI RELATES TO YOUR DISCIPLINE PAGE
    // ====================================================================
    function openDisciplineRelevanceView(deptKey = 'eee') {
      currentRelevanceDeptKey = deptKey;
      safeSetHash(`#discipline-relevance-${deptKey}`);
      renderDedicatedDisciplinePage(deptKey);
      switchView('viewDisciplineRelevance');
    }

    function renderDedicatedDisciplinePage(deptKey) {
      currentRelevanceDeptKey = deptKey;
      const data = essentialDisciplineData[deptKey] || essentialDisciplineData['eee'];
      const dept = projectsData[data.deptKey || deptKey] || {};
      const projs = dept.projects || [];

      // Update tabs on dedicated page
      const tabs = document.querySelectorAll('#pageRelevanceTabs .dept-tab-btn');
      tabs.forEach(t => t.classList.remove('active'));
      const activeTab = Array.from(tabs).find(t => t.textContent.toLowerCase().includes(deptKey.toLowerCase()) || (deptKey === 'cs' && t.textContent.includes('Computer Science')));
      if (activeTab) activeTab.classList.add('active');

      const container = document.getElementById('dedicatedDisciplineContent');
      if (!container) return;

      container.innerHTML = `
        <!-- Section 1: The Paradigm Shift -->
        <div class="department-banner" style="margin-bottom: 36px;">
          <div class="dept-banner-content" style="max-width: 100%;">
            <div class="dept-banner-top">
              <div class="dept-banner-icon">${data.icon}</div>
              <div>
                <h2 class="dept-banner-title">${data.name}</h2>
                <span class="dept-metric-tag" style="margin-top: 6px; display: inline-block;">AI Transformation & Industry Mapping</span>
              </div>
            </div>
            <p class="dept-banner-desc" style="margin-bottom: 16px;">
              ${data.shiftSummary}
            </p>
            <div class="dept-metric-tag" style="background: rgba(16, 185, 129, 0.15); border-color: #10b981; color: #34d399;">
              <span>📈 Quantified Impact:</span> <span>${data.roiMetric}</span>
            </div>
          </div>
        </div>

        <!-- Section 2: 3 Pillars Deep Dive for this Department -->
        <div class="portal-heading-wrap" style="text-align: left; margin-bottom: 24px;">
          <span class="section-tag-badge">THE 3 AI PILLARS</span>
          <h3>Why Each Course is Mission-Critical for ${data.name}</h3>
        </div>

        <div class="essential-cards-grid" style="margin-bottom: 48px;">
          <!-- Python Card -->
          <div class="dept-essential-card" style="--pillar-color: #38bdf8; --pillar-glow: rgba(56, 189, 248, 0.35);">
            <div class="essential-card-head">
              <div class="essential-icon-pill" style="border-color: #38bdf8; color: #38bdf8;">🐍</div>
              <div class="essential-card-title">${data.python.title}</div>
            </div>
            <p class="essential-card-body">${data.python.desc}</p>
            <span class="essential-card-footer">Key Competency: ${data.python.keySkills}</span>
          </div>

          <!-- Machine Learning Card -->
          <div class="dept-essential-card" style="--pillar-color: #2563eb; --pillar-glow: rgba(37, 99, 235, 0.35);">
            <div class="essential-card-head">
              <div class="essential-icon-pill" style="border-color: #2563eb; color: #2563eb;">🤖</div>
              <div class="essential-card-title">${data.ml.title}</div>
            </div>
            <p class="essential-card-body">${data.ml.desc}</p>
            <span class="essential-card-footer">Key Competency: ${data.ml.keySkills}</span>
          </div>

          <!-- Deep Learning Card -->
          <div class="dept-essential-card" style="--pillar-color: #8b5cf6; --pillar-glow: rgba(139, 92, 246, 0.35);">
            <div class="essential-card-head">
              <div class="essential-icon-pill" style="border-color: #8b5cf6; color: #8b5cf6;">🧠</div>
              <div class="essential-card-title">${data.dl.title}</div>
            </div>
            <p class="essential-card-body">${data.dl.desc}</p>
            <span class="essential-card-footer">Key Competency: ${data.dl.keySkills}</span>
          </div>
        </div>

        <!-- Section 3: Dedicated Production Blueprints Showcase -->
        <div class="portal-heading-wrap" style="text-align: left; margin-bottom: 24px;">
          <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
            <div>
              <span class="section-tag-badge">APPLIED PRODUCTION SHOWCASE</span>
              <h3>20 Dedicated Blueprints for ${data.name}</h3>
              <p style="color: var(--text-secondary); margin: 0;">Click any blueprint below to inspect its full neural architecture and Python code:</p>
            </div>
            <button class="btn-primary-blue" onclick="openSubjectProjects('${data.deptKey || deptKey}')">
              <span>⚡ View All 10 in Department View</span>
              <span>→</span>
            </button>
          </div>
        </div>

        <div class="ten-projects-grid-expanded" style="margin-bottom: 50px;">
          ${projs.map((p, idx) => `
            <div class="project-card-interactive" onclick="openProjectDetail('${data.deptKey || deptKey}', ${idx})">
              <div class="project-index-pill">#${String(idx + 1).padStart(2, '0')}</div>
              
              <div class="project-card-top-matter">
                <span class="project-category-lbl">${data.name}</span>
                <h4 class="project-card-heading">${p.title}</h4>
                <p class="project-card-summary">${p.description}</p>
              </div>

              <div>
                <div class="tech-stack-tag-row">
                  ${(p.tech_stack || []).map(t => `<span class="tech-tag-pill">${t}</span>`).join('')}
                </div>
                <div class="btn-project-deepdive">
                  <span>Deep-Dive Blueprint</span>
                  <span>→</span>
                </div>
              </div>
            </div>
          `).join('')}
        </div>
      `;
    }

    // ====================================================================
    // INTERACTIVE WHY EACH COURSE IS ESSENTIAL (DEFAULT HOME PAGE)
    // ====================================================================
    function switchEssentialDept(deptKey) {
      const data = essentialDisciplineData[deptKey] || essentialDisciplineData['eee'];

      const tabs = document.querySelectorAll('#essentialDeptTabs .dept-tab-btn');
      tabs.forEach(t => t.classList.remove('active'));
      const activeTab = Array.from(tabs).find(t => t.textContent.toLowerCase().includes(deptKey.toLowerCase()) || (deptKey === 'cs' && t.textContent.includes('Computer Science')));
      if (activeTab) activeTab.classList.add('active');

      const grid = document.getElementById('essentialCardsGrid');
      if (!grid) return;

      grid.innerHTML = `
        <!-- Python Card -->
        <div class="dept-essential-card" style="--pillar-color: #38bdf8; --pillar-glow: rgba(56, 189, 248, 0.35);">
          <div class="essential-card-head">
            <div class="essential-icon-pill" style="border-color: #38bdf8; color: #38bdf8;">🐍</div>
            <div class="essential-card-title">${data.python.title}</div>
          </div>
          <p class="essential-card-body">${data.python.desc}</p>
          <span class="essential-card-footer">Key Competency: ${data.python.keySkills}</span>
        </div>

        <!-- Machine Learning Card -->
        <div class="dept-essential-card" style="--pillar-color: #2563eb; --pillar-glow: rgba(37, 99, 235, 0.35);">
          <div class="essential-card-head">
            <div class="essential-icon-pill" style="border-color: #2563eb; color: #2563eb;">🤖</div>
            <div class="essential-card-title">${data.ml.title}</div>
          </div>
          <p class="essential-card-body">${data.ml.desc}</p>
          <span class="essential-card-footer">Key Competency: ${data.ml.keySkills}</span>
        </div>

        <!-- Deep Learning Card -->
        <div class="dept-essential-card" style="--pillar-color: #8b5cf6; --pillar-glow: rgba(139, 92, 246, 0.35);">
          <div class="essential-card-head">
            <div class="essential-icon-pill" style="border-color: #8b5cf6; color: #8b5cf6;">🧠</div>
            <div class="essential-card-title">${data.dl.title}</div>
          </div>
          <p class="essential-card-body">${data.dl.desc}</p>
          <span class="essential-card-footer">Key Competency: ${data.dl.keySkills}</span>
        </div>
      `;

      const btnExplore = document.getElementById('btnExploreSelectedDeptBlueprints');
      if (btnExplore) {
        btnExplore.innerHTML = `<span>${data.icon} Explore 20 Dedicated ${data.name} Blueprints</span> <span>→</span>`;
        btnExplore.onclick = () => openSubjectProjects(data.deptKey || deptKey);
      }
    }

    // ====================================================================
    // DEPARTMENT & PROJECT VIEW RENDERING
    // ====================================================================
    function openSubjectProjects(deptKey) {
      currentDeptKey = deptKey;
      currentProjIndex = 0;
      safeSetHash(`#dept-${deptKey}`);
      renderTenProjects(deptKey);
      switchView('viewDepartment');
    }

    function openProjectDetail(deptKey, projIndex) {
      currentDeptKey = deptKey;
      currentProjIndex = projIndex;
      safeSetHash(`#proj-${deptKey}-${projIndex}`);
      renderProjectCockpit(deptKey, projIndex);
      switchView('viewProjectDetail');
    }

    function backToCurrentDept() {
      if (currentDeptKey) {
        openSubjectProjects(currentDeptKey);
      } else {
        navigateToSubjects();
      }
    }

    function navigateProjectOffset(offset) {
      if (!currentDeptKey) return;
      const dept = projectsData[currentDeptKey];
      if (!dept || !dept.projects) return;
      const projs = dept.projects;
      let newIdx = currentProjIndex + offset;
      if (newIdx < 0) newIdx = projs.length - 1;
      if (newIdx >= projs.length) newIdx = 0;
      openProjectDetail(currentDeptKey, newIdx);
    }

    function renderSubjectsCatalog() {
      const grid = document.getElementById('subjectsGrid');
      if (!grid) return;
      grid.innerHTML = '';

      Object.keys(projectsData).forEach(deptKey => {
        const dept = projectsData[deptKey];
        const meta = deptMetadata[deptKey] || {};
        const projs = dept.projects || [];
        const top3 = projs.slice(0, 3);

        const card = document.createElement('div');
        card.className = 'subject-block-card';
        card.style.setProperty('--dept-color', meta.color || '#3b82f6');
        card.style.setProperty('--dept-bg', meta.bg || 'rgba(59, 130, 246, 0.2)');
        card.style.setProperty('--dept-glow', meta.glow || 'rgba(59, 130, 246, 0.45)');

        card.innerHTML = `
          <div class="subject-card-top">
            <div class="subject-icon-badge">${meta.icon || '⚡'}</div>
            <div class="subject-badge-count">
              <span>●</span> <span>${projs.length} Dedicated Blueprints</span>
            </div>
            <h3 class="subject-card-title">${dept.name}</h3>
            <p class="subject-card-desc">${dept.description}</p>

            <div class="subject-top-projects-preview">
              <span class="preview-header-lbl">FEATURED ARCHITECTURES:</span>
              <ul class="preview-proj-list">
                ${top3.map((p, pIdx) => `
                  <li class="preview-proj-item" onclick="event.stopPropagation(); openProjectDetail('${deptKey}', ${pIdx});" title="Open ${p.title} deep-dive cockpit">
                    <span class="preview-proj-bullet">→</span>
                    <span>${p.title}</span>
                  </li>
                `).join('')}
              </ul>
            </div>
          </div>

          <div class="subject-action-row">
            <span class="dept-stat-pill">200 Total Blueprints</span>
            <div class="btn-view-10-projects">
              <span>Access ${projs.length} Projects</span> <span>→</span>
            </div>
          </div>
        `;

        card.addEventListener('click', () => openSubjectProjects(deptKey));
        grid.appendChild(card);
      });
    }

    function renderTenProjects(deptKey) {
      const dept = projectsData[deptKey];
      const meta = deptMetadata[deptKey] || {};
      if (!dept) return;

      document.getElementById('breadcrumbDeptName').textContent = dept.name;

      document.getElementById('deptBannerIcon').textContent = meta.icon || '⚡';
      document.getElementById('deptBannerTitle').textContent = dept.name;
      document.getElementById('deptBannerDesc').textContent = dept.description;

      const banner = document.getElementById('deptBanner');
      banner.style.setProperty('--dept-color', meta.color || '#3b82f6');
      banner.style.setProperty('--dept-bg', meta.bg || 'rgba(59, 130, 246, 0.2)');
      banner.style.setProperty('--dept-glow', meta.glow || 'rgba(59, 130, 246, 0.45)');

      const grid = document.getElementById('tenProjectsGrid');
      grid.innerHTML = '';

      const projs = dept.projects || [];
      document.getElementById('projectCountDisplay').textContent = projs.length;
      const totalDisplay = document.getElementById('projectTotalDisplay');
      if (totalDisplay) totalDisplay.textContent = projs.length;

      projs.forEach((p, index) => {
        const card = document.createElement('div');
        card.className = 'project-card-interactive';
        card.style.setProperty('--dept-color', meta.color || '#3b82f6');
        card.style.setProperty('--dept-glow', meta.glow || 'rgba(59, 130, 246, 0.45)');

        const metricSnippet = p.performance_metrics ? p.performance_metrics.split(';')[0] : '';

        card.innerHTML = `
          <div class="project-index-pill">#${String(index + 1).padStart(2, '0')}</div>
          
          <div class="project-card-top-matter">
            <span class="project-category-lbl">${meta.shortName}</span>
            <h3 class="project-card-heading">${p.title}</h3>
            <p class="project-card-summary">${p.description}</p>
          </div>

          <div>
            <div class="tech-stack-tag-row">
              ${(p.tech_stack || []).map(t => `<span class="tech-tag-pill">${t}</span>`).join('')}
            </div>

            ${metricSnippet ? `
              <div class="metric-badge-strip metric-badge-strip-value-only">
                <span class="metric-badge-val">${metricSnippet}</span>
              </div>
            ` : ''}

            <div class="btn-project-deepdive">
              <span>Deep-Dive Blueprint</span>
              <span>→</span>
            </div>
          </div>
        `;

        card.addEventListener('click', () => openProjectDetail(deptKey, index));
        grid.appendChild(card);
      });

      const searchInput = document.getElementById('projectSearchInput');
      searchInput.value = '';
      searchInput.oninput = function() {
        const q = this.value.toLowerCase().trim();
        const cards = grid.getElementsByClassName('project-card-interactive');
        let visible = 0;
        Array.from(cards).forEach((c, idx) => {
          const proj = projs[idx];
          const text = (proj.title + ' ' + proj.description + ' ' + (proj.tech_stack || []).join(' ')).toLowerCase();
          if (text.includes(q)) {
            c.style.display = 'flex';
            visible++;
          } else {
            c.style.display = 'none';
          }
        });
        document.getElementById('projectCountDisplay').textContent = visible;
      };
    }

    function renderProjectCockpit(deptKey, projIndex) {
      const dept = projectsData[deptKey];
      const meta = deptMetadata[deptKey] || {};
      if (!dept) return;
      const p = (dept.projects || [])[projIndex];
      if (!p) return;

      currentDeptKey = deptKey;
      currentProjIndex = projIndex;

      document.getElementById('breadcrumbDetailDept').textContent = dept.name;
      document.getElementById('breadcrumbDetailProj').textContent = p.title;

      const hero = document.getElementById('cockpitHeroBanner');
      hero.style.setProperty('--dept-color', meta.color || '#3b82f6');
      hero.style.setProperty('--dept-bg', meta.bg || 'rgba(59, 130, 246, 0.2)');
      document.getElementById('viewProjectDetail').style.setProperty('--dept-color', meta.color || '#3b82f6');

      document.getElementById('cockpitDeptBadge').textContent = meta.name;
      document.getElementById('cockpitIndexBadge').textContent = `BLUEPRINT #${String(projIndex + 1).padStart(2, '0')} OF ${dept.projects.length}`;
      document.getElementById('cockpitTitle').textContent = p.title;
      document.getElementById('cockpitSummary').textContent = p.description;
      document.title = `${p.title} | AI Connectra Capstone Lab`;
      document.getElementById('heroVisualTitle').textContent = `Illustrative ${p.title} response`;
      document.getElementById('heroVisualCaption').textContent = `Conceptual model-versus-baseline evaluation for this ${meta.shortName || 'capstone'} project.`;

      document.getElementById('projectFactLevel').textContent = p.difficulty || 'Intermediate';
      document.getElementById('projectFactDuration').textContent = p.duration || '8–10 weeks';
      document.getElementById('projectFactFormat').textContent = p.format || 'Guided capstone';
      document.getElementById('projectFactWorkspace').textContent = 'LMS + Google Colab';
      document.getElementById('projectAudience').textContent = p.audience;
      document.getElementById('projectPrerequisites').textContent = p.prerequisites;
      document.getElementById('projectBuildOutcome').textContent = p.business_impact || p.impact || `A working ${p.title} prototype.`;
      document.getElementById('projectAiMethod').textContent = p.architecture || p.ai_technique || 'A documented AI model and baseline.';
      document.getElementById('projectSuccessMeasure').textContent = p.performance_metrics || 'Compare the solution against a simple baseline using task-appropriate quality, latency, and robustness metrics.';

      const techRow = document.getElementById('cockpitTechStack');
      techRow.innerHTML = (p.tech_stack || []).map(t => `<span class="tech-tag-pill" style="font-size: 0.82rem; padding: 6px 14px; background: rgba(59, 130, 246, 0.15); border-color: rgba(59, 130, 246, 0.4); color: #60a5fa;">${t}</span>`).join('');

      document.getElementById('cockpitChallenge').textContent = p.challenge || p.description;
      document.getElementById('cockpitArchitecture').textContent = p.architecture || 'Detailed neural network pipeline.';

      const pipelineSteps = document.getElementById('cockpitPipelineSteps');
      pipelineSteps.innerHTML = '';
      const pipelineStepsForProject = [
        { title: 'Source & validate data', desc: p.data_pipeline || `Collect and document representative data for ${p.title}.` },
        { title: 'Prepare the learning set', desc: `Clean, label where needed, split, and version the data so the ${p.title} baseline can be reproduced.` },
        { title: 'Train & compare models', desc: `Implement ${p.architecture || p.ai_technique || 'the selected AI approach'} and compare it with a transparent baseline.` },
        { title: 'Evaluate & demonstrate', desc: `${p.performance_metrics || 'Measure quality, latency, and robustness.'} Deliver the result through ${p.hardware_target || 'a documented demo environment'}.` }
      ];
      pipelineStepsForProject.forEach((s, idx) => {
        pipelineSteps.innerHTML += `
          <div class="pipeline-step-box">
            <span class="step-counter-tag">STAGE 0${idx + 1}</span>
            <div class="step-title-text">${s.title}</div>
            <div class="step-desc-text">${s.desc}</div>
          </div>
        `;
      });

      document.getElementById('cockpitMetrics').textContent = p.performance_metrics || '99.2% Accuracy, F1-Score: 0.96';
      document.getElementById('cockpitDatasets').textContent = p.data_pipeline || 'Curated benchmark repository with train/validation/test split.';

      const rawCode = p.code_snippet || '# Code architecture snippet\nimport torch\nimport torch.nn as nn\n\nprint("AI Connectra Architecture Ready")';
      document.getElementById('cockpitCodeBlock').textContent = rawCode;

      document.getElementById('cockpitImpact').textContent = p.business_impact || 'Substantial reduction in downtime and 10x throughput enhancement.';

      const workflow = [
        { title: 'Define the problem', copy: `Set the users, decision, scope, constraints, and success criteria for ${p.title}.`, outcome: 'A concise, testable project specification.' },
        { title: 'Prepare the data', copy: p.data_pipeline, outcome: 'A documented, versioned train/validation/test dataset.' },
        { title: 'Build a baseline', copy: 'Implement a simple rule-based or classical model before the advanced AI approach.', outcome: 'A fair baseline and repeatable evaluation harness.' },
        { title: 'Train the AI model', copy: `Implement and tune ${p.architecture || p.ai_technique} while logging experiments and assumptions.`, outcome: 'A saved model, configuration, and training evidence.' },
        { title: 'Evaluate & stress-test', copy: p.performance_metrics, outcome: 'Held-out results, error analysis, and limitations.' },
        { title: 'Demonstrate & report', copy: 'Build a small dashboard or API and explain the method, results, ethics, and next steps.', outcome: 'A reproducible demo, report, and presentation.' }
      ];
      const workflowHost = document.getElementById('projectWorkflowSteps');
      const workflowDetail = document.getElementById('projectWorkflowDetail');
      const showWorkflowStep = (selected) => {
        const step = workflow[selected];
        workflowDetail.innerHTML = `<strong>${step.title}</strong><p>${step.copy}</p><span>Your outcome: ${step.outcome}</span>`;
        workflowHost.querySelectorAll('button').forEach((button, index) => {
          button.setAttribute('aria-pressed', index === selected ? 'true' : 'false');
        });
      };
      workflowHost.innerHTML = workflow.map((step, index) => `
        <button type="button" aria-pressed="${index === 0}" data-workflow-step="${index}">
          <span>${String(index + 1).padStart(2, '0')}</span><strong>${step.title}</strong>
        </button>`).join('');
      workflowHost.querySelectorAll('button').forEach((button) => {
        button.addEventListener('click', () => showWorkflowStep(Number(button.dataset.workflowStep)));
      });
      showWorkflowStep(0);

      const deliverables = [
        `Problem definition, assumptions, and ${p.title} baseline`,
        'Clean dataset with preprocessing and reproducibility notes',
        'Training notebook, saved configuration, and evaluation plots',
        'Working demo, technical report, presentation, and limitations statement'
      ];
      document.getElementById('projectDeliverables').innerHTML = deliverables
        .map(item => `<li><span>✓</span>${item}</li>`).join('');
      document.getElementById('projectDataset').textContent = p.data_pipeline;
      document.getElementById('projectHardware').textContent = p.hardware_target;

      const roadmap = [
        ['Week 1', 'Problem, objectives, scope, and success criteria'],
        ['Week 2', 'Data sourcing, permissions, and baseline setup'],
        ['Week 3', `Build ${p.architecture || p.ai_technique || 'the AI model'} prototype`],
        ['Week 4', 'Train, tune, and record experiments'],
        ['Week 5', 'Held-out testing, error analysis, and limitations'],
        ['Week 6', 'Demo, report, presentation, and reproducibility check']
      ];
      document.getElementById('projectRoadmap').innerHTML = roadmap.map(([week, task]) =>
        `<div class="roadmap-item"><strong>${week}</strong><span>${task}</span></div>`).join('');

      const rubric = [
        ['Problem definition & background', 'Clear objective, scope, relevant sources, and safety constraints', '10%'],
        ['Data & baseline', 'Documented dataset, preprocessing, and a working baseline', '20%'],
        ['AI method & reproducibility', `Saved configuration, runnable notebook, and ${p.architecture || p.ai_technique || 'model'} rationale`, '20%'],
        ['Testing & results', 'Held-out comparison, metrics, plots, and error analysis', '25%'],
        ['Demonstration & communication', 'Understandable demo, report, and presentation', '15%'],
        ['Limitations & responsible use', 'Safety boundary, assumptions, and next steps', '10%']
      ];
      document.getElementById('projectRubric').innerHTML = rubric.map(row =>
        `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('');

      const guideSteps = [
        'Specify the problem and acceptance criteria',
        'Build and verify a baseline',
        'Design the data and model pipeline',
        'Train the selected AI approach',
        'Evaluate on held-out scenarios',
        'Explain and demonstrate the system'
      ];
      document.getElementById('projectGuideSteps').innerHTML = guideSteps.map(step => `<li>${step}</li>`).join('');
      document.getElementById('projectSubmissionItems').innerHTML = deliverables.map(item => `<li>${item}</li>`).join('');
      document.getElementById('projectToolTags').innerHTML = (p.tech_stack || []).map(tool => `<span>${tool}</span>`).join('');

      const projs = dept.projects || [];
      const prevIdx = (projIndex - 1 + projs.length) % projs.length;
      const nextIdx = (projIndex + 1) % projs.length;
      const prevBtn = document.getElementById('btnPrevProject');
      const nextBtn = document.getElementById('btnNextProject');
      if (prevBtn && projs[prevIdx]) {
        prevBtn.innerHTML = `<span>←</span> <span>Previous: #${String(prevIdx + 1).padStart(2, '0')} ${projs[prevIdx].title.substring(0, 24)}...</span>`;
      }
      if (nextBtn && projs[nextIdx]) {
        nextBtn.innerHTML = `<span>Next: #${String(nextIdx + 1).padStart(2, '0')} ${projs[nextIdx].title.substring(0, 24)}...</span> <span>→</span>`;
      }
    }

    function copyCodeSnippet() {
      const code = document.getElementById('cockpitCodeBlock').textContent;
      navigator.clipboard.writeText(code).then(() => {
        const btn = document.getElementById('btnCopyCode');
        const oldHtml = btn.innerHTML;
        btn.innerHTML = '<span>✅ Copied!</span>';
        setTimeout(() => { btn.innerHTML = oldHtml; }, 2000);
      }).catch(err => {
        console.error('Failed to copy code', err);
      });
    }

    // ====================================================================
    // THEME TOGGLE
    // ====================================================================
    function toggleTheme() {
      const html = document.documentElement;
      const btn = document.getElementById('themeToggleBtn');
      if (html.classList.contains('dark')) {
        html.classList.remove('dark');
        html.classList.add('light');
        if (btn) btn.innerHTML = '<span>🌙</span>';
      } else {
        html.classList.remove('light');
        html.classList.add('dark');
        if (btn) btn.innerHTML = '<span>☀️</span>';
      }
    }

    // ====================================================================
    // HASH ROUTER (NEVER ACCIDENTALLY NAVIGATES TO HERO)
    // ====================================================================
    function parseHash() {
      const hash = window.location.hash || '#subjects';
      if (hash === '#compare-tracks' || hash === '#pricing' || hash === '#tiers') {
        openCompareTracksView();
      } else if (hash.startsWith('#discipline-relevance')) {
        const parts = hash.split('-');
        const deptKey = parts.length > 2 ? parts[2] : 'eee';
        openDisciplineRelevanceView(deptKey);
      } else if (hash.startsWith('#course-')) {
        const courseId = hash.replace('#course-', '');
        openCourseView(courseId);
      } else if (hash.startsWith('#proj-')) {
        const parts = hash.replace('#proj-', '').split('-');
        const deptKey = parts[0];
        const projIdx = parseInt(parts[1], 10) || 0;
        openProjectDetail(deptKey, projIdx);
      } else if (hash.startsWith('#dept-')) {
        const deptKey = hash.replace('#dept-', '');
        openSubjectProjects(deptKey);
      } else if (hash === '#subjects') {
        navigateToSubjects();
      } else {
        navigateToSubjects();
      }
    }

    window.addEventListener('hashchange', parseHash);

    
    // ====================================================================
    // HERO INTERACTIVE DISCIPLINE RADAR (WHY EVERY MAJOR MUST LEARN AI)
    // ====================================================================
    function switchHeroDiscipline(deptKey) {
      const data = heroDisciplineWhyAI[deptKey] || heroDisciplineWhyAI['eee'];

      // Update tabs active state in Hero
      const tabs = document.querySelectorAll('#heroRadarTabs .hero-radar-tab-btn');
      tabs.forEach(btn => {
        if (btn.getAttribute('onclick') && btn.getAttribute('onclick').includes(`'${deptKey}'`)) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });

      const container = document.getElementById('heroRadarDisplayCard');
      if (!container) return;

      container.innerHTML = `
        <div class="radar-card-top">
          <div class="radar-card-title-group">
            <div class="radar-card-icon">${data.icon}</div>
            <div>
              <div class="radar-card-name">${data.name}</div>
              <div class="radar-card-tagline">✦ ${data.tagline} ✦</div>
            </div>
          </div>
          <button class="btn-radar-cta" onclick="openDisciplineRelevanceView('${data.deptKey}')">
            <span>🧭 Explore Full ${data.name.split(' ')[0]} AI Deep-Dive</span>
            <span>→</span>
          </button>
        </div>

        <div class="radar-card-grid">
          <div class="radar-card-section">
            <div class="radar-section-label label-limit">
              <span>⚠️</span> <span>Where Traditional Formulas Fail in ${data.name.split(' ')[0]}</span>
            </div>
            <div class="radar-section-body">
              ${data.classicalLimit}
            </div>
          </div>

          <div class="radar-card-section">
            <div class="radar-section-label label-breakthrough">
              <span>✨</span> <span>The AI & Machine Learning Breakthrough</span>
            </div>
            <div class="radar-section-body">
              ${data.aiBreakthrough}
            </div>
          </div>
        </div>

        <div class="radar-card-section" style="margin-bottom: 20px;">
          <div class="radar-section-label label-career">
            <span>💼</span> <span>Why Students in ${data.name.split(' ')[0]} Must Learn AI (Industry Career Impact)</span>
          </div>
          <div class="radar-section-body">
            ${data.careerWhy}
          </div>
        </div>

        <div class="radar-card-bottom">
          <div class="radar-blueprint-highlight">
            <span class="radar-blueprint-tag">FEATURED PRODUCTION BLUEPRINT</span>
            <span>${data.blueprint}</span>
          </div>
          <button class="btn-secondary-glass" style="padding: 6px 16px; font-size: 0.84rem;" onclick="openProjectDetail('${data.deptKey}', ${data.blueprintIndex})">
            <span>⚡ Launch Blueprint Cockpit</span>
            <span>↗</span>
          </button>
        </div>
      `;
    }


    window.addEventListener('DOMContentLoaded', () => {
      switchHeroDiscipline('eee');
      initBlueParticles();
      renderSubjectsCatalog();
      switchEssentialDept('eee');
      parseHash();
    });
