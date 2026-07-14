/**
 * Knowledge Graph Explorer for Torch-Spyre documentation.
 * Multi-view tabbed interface with per-view layouts and filtering.
 */
(function () {
  "use strict";

  // -------------------------------------------------------------------------
  // View definitions
  // -------------------------------------------------------------------------

  var VIEWS = {
    ops: {
      label: "Operations",
      description:
        "Each PyTorch op and its Spyre implementation path: " +
        "decompositions, lowerings, custom ops, fallbacks, and eager kernels.",
      types: [
        "op",
        "decomposition",
        "lowering",
        "custom_op",
        "fallback",
        "eager_kernel",
      ],
      relationships: [
        "decomposed_by",
        "lowered_by",
        "falls_back_to",
        "eager_via",
      ],
      layout: {
        name: "cose",
        animate: false,
        nodeRepulsion: function () {
          return 6000;
        },
        idealEdgeLength: function () {
          return 70;
        },
        nodeOverlap: 20,
        gravity: 0.4,
        numIter: 1500,
      },
    },
    compiler: {
      label: "Compiler Passes",
      description:
        "Pass groups and their constituent functions that transform " +
        "the graph during compilation.",
      types: ["pass_group", "pass_function"],
      relationships: ["contains_pass"],
      layout: {
        name: "breadthfirst",
        animate: false,
        directed: true,
        spacingFactor: 1.5,
        avoidOverlap: true,
        roots: function (nodes) {
          return nodes
            .filter(function (n) {
              return n.data("type") === "pass_group";
            })
            .map(function (n) {
              return n.id();
            });
        },
      },
    },
    architecture: {
      label: "Architecture",
      description:
        "Module dependencies, class hierarchies, and dataclass " +
        "definitions across the torch_spyre package.",
      types: ["module", "class", "dataclass"],
      relationships: ["imports", "inherits_from", "contains_field"],
      layout: {
        name: "cose",
        animate: false,
        nodeRepulsion: function () {
          return 10000;
        },
        idealEdgeLength: function () {
          return 120;
        },
        nodeOverlap: 30,
        gravity: 0.2,
        numIter: 2000,
        nestingFactor: 1.2,
      },
    },
    config: {
      label: "Configuration",
      description:
        "Environment variables and the modules that read them to " +
        "control runtime and compilation behavior.",
      types: ["env_var", "module"],
      relationships: ["reads_env"],
      layout: {
        name: "cose",
        animate: false,
        nodeRepulsion: function () {
          return 4000;
        },
        idealEdgeLength: function () {
          return 90;
        },
        nodeOverlap: 15,
        gravity: 0.5,
        numIter: 1000,
      },
    },
  };

  // -------------------------------------------------------------------------
  // Color and label maps
  // -------------------------------------------------------------------------

  var TYPE_META = {
    op: { color: "#4a90d9", label: "Operations" },
    decomposition: { color: "#2ecc71", label: "Decompositions" },
    lowering: { color: "#27ae60", label: "Lowerings" },
    custom_op: { color: "#e67e22", label: "Custom Ops" },
    fallback: { color: "#e74c3c", label: "CPU Fallbacks" },
    eager_kernel: { color: "#9b59b6", label: "Eager Kernels" },
    pass_group: { color: "#7f8c8d", label: "Pass Groups" },
    pass_function: { color: "#95a5a6", label: "Pass Functions" },
    module: { color: "#f39c12", label: "Modules" },
    class: { color: "#8e44ad", label: "Classes" },
    dataclass: { color: "#d35400", label: "Dataclasses" },
    env_var: { color: "#16a085", label: "Env Variables" },
  };

  // -------------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------------

  var graphData = null;
  var graphMeta = {};
  var activeCy = null;
  var activeView = null;
  var focusMode = false;

  // -------------------------------------------------------------------------
  // Build a "view source" URL back to the code that defines a node.
  //
  // Node source_file paths are repo-relative (e.g.
  // "torch_spyre/_inductor/lowering.py"). We pin to the exact commit the
  // graph was extracted from so a link never rots against a moved line;
  // if the commit is unknown (graph built outside a git checkout), fall
  // back to the default branch.
  // -------------------------------------------------------------------------

  function buildSourceUrl(sourceFile, line) {
    if (!sourceFile) return null;
    var repo = graphMeta.repo_url || "https://github.com/torch-spyre/torch-spyre";
    var commit = graphMeta.source_commit;
    var ref = commit && commit !== "unknown" ? commit : graphMeta.default_branch || "main";
    var url = repo + "/blob/" + ref + "/" + sourceFile;
    if (line) url += "#L" + line;
    return url;
  }

  // -------------------------------------------------------------------------
  // Cytoscape style shared across views
  // -------------------------------------------------------------------------

  function getStyles() {
    return [
      {
        selector: "node",
        style: {
          label: "data(label)",
          "font-size": "10px",
          "text-valign": "bottom",
          "text-halign": "center",
          "text-margin-y": 3,
          "background-color": function (ele) {
            var meta = TYPE_META[ele.data("type")];
            return meta ? meta.color : "#bdc3c7";
          },
          width: function (ele) {
            var t = ele.data("type");
            if (t === "pass_group" || t === "module") return 24;
            if (t === "class" || t === "dataclass" || t === "op") return 18;
            return 14;
          },
          height: function (ele) {
            var t = ele.data("type");
            if (t === "pass_group" || t === "module") return 24;
            if (t === "class" || t === "dataclass" || t === "op") return 18;
            return 14;
          },
          "border-width": 1.5,
          "border-color": "#444",
        },
      },
      {
        selector: "edge",
        style: {
          width: 1.5,
          "line-color": "#ccc",
          "target-arrow-color": "#aaa",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          "arrow-scale": 0.8,
        },
      },
      {
        selector: "edge[relationship = 'falls_back_to']",
        style: {
          "line-style": "dashed",
          "line-color": "#e74c3c",
          "target-arrow-color": "#e74c3c",
        },
      },
      {
        selector: "edge[relationship = 'inherits_from']",
        style: {
          "line-style": "dotted",
          "line-color": "#8e44ad",
          "target-arrow-color": "#8e44ad",
          "target-arrow-shape": "diamond",
        },
      },
      {
        selector: "edge[relationship = 'imports']",
        style: {
          "line-color": "#f39c12",
          "target-arrow-color": "#f39c12",
          width: 1,
          opacity: 0.5,
        },
      },
      {
        selector: "edge[relationship = 'reads_env']",
        style: {
          "line-color": "#16a085",
          "target-arrow-color": "#16a085",
          width: 1.2,
        },
      },
      {
        selector: "edge[relationship = 'contains_pass']",
        style: {
          "line-color": "#7f8c8d",
          "target-arrow-color": "#7f8c8d",
          width: 1.5,
        },
      },
      {
        selector: "node.highlighted",
        style: {
          "border-width": 3,
          "border-color": "#f1c40f",
          "font-weight": "bold",
          "font-size": "12px",
        },
      },
      {
        selector: "node.faded",
        style: { opacity: 0.15 },
      },
      {
        selector: "edge.faded",
        style: { opacity: 0.05 },
      },
      {
        selector: "node.selected-node",
        style: {
          "border-width": 4,
          "border-color": "#e74c3c",
          "font-weight": "bold",
        },
      },
      {
        selector: "node.neighbor",
        style: {
          "border-width": 2,
          "border-color": "#f39c12",
        },
      },
    ];
  }

  // -------------------------------------------------------------------------
  // Filter graph data to a specific view
  // -------------------------------------------------------------------------

  function filterForView(viewKey) {
    var view = VIEWS[viewKey];
    var typeSet = {};
    view.types.forEach(function (t) {
      typeSet[t] = true;
    });
    var relSet = {};
    view.relationships.forEach(function (r) {
      relSet[r] = true;
    });

    var nodeIds = {};
    var nodes = graphData.nodes.filter(function (n) {
      if (typeSet[n.type]) {
        nodeIds[n.id] = true;
        return true;
      }
      return false;
    });

    // For config view, only include modules that have a reads_env edge
    if (viewKey === "config") {
      var modulesWithEnv = {};
      graphData.edges.forEach(function (e) {
        if (e.relationship === "reads_env") {
          modulesWithEnv[e.source] = true;
        }
      });
      nodes = nodes.filter(function (n) {
        if (n.type === "module") return modulesWithEnv[n.id];
        return true;
      });
      nodeIds = {};
      nodes.forEach(function (n) {
        nodeIds[n.id] = true;
      });
    }

    var edges = graphData.edges.filter(function (e) {
      return relSet[e.relationship] && nodeIds[e.source] && nodeIds[e.target];
    });

    return { nodes: nodes, edges: edges };
  }

  // -------------------------------------------------------------------------
  // Render a view
  // -------------------------------------------------------------------------

  function renderView(viewKey) {
    if (activeView === viewKey && activeCy) return;
    activeView = viewKey;

    // A new view has no selection; drop any lingering focus state.
    focusMode = false;
    var focusBtn = document.getElementById("kg-focus");
    if (focusBtn) focusBtn.classList.remove("active");

    var view = VIEWS[viewKey];
    var filtered = filterForView(viewKey);

    var elements = [];
    filtered.nodes.forEach(function (n) {
      elements.push({
        group: "nodes",
        data: {
          id: n.id,
          label: n.label,
          type: n.type,
          source_file: n.source_file || "",
          line: n.line || 0,
        },
      });
    });
    filtered.edges.forEach(function (e) {
      elements.push({
        group: "edges",
        data: {
          id: e.source + "->" + e.target + "->" + e.relationship,
          source: e.source,
          target: e.target,
          relationship: e.relationship,
        },
      });
    });

    if (activeCy) {
      activeCy.destroy();
      activeCy = null;
    }

    var container = document.getElementById("cy");
    container.innerHTML = "";

    var layoutOpts = Object.assign({}, view.layout);

    // breadthfirst roots need elements loaded first
    var rootsFn = layoutOpts.roots;
    delete layoutOpts.roots;

    activeCy = cytoscape({
      container: container,
      elements: elements,
      style: getStyles(),
      layout: { name: "preset" },
      minZoom: 0.15,
      maxZoom: 5,
    });

    if (rootsFn) {
      layoutOpts.roots = rootsFn(activeCy.nodes());
    }

    activeCy.layout(layoutOpts).run();

    // Single click: select the node (highlight + info panel).
    activeCy.on("tap", "node", function (evt) {
      selectNode(evt.target.id());
    });

    // Double click: jump straight to the defining source on GitHub.
    activeCy.on("dbltap", "node", function (evt) {
      var d = evt.target.data();
      var url = buildSourceUrl(d.source_file, d.line);
      if (url) window.open(url, "_blank", "noopener");
    });

    // Background click: clear selection, focus, and the deep link.
    activeCy.on("tap", function (evt) {
      if (evt.target === activeCy) {
        clearSelection();
      }
    });

    // Update legend
    buildLegend(viewKey);

    // Update stats
    var stats = document.getElementById("kg-stats");
    if (stats) {
      stats.textContent =
        filtered.nodes.length +
        " nodes, " +
        filtered.edges.length +
        " edges in this view";
    }

    // Update description
    var desc = document.getElementById("kg-view-desc");
    if (desc) {
      desc.textContent = view.description;
    }

    // Clear search
    var search = document.getElementById("kg-search");
    if (search) search.value = "";

    // Reflect the active view in the URL (node cleared on view switch).
    updateHash(null);
  }

  // -------------------------------------------------------------------------
  // UI: info panel
  // -------------------------------------------------------------------------

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function showNodeInfo(data) {
    var info = document.getElementById("kg-info");
    var meta = TYPE_META[data.type] || { label: data.type, color: "#bdc3c7" };
    var html =
      '<span style="display:inline-block;width:12px;height:12px;' +
      "background:" +
      meta.color +
      ';border-radius:2px;vertical-align:middle;margin-right:6px;"></span>';
    html += "<strong>" + escapeHtml(data.label) + "</strong>";
    html += " <em>(" + escapeHtml(meta.label) + ")</em>";

    // Source location: a clickable link straight to the defining code.
    if (data.source_file) {
      var url = buildSourceUrl(data.source_file, data.line);
      var locText = escapeHtml(data.source_file);
      if (data.line) locText += ":" + data.line;
      if (url) {
        html +=
          '<br><a class="kg-source-link" href="' +
          escapeHtml(url) +
          '" target="_blank" rel="noopener noreferrer" ' +
          'title="Open this definition on GitHub">' +
          "&#128196; <code>" +
          locText +
          "</code> &#8599;</a>";
      } else {
        html += "<br>File: <code>" + locText + "</code>";
      }
    } else {
      html +=
        '<br><span style="color:#999;">No single source location ' +
        "(this node is referenced from several places).</span>";
    }

    var neighbors = activeCy.getElementById(data.id).neighborhood("node");
    if (neighbors.length > 0) {
      html += "<br><strong>Connected to:</strong> ";
      var items = neighbors
        .map(function (n) {
          var nm = TYPE_META[n.data("type")] || { color: "#bdc3c7" };
          return (
            '<a href="#" class="kg-neighbor" data-node-id="' +
            escapeHtml(n.id()) +
            '" style="border-bottom:2px solid ' +
            nm.color +
            ';text-decoration:none;color:inherit;" ' +
            'title="Focus this node">' +
            escapeHtml(n.data("label")) +
            "</a>"
          );
        })
        .slice(0, 15);
      html += items.join(", ");
      if (neighbors.length > 15) {
        html += " <em>+" + (neighbors.length - 15) + " more</em>";
      }
    }
    info.innerHTML = html;
  }

  function clearInfo() {
    document.getElementById("kg-info").innerHTML =
      "<em>Click a node to see its source location and connections. " +
      "Double-click a node to jump straight to its code.</em>";
  }

  // Select a node by id: highlight it and its neighbors, center the
  // viewport on it, and refresh the info panel. Shared by tap handlers,
  // the "Connected to" neighbor links, and deep-link restoration.
  function selectNode(nodeId, opts) {
    if (!activeCy) return;
    var node = activeCy.getElementById(nodeId);
    if (!node || node.empty()) return;
    activeCy.elements().removeClass("selected-node neighbor");
    node.addClass("selected-node");
    node.neighborhood("node").addClass("neighbor");
    if (focusMode) applyFocus(node);
    if (!opts || opts.center !== false) {
      activeCy.animate({ center: { eles: node }, duration: 250 });
    }
    showNodeInfo(node.data());
    updateHash(nodeId);
  }

  // -------------------------------------------------------------------------
  // Focus mode, selection, and deep-linking
  // -------------------------------------------------------------------------

  // Fade everything except a node and its immediate neighborhood, so a
  // dense view collapses to just the relationships around one concept.
  function applyFocus(node) {
    activeCy.elements().addClass("faded");
    node.closedNeighborhood().removeClass("faded");
  }

  function clearFocus() {
    if (activeCy) activeCy.elements().removeClass("faded");
  }

  function toggleFocus() {
    focusMode = !focusMode;
    var btn = document.getElementById("kg-focus");
    if (btn) btn.classList.toggle("active", focusMode);
    var selected = activeCy ? activeCy.$(".selected-node") : null;
    if (focusMode && selected && selected.length) {
      applyFocus(selected);
    } else {
      clearFocus();
    }
  }

  function clearSelection() {
    if (activeCy) activeCy.elements().removeClass("selected-node neighbor");
    clearFocus();
    clearInfo();
    updateHash(null);
  }

  // Reflect the current view (and optionally a selected node) in the URL
  // hash so a specific node can be linked and shared, e.g. the page can be
  // opened at "#ops/op::mm" to land directly on that op.
  function updateHash(nodeId) {
    if (!activeView) return;
    var hash = "#" + activeView;
    if (nodeId) hash += "/" + encodeURIComponent(nodeId);
    if (typeof history !== "undefined" && history.replaceState) {
      history.replaceState(null, "", hash);
    } else {
      location.hash = hash;
    }
  }

  function parseHash() {
    var h = (location.hash || "").replace(/^#/, "");
    if (!h) return null;
    var slash = h.indexOf("/");
    if (slash === -1) return { view: h, nodeId: null };
    return {
      view: h.slice(0, slash),
      nodeId: decodeURIComponent(h.slice(slash + 1)),
    };
  }

  // -------------------------------------------------------------------------
  // UI: view controls (fit / focus / reset / export)
  // -------------------------------------------------------------------------

  function setupControls() {
    var fit = document.getElementById("kg-fit");
    if (fit) {
      fit.addEventListener("click", function () {
        if (activeCy) activeCy.animate({ fit: { padding: 30 }, duration: 250 });
      });
    }

    var focus = document.getElementById("kg-focus");
    if (focus) focus.addEventListener("click", toggleFocus);

    var reset = document.getElementById("kg-reset");
    if (reset) {
      reset.addEventListener("click", function () {
        focusMode = false;
        var fb = document.getElementById("kg-focus");
        if (fb) fb.classList.remove("active");
        var search = document.getElementById("kg-search");
        if (search) search.value = "";
        if (activeCy) {
          activeCy.elements().removeClass("highlighted faded");
          clearSelection();
          activeCy.animate({ fit: { padding: 30 }, duration: 250 });
        }
      });
    }

    var png = document.getElementById("kg-png");
    if (png) {
      png.addEventListener("click", function () {
        if (!activeCy) return;
        var uri = activeCy.png({ full: true, scale: 2, bg: "#ffffff" });
        var a = document.createElement("a");
        a.href = uri;
        a.download = "torch-spyre-" + (activeView || "graph") + ".png";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      });
    }

    // Escape clears the current selection.
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") clearSelection();
    });

    // Delegate clicks on the "Connected to" neighbor links so navigating
    // between related nodes stays inside the graph.
    var info = document.getElementById("kg-info");
    if (info) {
      info.addEventListener("click", function (e) {
        var link = e.target.closest ? e.target.closest(".kg-neighbor") : null;
        if (link && link.dataset && link.dataset.nodeId) {
          e.preventDefault();
          selectNode(link.dataset.nodeId);
        }
      });
    }
  }

  // -------------------------------------------------------------------------
  // UI: legend for current view
  // -------------------------------------------------------------------------

  function buildLegend(viewKey) {
    var container = document.getElementById("kg-legend");
    if (!container) return;
    container.innerHTML = "";
    var view = VIEWS[viewKey];
    view.types.forEach(function (type) {
      var meta = TYPE_META[type];
      if (!meta) return;
      var item = document.createElement("span");
      item.className = "kg-legend-item";
      item.innerHTML =
        '<span class="kg-legend-swatch" style="background:' +
        meta.color +
        '"></span>' +
        meta.label;
      container.appendChild(item);
    });
  }

  // -------------------------------------------------------------------------
  // UI: tabs
  // -------------------------------------------------------------------------

  function buildTabs() {
    var tabBar = document.getElementById("kg-tabs");
    if (!tabBar) return;
    tabBar.innerHTML = "";
    Object.keys(VIEWS).forEach(function (key) {
      var btn = document.createElement("button");
      btn.className = "kg-tab";
      btn.dataset.view = key;
      btn.textContent = VIEWS[key].label;
      btn.addEventListener("click", function () {
        tabBar.querySelectorAll(".kg-tab").forEach(function (b) {
          b.classList.remove("active");
        });
        btn.classList.add("active");
        renderView(key);
      });
      tabBar.appendChild(btn);
    });
  }

  // -------------------------------------------------------------------------
  // UI: search
  // -------------------------------------------------------------------------

  function setupSearch() {
    var input = document.getElementById("kg-search");
    if (!input) return;
    input.addEventListener("input", function () {
      if (!activeCy) return;
      var query = this.value.toLowerCase().trim();
      if (!query) {
        activeCy.elements().removeClass("highlighted faded");
        return;
      }
      activeCy.elements().addClass("faded");
      var matched = activeCy.nodes().filter(function (n) {
        return n.data("label").toLowerCase().indexOf(query) !== -1;
      });
      matched.removeClass("faded").addClass("highlighted");
      matched.connectedEdges().removeClass("faded");
      matched.neighborhood("node").removeClass("faded");
    });
  }

  // -------------------------------------------------------------------------
  // Bootstrap
  // -------------------------------------------------------------------------

  function init(data) {
    graphData = data;
    graphMeta = data.metadata || {};
    buildTabs();
    setupSearch();
    setupControls();

    // Restore the view (and node) from the URL hash if present, so shared
    // deep links like "#architecture/class::SpyreScheduler" land correctly.
    var parsed = parseHash();
    var initialView = parsed && VIEWS[parsed.view] ? parsed.view : "ops";

    var tabBar = document.getElementById("kg-tabs");
    if (tabBar) {
      tabBar.querySelectorAll(".kg-tab").forEach(function (b) {
        b.classList.toggle("active", b.dataset.view === initialView);
      });
    }
    renderView(initialView);

    if (parsed && parsed.nodeId) {
      // Defer until the layout has placed nodes for the freshly rendered view.
      setTimeout(function () {
        selectNode(parsed.nodeId);
      }, 0);
    }
  }

  // -------------------------------------------------------------------------
  // Load and boot
  // -------------------------------------------------------------------------

  var scriptEl = document.currentScript;

  function boot() {
    var basePath = scriptEl
      ? scriptEl.src.replace(/js\/knowledge_graph\.js.*/, "")
      : "../_static/";

    fetch(basePath + "js/graph.json")
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        try {
          init(data);
        } catch (e) {
          console.error("[knowledge_graph] init error:", e);
          var cy = document.getElementById("cy");
          if (cy) {
            cy.innerHTML =
              "<p style='padding:2em;color:#c0392b;'>Graph init error: " +
              e.message +
              "</p>";
          }
        }
      })
      .catch(function (err) {
        console.error("[knowledge_graph] fetch error:", err);
        var cy = document.getElementById("cy");
        if (cy) {
          cy.innerHTML =
            "<p style='padding:2em;color:#c0392b;'>Failed to load graph data: " +
            err.message +
            "</p>";
        }
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
