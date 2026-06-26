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
  var activeCy = null;
  var activeView = null;

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

    // Click handler
    activeCy.on("tap", "node", function (evt) {
      activeCy.elements().removeClass("selected-node neighbor");
      var node = evt.target;
      node.addClass("selected-node");
      node.neighborhood("node").addClass("neighbor");
      showNodeInfo(node.data());
    });

    activeCy.on("tap", function (evt) {
      if (evt.target === activeCy) {
        activeCy.elements().removeClass("selected-node neighbor");
        clearInfo();
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
  }

  // -------------------------------------------------------------------------
  // UI: info panel
  // -------------------------------------------------------------------------

  function showNodeInfo(data) {
    var info = document.getElementById("kg-info");
    var meta = TYPE_META[data.type] || { label: data.type, color: "#bdc3c7" };
    var html =
      '<span style="display:inline-block;width:12px;height:12px;' +
      "background:" +
      meta.color +
      ';border-radius:2px;vertical-align:middle;margin-right:6px;"></span>';
    html += "<strong>" + data.label + "</strong>";
    html += " <em>(" + meta.label + ")</em>";
    if (data.source_file) {
      html += "<br>File: <code>" + data.source_file + "</code>";
      if (data.line) {
        html += " line " + data.line;
      }
    }
    var neighbors = activeCy
      .getElementById(data.id)
      .neighborhood("node");
    if (neighbors.length > 0) {
      html += "<br><strong>Connected to:</strong> ";
      var items = neighbors
        .map(function (n) {
          var nm = TYPE_META[n.data("type")] || { color: "#bdc3c7" };
          return (
            '<span style="border-bottom:2px solid ' +
            nm.color +
            '">' +
            n.data("label") +
            "</span>"
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
      "<em>Click a node to see its source location and connections.</em>";
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
    buildTabs();
    setupSearch();

    // Activate first tab
    var tabBar = document.getElementById("kg-tabs");
    if (tabBar && tabBar.firstChild) {
      tabBar.firstChild.classList.add("active");
    }
    renderView("ops");
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
