/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const { JSDOM } = require("jsdom");

const html = fs.readFileSync(process.argv[2], "utf8");
let pageX = 17;
let pageY = 311;
const dom = new JSDOM(html, {
  pretendToBeVisual: true,
  runScripts: "dangerously",
  beforeParse(window) {
    Object.defineProperty(window, "scrollX", { get: () => pageX });
    Object.defineProperty(window, "scrollY", { get: () => pageY });
    window.scrollTo = (x, y) => {
      pageX = x;
      pageY = y;
    };
    Object.defineProperty(window.HTMLElement.prototype, "clientHeight", {
      get() {
        return this.classList.contains("rows") ? 120 : 20;
      },
    });
    window.HTMLElement.prototype.getBoundingClientRect = function () {
      if (this.classList.contains("rows")) {
        return { top: 0, height: 120 };
      }
      if (this.classList.contains("evidence-row")) {
        const rows = Array.from(
          this.closest(".rows").querySelectorAll(".evidence-row")
        );
        return { top: 180 + rows.indexOf(this) * 60, height: 20 };
      }
      return { top: 0, height: 20 };
    };
  },
});
const { document, Event } = dom.window;

assert.equal(document.querySelectorAll("select").length, 2);
assert.equal(document.querySelectorAll(".panel").length, 6);
assert.equal(document.querySelectorAll(".panel-description").length, 6);
assert.equal(document.querySelectorAll(".row-explanation").length, 0);
assert.ok(
  Array.from(document.querySelectorAll(".evidence-row .badge")).every(
    (badge) => badge.textContent.trim().length > 0
  )
);
assert.equal(document.querySelectorAll(".panel-header button").length, 0);
assert.equal(document.querySelectorAll(".empty-state[tabindex]").length, 0);
assert.equal(document.body.textContent.includes("Show complete attribution"), false);
assert.equal(document.body.textContent.includes("typed rewrite edges"), false);
assert.equal(document.getElementById("run-summary").textContent.includes("0 unresolved"), false);

const opspec = document.querySelector('[data-panel-id="opspec"]');
const opspecRows = Array.from(opspec.querySelectorAll(".evidence-row"));
assert.equal(opspecRows.length, 2);
assert.ok(
  opspecRows.every((row) => row.textContent.includes("SpecPath"))
);
assert.equal(opspec.textContent.includes("Finalized bundle"), false);
assert.equal(opspec.textContent.includes("compile candidates"), false);
assert.equal(opspec.textContent.includes("Compiler kernels"), false);
assert.equal(opspec.textContent.includes("Aliases:"), false);
assert.equal(document.getElementById("fact-candidates").textContent, "2");

const sourcePanel = document.querySelector('[data-panel-id="source"]');
const sourceBody = sourcePanel.querySelector(".rows");
const sourceRow = sourcePanel.querySelector(".evidence-row");
sourceBody.scrollTop = 47;
sourceBody.scrollLeft = 9;
sourceRow.focus();
sourceRow.click();

assert.equal(document.activeElement, sourceRow);
assert.equal(sourceBody.scrollTop, 47);
assert.equal(sourceBody.scrollLeft, 9);
assert.equal(pageX, 17);
assert.equal(pageY, 311);
assert.equal(sourceRow.classList.contains("is-focused"), true);
assert.equal(sourceRow.getAttribute("aria-pressed"), "true");
assert.equal(
  sourcePanel.querySelectorAll(".evidence-row.is-related").length,
  0
);
assert.ok(document.querySelectorAll(".evidence-row.is-related").length > 0);
assert.ok(document.querySelectorAll(".evidence-row.is-dimmed").length > 0);
assert.ok(
  Array.from(document.querySelectorAll(".panel"))
    .filter((panel) => panel !== sourcePanel)
    .some((panel) => panel.querySelector(".rows").scrollTop > 0)
);

sourceRow.click();
assert.equal(sourceRow.classList.contains("is-focused"), false);
assert.equal(sourceRow.getAttribute("aria-pressed"), "false");
assert.equal(document.querySelectorAll(".evidence-row.is-related").length, 0);
assert.equal(document.querySelectorAll(".evidence-row.is-dimmed").length, 0);

const eventSelect = document.getElementById("event-select");
eventSelect.value = "1";
eventSelect.dispatchEvent(new Event("change"));
assert.equal(document.querySelectorAll(".panel").length, 6);
assert.equal(document.querySelectorAll(".evidence-row.is-focused").length, 0);
assert.equal(
  dom.window.__spyreProvenanceViewer.state.focusedRowId,
  null
);
assert.equal(
  dom.window.__spyreProvenanceViewer.state.focusedPanelId,
  null
);

process.stdout.write("Spyre provenance viewer DOM check passed\n");
