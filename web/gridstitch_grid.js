import { app } from "../../scripts/app.js";

// Interactive rows×cols picker for GridStitchAdvanced, drawn INSIDE the node.
//
// GRID mode (default): hover the 4×4 grid (fills from the BOTTOM-LEFT) to preview,
//   click to lock rows×cols. Each cell shows its image_i slot number (top-left origin,
//   row-major — matches how stitch() lays cells out). The node exposes rows×cols ports.
//
// MASK mode (advanced node only): flip the toggle and the SAME grid becomes a per-cell
//   mask picker — click the selected cells to mark which ones DENOISE (amber) vs stay
//   preserved (green). The selection is written to the hidden `mask_cells` string widget
//   (row-major, 1-indexed), which the backend reads to build the MASK output. Because it's
//   a real string param, an app can set it programmatically too.

const TARGETS = {
  GridStitchAdvanced: { dynamicInputs: true, mask: true },
  GridStitch: { dynamicInputs: false, mask: false },
};
const GRID = 4; // 4×4 max
const MAX = GRID * GRID; // 16
const CELL = 28;
const GAP = 5;
const HEIGHT = 232; // reserved widget height inside the node (grid + footer + toggle)

function ensureStyles() {
  if (document.getElementById("gs-picker-styles")) return;
  const s = document.createElement("style");
  s.id = "gs-picker-styles";
  s.textContent = `
    .gs-picker{box-sizing:border-box;height:${HEIGHT}px;display:flex;align-items:center;
      justify-content:center;padding:6px;user-select:none;}
    .gs-card{display:flex;flex-direction:column;align-items:center;gap:9px;padding:4px;
      background:transparent;border:none;box-shadow:none;}
    .gs-grid{display:grid;grid-template-columns:repeat(${GRID},${CELL}px);
      grid-template-rows:repeat(${GRID},${CELL}px);gap:${GAP}px;}
    .gs-cell{width:${CELL}px;height:${CELL}px;border-radius:0;background:#232833;
      border:1px solid #333a47;cursor:pointer;display:flex;align-items:center;justify-content:center;
      font:700 14px ui-sans-serif,system-ui,-apple-system,sans-serif;font-variant-numeric:tabular-nums;
      color:transparent;transition:background-color .1s ease,border-color .1s ease;}
    .gs-cell.preview{background:rgba(16,185,129,.20);border-color:#10b981;color:#d1fae5;}
    .gs-cell.on{background:linear-gradient(180deg,#10b981,#0ea372);border-color:#34d399;
      color:#ffffff;text-shadow:0 1px 1px rgba(0,0,0,.25);}
    .gs-cell.mask{background:linear-gradient(180deg,#f59e0b,#d97706);border-color:#fbbf24;
      color:#ffffff;text-shadow:0 1px 1px rgba(0,0,0,.25);}
    .gs-cell.dim{background:#1b1f27;border-color:#262b34;cursor:default;color:transparent;}
    .gs-foot{display:flex;align-items:baseline;gap:9px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
    .gs-dim{font-size:13px;font-weight:600;color:#e5e7eb;letter-spacing:.03em;}
    .gs-sep{width:1px;height:11px;background:#3a3f4a;align-self:center;}
    .gs-cnt{font-size:12px;color:#8b909b;}
    .gs-cnt b{color:#34d399;font-weight:600;}
    .gs-cnt.mask b{color:#fbbf24;}
    .gs-toggle{font:600 11px ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.04em;
      padding:3px 12px;border-radius:5px;cursor:pointer;background:#232833;color:#8b909b;
      border:1px solid #333a47;transition:all .1s ease;}
    .gs-toggle:hover{border-color:#4b5566;color:#c4c9d2;}
    .gs-toggle.active{background:linear-gradient(180deg,#f59e0b,#d97706);color:#fff;border-color:#fbbf24;}
  `;
  document.head.appendChild(s);
}

app.registerExtension({
  name: "GridStitch.InteractiveGrid",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const cfg = TARGETS[nodeData.name];
    if (!cfg) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      buildPicker(this, cfg);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      requestAnimationFrame(() => this._gsSync && this._gsSync());
      return r;
    };
  },
});

function parseCells(spec) {
  const out = new Set();
  (spec || "").split(",").forEach((p) => {
    p = p.trim();
    if (!p) return;
    if (p.includes("-")) {
      let [a, b] = p.split("-").map((x) => parseInt(x, 10));
      if (Number.isFinite(a) && Number.isFinite(b)) {
        if (a > b) [a, b] = [b, a];
        for (let k = a; k <= b; k++) out.add(k);
      }
    } else {
      const k = parseInt(p, 10);
      if (Number.isFinite(k)) out.add(k);
    }
  });
  return out;
}

function formatCells(set) {
  return [...set].sort((a, b) => a - b).join(",");
}

function buildPicker(node, cfg) {
  ensureStyles();
  const dynamicInputs = cfg.dynamicInputs;
  const rowsW = node.widgets && node.widgets.find((w) => w.name === "rows");
  const colsW = node.widgets && node.widgets.find((w) => w.name === "cols");
  const maskW = cfg.mask && node.widgets ? node.widgets.find((w) => w.name === "mask_cells") : null;
  if (!rowsW || !colsW || !node.addDOMWidget) return;

  // Keep rows/cols/mask_cells as the serialized backing values, but hide their boxes.
  const hide = (w) => { if (w) { w.hidden = true; w.computeSize = () => [0, -4]; } };
  hide(rowsW); hide(colsW); hide(maskW);

  node._gsMask = false;                                   // current interaction mode
  node._gsMaskSet = parseCells(maskW ? maskW.value : ""); // cells marked to denoise

  const wrap = document.createElement("div");
  wrap.className = "gs-picker";
  const card = document.createElement("div");
  card.className = "gs-card";
  const grid = document.createElement("div");
  grid.className = "gs-grid";
  const cells = [];
  for (let i = 0; i < MAX; i++) {
    const c = document.createElement("div");
    c.className = "gs-cell";
    c.dataset.i = String(i);
    grid.appendChild(c);
    cells.push(c);
  }
  const foot = document.createElement("div");
  foot.className = "gs-foot";
  card.appendChild(grid);
  card.appendChild(foot);

  let toggle = null;
  if (cfg.mask) {
    toggle = document.createElement("button");
    toggle.className = "gs-toggle";
    toggle.textContent = "▦ mask cells";
    toggle.addEventListener("click", () => {
      node._gsMask = !node._gsMask;
      toggle.classList.toggle("active", node._gsMask);
      toggle.textContent = node._gsMask ? "▦ mask: on" : "▦ mask cells";
      repaint(false);
    });
    card.appendChild(toggle);
  }
  wrap.appendChild(card);

  const colOf = (i) => i % GRID;
  const rowBotOf = (i) => GRID - 1 - Math.floor(i / GRID); // 0 = bottom row
  const numOf = (i, cols, rows) => (rows - 1 - rowBotOf(i)) * cols + colOf(i) + 1;
  const inBlock = (i, cols, rows) => colOf(i) < cols && rowBotOf(i) < rows;

  const current = () => ({
    cols: Math.min(GRID, Math.max(1, colsW.value | 0)),
    rows: Math.min(GRID, Math.max(1, rowsW.value | 0)),
  });

  // GRID-mode paint: highlight a rows×cols block from the bottom-left; slot numbers.
  function paintGrid(selCols, selRows, preview) {
    for (let i = 0; i < MAX; i++) {
      const on = inBlock(i, selCols, selRows);
      cells[i].className = "gs-cell" + (on ? (preview ? " preview" : " on") : "");
      cells[i].textContent = on ? String(numOf(i, selCols, selRows)) : "";
    }
    foot.innerHTML =
      `<span class="gs-dim">${selCols} × ${selRows}</span>` +
      `<span class="gs-sep"></span>` +
      `<span class="gs-cnt"><b>${selCols * selRows}</b> image${selCols * selRows > 1 ? "s" : ""}</span>`;
  }

  // MASK-mode paint: block cells are green (preserve) or amber (denoise); rest dimmed.
  function paintMask() {
    const { cols, rows } = current();
    for (let i = 0; i < MAX; i++) {
      if (!inBlock(i, cols, rows)) {
        cells[i].className = "gs-cell dim";
        cells[i].textContent = "";
        continue;
      }
      const num = numOf(i, cols, rows);
      cells[i].className = "gs-cell " + (node._gsMaskSet.has(num) ? "mask" : "on");
      cells[i].textContent = String(num);
    }
    const d = [...node._gsMaskSet].filter((k) => k <= cols * rows).length;
    foot.innerHTML =
      `<span class="gs-dim">denoise</span>` +
      `<span class="gs-sep"></span>` +
      `<span class="gs-cnt mask"><b>${d}</b> / ${cols * rows} cell${cols * rows > 1 ? "s" : ""}</span>`;
  }

  function repaint(preview) {
    if (node._gsMask) paintMask();
    else {
      const { cols, rows } = current();
      paintGrid(cols, rows, !!preview);
    }
  }

  // Drop any masked cell numbers that no longer exist at the current grid size.
  function pruneMask() {
    const { cols, rows } = current();
    const n = cols * rows;
    let changed = false;
    for (const k of [...node._gsMaskSet]) if (k > n) { node._gsMaskSet.delete(k); changed = true; }
    if (changed && maskW) maskW.value = formatCells(node._gsMaskSet);
  }

  grid.addEventListener("mousemove", (e) => {
    if (node._gsMask) return;                            // no dim-preview in mask mode
    const t = e.target.closest(".gs-cell");
    if (!t) return;
    const i = +t.dataset.i;
    paintGrid(colOf(i) + 1, rowBotOf(i) + 1, true);
  });
  grid.addEventListener("mouseleave", () => {
    if (!node._gsMask) repaint(false);
  });
  grid.addEventListener("click", (e) => {
    const t = e.target.closest(".gs-cell");
    if (!t) return;
    const i = +t.dataset.i;

    if (node._gsMask) {                                  // toggle this cell's denoise state
      const { cols, rows } = current();
      if (!inBlock(i, cols, rows)) return;
      const num = numOf(i, cols, rows);
      if (node._gsMaskSet.has(num)) node._gsMaskSet.delete(num);
      else node._gsMaskSet.add(num);
      if (maskW) maskW.value = formatCells(node._gsMaskSet);
      paintMask();
      return;
    }

    // grid mode: lock rows×cols
    const selCols = colOf(i) + 1;
    const selRows = rowBotOf(i) + 1;
    const changed = selCols !== (colsW.value | 0) || selRows !== (rowsW.value | 0);
    colsW.value = selCols;
    rowsW.value = selRows;
    if (dynamicInputs) applyInputs(node, selRows, selCols);
    if (changed) { node._gsMaskSet.clear(); if (maskW) maskW.value = ""; } // cell meaning changed -> reset
    else pruneMask();
    paintGrid(selCols, selRows, false);
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
  });

  node.addDOMWidget("gs_picker", "gs_picker", wrap, {
    serialize: false,
    getMinHeight: () => HEIGHT,
    getMaxHeight: () => HEIGHT,
  });

  node._gsSync = () => {
    const { cols, rows } = current();
    if (dynamicInputs) applyInputs(node, rows, cols);
    if (maskW) node._gsMaskSet = parseCells(maskW.value); // re-read after load
    pruneMask();
    if (node._gsRatioSync) node._gsRatioSync();
    repaint(false);
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
  };

  wireModeRatio(node);
  requestAnimationFrame(() => node._gsSync());
}

// Show the `ratio` dropdown only in manual mode.
function wireModeRatio(node) {
  const modeW = node.widgets && node.widgets.find((w) => w.name === "mode");
  const ratioW = node.widgets && node.widgets.find((w) => w.name === "ratio");
  if (!modeW || !ratioW) return;

  const update = () => {
    ratioW.hidden = modeW.value !== "manual";
    ratioW.computeSize = ratioW.hidden ? () => [0, -4] : undefined;
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
  };
  const prev = modeW.callback;
  modeW.callback = function () {
    const r = prev ? prev.apply(this, arguments) : undefined;
    update();
    return r;
  };
  node._gsRatioSync = update;
  update();
}

// Make node have exactly rows*cols contiguous image_i input ports.
function applyInputs(node, rows, cols) {
  const count = Math.min(Math.max(rows * cols, 1), MAX);

  for (let i = (node.inputs || []).length - 1; i >= 0; i--) {
    const inp = node.inputs[i];
    if (inp && inp.name && inp.name.startsWith("image_")) {
      if (parseInt(inp.name.slice(6), 10) > count) node.removeInput(i);
    }
  }
  const have = new Set(
    (node.inputs || [])
      .filter((x) => x.name && x.name.startsWith("image_"))
      .map((x) => x.name),
  );
  for (let k = 1; k <= count; k++) {
    if (!have.has(`image_${k}`)) node.addInput(`image_${k}`, "IMAGE");
  }

  node.setSize(node.computeSize());
  node.setDirtyCanvas(true, true);
}
