"use strict";

const $ = (id) => document.getElementById(id);
const STAGES = ["prepare", "shots", "speech", "visual_features", "scenes", "importance", "clips"];
const STAGE_LABELS = {
  prepare: "Prepare video (H.264 copy for playback)",
  shots: "Shot detection · TransNetV2",
  speech: "Speech recognition · Whisper",
  visual_features: "Visual features · CLIP",
  scenes: "Scene segmentation · visual + dialogue",
  importance: "Importance scoring · visual / audio / text",
  clips: "Clips, thumbnails & summary",
};

const API = (window.CLIPRANKER_API || "").replace(/\/$/, "");
const url = (path) => API + path;
// Showcase mode: no backend, pre-computed example runs served as static files.
const STATIC = window.CLIPRANKER_STATIC || null;
const sample = (id) => STATIC.samples.find((x) => x.id === id);
const media = (id, path) => (STATIC ? sample(id).base + path : url(`/runs/${id}/${path}`));
const getJSON = async (full) => { const r = await fetch(full); if (!r.ok) throw new Error(r.statusText); return r.json(); };
const state = { runs: [], current: null, result: null, weights: {}, budget: 0.15, sort: "score", poll: null, summaryPlay: null, summary: [], health: null };

// Runs uploaded from this browser (the public server never lists other people's runs).
const MINE_KEY = "clipranker_runs";
const mine = () => { try { return JSON.parse(localStorage.getItem(MINE_KEY)) || []; } catch { return []; } };
const remember = (id) => { try { localStorage.setItem(MINE_KEY, JSON.stringify([...new Set([id, ...mine()])].slice(0, 50))); } catch {} };
const forget = (id) => { try { localStorage.setItem(MINE_KEY, JSON.stringify(mine().filter((x) => x !== id))); } catch {} };

const fmt = (t) => {
  t = Math.max(0, t);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(s).padStart(2, "0");
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts) {
  const r = await fetch(url(path), opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

function show(view) {
  for (const v of ["emptyView", "progressView", "runView", "benchView", "howView"]) $(v).hidden = v !== view;
  $("navBenchmark").classList.toggle("active", view === "benchView");
  $("upload").classList.toggle("active", view === "howView");
}

/* ---------------- runs list ---------------- */
async function loadRuns() {
  state.runs = STATIC
    ? STATIC.samples.map((x) => ({ id: x.id, name: x.name, status: "done", duration: x.duration, n_scenes: x.n_scenes }))
    : await api("/api/runs?ids=" + encodeURIComponent(mine().join(",")));
  $("runs").innerHTML = state.runs.map((r) => `
    <li data-id="${r.id}" class="${state.current === r.id ? "active" : ""}">
      <div class="run-name" title="${esc(r.name)}">${esc(r.name)}${r.demo ? ' <span class="badge">demo</span>' : ""}</div>
      <div class="run-sub"><span class="badge ${r.status}">${r.status}</span>
        ${r.duration ? fmt(r.duration) : ""} ${r.n_scenes ? "· " + r.n_scenes + " scenes" : ""}</div>
    </li>`).join("") || `<li class="muted">Nothing uploaded yet</li>`;
  $("runsHead").textContent = STATIC ? "Examples" : "Your videos";
  for (const li of $("runs").querySelectorAll("li[data-id]")) li.onclick = () => openRun(li.dataset.id);
}

async function openRun(id) {
  state.current = id;
  clearInterval(state.poll);
  history.replaceState(null, "", "#" + id);
  if (STATIC) { await loadRuns(); return showResult(id); }
  remember(id);
  await loadRuns();
  const run = await api(`/api/runs/${id}`);
  if (run.status === "done") return showResult(id);
  showProgress(run);
  state.poll = setInterval(async () => {
    const r = await api(`/api/runs/${id}`);
    if (r.status === "done") { clearInterval(state.poll); loadRuns(); showResult(id); }
    else { showProgress(r); if (r.status === "error") { clearInterval(state.poll); loadRuns(); } }
  }, 1500);
}

function showProgress(run) {
  show("progressView");
  $("progressTitle").textContent = run.status === "error" ? `Failed: ${run.name}` :
    run.status === "queued" ? `Waiting: ${run.name}` : `Analyzing: ${run.name}`;
  $("progressHint").textContent = run.status === "queued" && run.queue_position ? `Position ${run.queue_position} in the queue.` :
    run.status === "running" ? "This runs on a free CPU server, so a 5 minute video takes a few minutes. You can leave this page and come back." : "";
  const cur = run.stage_index ?? -1;
  $("stageList").innerHTML = STAGES.map((s, i) => {
    const cls = run.status === "done" || i < cur ? "done" : i === cur && run.status === "running" ? "active" : "";
    return `<li class="${cls}">${STAGE_LABELS[s]}</li>`;
  }).join("");
  $("log").textContent = (run.error ? run.error + "\n\n" : "") + (run.log || []).join("\n");
  $("log").scrollTop = $("log").scrollHeight;
}

/* ---------------- upload ---------------- */
async function upload(file) {
  const fd = new FormData();
  fd.append("file", file);
  $("upload").querySelector(".upload-title").textContent = "Uploading...";
  try {
    const run = await api("/api/runs", { method: "POST", body: fd });
    remember(run.id);
    await openRun(run.id);
  } catch (e) { alert("Upload failed: " + e.message); }
  $("upload").querySelector(".upload-title").textContent = "Upload a video";
}
$("fileInput").onchange = (e) => e.target.files[0] && upload(e.target.files[0]);
const up = $("upload");
up.ondragover = (e) => { e.preventDefault(); up.classList.add("drag"); };
up.ondragleave = () => up.classList.remove("drag");
up.ondrop = (e) => { e.preventDefault(); up.classList.remove("drag"); e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]); };

/* ---------------- result view ---------------- */
async function showResult(id) {
  const res = await getJSON(media(id, "result.json"));
  state.result = res;
  state.weights = {};
  for (const m of res.available_modalities) state.weights[m] = res.default_weights[m];
  state.budget = res.budget;
  show("runView");
  const v = res.video;
  $("runTitle").textContent = state.runs.find((r) => r.id === id)?.name || v.source_name;
  $("stats").innerHTML = [
    ["Duration", fmt(v.duration)], ["Shots", res.shots.length], ["Scenes", res.scenes.length],
    ["Speech", res.transcript.length ? `${res.transcript.length} segments (${res.language})` : "none"],
    ["Processed in", fmt(res.timings.total)],
  ].map(([k, x]) => `<span>${k} <b>${x}</b></span>`).join("");
  const player = $("player");
  player.src = media(id, "video.mp4");
  $("budget").value = Math.round(state.budget * 100);
  $("summaryLink").hidden = true;
  $("renderSummary").hidden = $("deleteRun").hidden = !!STATIC;
  renderWeights();
  recompute();
  renderDetails();
}

function renderWeights() {
  const labels = { visual: "Visual (trained model)", audio: "Audio energy", text: "Dialogue relevance" };
  $("weights").innerHTML = Object.keys(state.weights).map((m) => `
    <div class="slider"><label for="w-${m}">${labels[m]} <span id="wv-${m}"></span></label>
    <input type="range" id="w-${m}" min="0" max="100" value="${Math.round(state.weights[m] * 100)}"></div>`).join("")
    + (state.result.available_modalities.includes("text") ? "" : `<p class="muted">No speech was found, so dialogue is not used.</p>`);
  for (const m of Object.keys(state.weights)) {
    $("w-" + m).oninput = (e) => { state.weights[m] = e.target.value / 100; recompute(); };
  }
}
$("budget").oninput = (e) => { state.budget = e.target.value / 100; recompute(); };

function knapsack(values, weights, cap) {
  const n = values.length;
  const best = new Float64Array(cap + 1);
  const take = Array.from({ length: n }, () => new Uint8Array(cap + 1));
  for (let i = 0; i < n; i++) {
    const w = weights[i];
    for (let c = cap; c >= w; c--) {
      const cand = best[c - w] + values[i];
      if (cand > best[c]) { best[c] = cand; take[i][c] = 1; }
    }
  }
  const sel = [];
  let c = cap;
  for (let i = n - 1; i >= 0; i--) if (take[i][c]) { sel.push(i); c -= weights[i]; }
  return new Set(sel);
}

function fused(item) {
  const total = Object.values(state.weights).reduce((a, b) => a + b, 0) || 1;
  return Object.entries(state.weights).reduce((acc, [m, w]) => acc + w * (item.modalities[m] ?? 0), 0) / total;
}

function selectSummary(units, budgetSec) {
  // Shot-level knapsack maximising score x duration (mirrors svs.summarize.select_summary).
  let idx = units.map((u, i) => i).filter((i) => units[i].end - units[i].start >= 1.5);
  if (!idx.length) idx = units.map((u, i) => i);
  const dur = idx.map((i) => units[i].end - units[i].start);
  const picked = knapsack(idx.map((i, k) => units[i]._score * dur[k]), dur.map((d) => Math.max(1, Math.round(d * 10))), Math.round(budgetSec * 10));
  if (!picked.size) {
    const best = idx.reduce((a, b) => (units[b]._score > units[a]._score ? b : a));
    const u = units[best], mid = (u.start + u.end) / 2, half = Math.min(budgetSec, u.end - u.start) / 2;
    return [{ start: mid - half, end: mid + half }];
  }
  const segs = [];
  for (const i of [...picked].map((p) => idx[p]).sort((a, b) => a - b)) {
    const u = units[i], last = segs[segs.length - 1];
    if (last && Math.abs(last.end - u.start) < 1e-3) last.end = u.end;
    else segs.push({ start: u.start, end: u.end });
  }
  return segs;
}

function recompute() {
  const res = state.result, scenes = res.scenes;
  for (const s of scenes) s._score = fused(s);
  for (const u of res.units) u._score = fused(u);
  const order = [...scenes].sort((a, b) => b._score - a._score);
  order.forEach((s, i) => (s._rank = i + 1));
  state.summary = selectSummary(res.units, state.budget * res.video.duration);
  for (const s of scenes) s._sel = state.summary.some((g) => Math.min(s.end, g.end) - Math.max(s.start, g.start) > 0.5);
  const total = Object.values(state.weights).reduce((a, b) => a + b, 0) || 1;
  for (const m of Object.keys(state.weights)) $("wv-" + m).textContent = Math.round(state.weights[m] * 100 / total) + "%";
  $("budgetVal").textContent = Math.round(state.budget * 100) + "%";
  const len = state.summary.reduce((a, g) => a + g.end - g.start, 0);
  $("summaryLen").innerHTML = `${fmt(len)} <span class="muted" style="font-size:.85rem;font-weight:400">${state.summary.length} moments from ${scenes.filter((s) => s._sel).length} clips</span>`;
  renderCards();
  drawTimeline();
}

function renderCards() {
  const res = state.result, id = state.current;
  const scenes = [...res.scenes].sort(state.sort === "score" ? (a, b) => a._rank - b._rank : (a, b) => a.start - b.start);
  $("sceneCount").textContent = `(${scenes.length})`;
  const labels = { visual: "visual", audio: "audio", text: "dialogue" };
  $("cards").innerHTML = scenes.map((s) => {
    const n = String(s.id).padStart(3, "0");
    const clip = STATIC ? `${media(id, "video.mp4")}#t=${s.start.toFixed(2)},${s.end.toFixed(2)}` : media(id, `clips/scene_${n}.mp4`);
    const bars = Object.keys(state.weights).map((m) => `<span>${labels[m]}</span><div class="bar"><div style="width:${(s.modalities[m] ?? 0) * 100}%"></div></div><span>${(s.modalities[m] ?? 0).toFixed(2)}</span>`).join("");
    return `<div class="card ${s._rank <= 3 ? "top" : ""} ${s._sel ? "sel" : ""}">
      <span class="rank-badge">#${s._rank}</span>
      <video src="${clip}" poster="${media(id, `thumbs/scene_${n}.jpg`)}" preload="none" controls playsinline></video>
      <div class="card-body">
        <div class="card-top"><span class="card-title">Scene ${s.id}${s._sel ? '<span class="tag">in reel</span>' : ""}</span><span class="rank">${fmt(s.start)} to ${fmt(s.end)}</span></div>
        <div class="muted" style="font-size:.8rem">${s.duration.toFixed(1)} s, ${s.shots.length} shot${s.shots.length > 1 ? "s" : ""}</div>
        <div class="bars"><span><b>score</b></span><div class="bar total"><div style="width:${s._score * 100}%"></div></div><span><b>${s._score.toFixed(2)}</b></span>${bars}</div>
        <div class="transcript">${s.transcript ? "\u201c" + esc(s.transcript) + "\u201d" : "<i>no speech</i>"}</div>
        <div class="card-actions"><button class="ghost" data-seek="${s.start}">Show in video</button>${STATIC ? "" : `<a href="${clip}" download="rank${String(s._rank).padStart(2, "0")}_scene${n}.mp4">Download clip</a>`}</div>
      </div></div>`;
  }).join("");
  for (const el of $("cards").querySelectorAll("[data-seek]")) el.onclick = () => seek(+el.dataset.seek, true);
}
for (const chip of document.querySelectorAll(".chip")) {
  chip.onclick = () => {
    state.sort = chip.dataset.sort;
    document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c === chip));
    renderCards();
  };
}

function seek(t, play) {
  state.summaryPlay = null;
  const p = $("player");
  p.currentTime = t;
  if (play) p.play();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ---------------- timeline ---------------- */
const LANES = { shots: [6, 22], scenes: [32, 62], curve: [82, 160], speech: [172, 190] };
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  // The CSS height is fixed once; reading canvas.height back would compound the DPR on every redraw.
  if (!canvas.dataset.h) canvas.dataset.h = canvas.getAttribute("height");
  const w = canvas.clientWidth, h = +canvas.dataset.h;
  canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, w, h];
}

function drawTimeline() {
  const res = state.result;
  if (!res) return;
  const canvas = $("timeline");
  const [ctx, W] = setupCanvas(canvas);
  const D = res.video.duration, X = (t) => (t / D) * W;
  const accent = css("--accent"), sel = css("--sel"), muted = css("--muted");
  ctx.font = "11px system-ui"; ctx.fillStyle = muted;

  // shots
  res.shots.forEach((s, i) => {
    ctx.fillStyle = css("--shot"); ctx.globalAlpha = i % 2 ? 0.55 : 0.9;
    ctx.fillRect(X(s.start), LANES.shots[0], Math.max(1, X(s.end) - X(s.start) - 1), LANES.shots[1] - LANES.shots[0]);
  });
  ctx.globalAlpha = 1;
  // scenes
  for (const s of res.scenes) {
    const x0 = X(s.start), x1 = X(s.end);
    ctx.fillStyle = accent; ctx.globalAlpha = 0.15 + 0.75 * s._score;
    ctx.fillRect(x0, LANES.scenes[0], Math.max(1, x1 - x0 - 2), LANES.scenes[1] - LANES.scenes[0]);
    ctx.globalAlpha = 1;
    if (x1 - x0 > 22) { ctx.fillStyle = s._score > 0.55 ? "#fff" : css("--text"); ctx.fillText(s.id, x0 + 4, LANES.scenes[0] + 19); }
  }
  ctx.fillStyle = sel;
  for (const g of state.summary) ctx.fillRect(X(g.start), LANES.scenes[1] + 3, Math.max(2, X(g.end) - X(g.start)), 6);
  // importance curve + per-scene fused score
  const fi = res.frame_importance, [c0, c1] = LANES.curve;
  if (fi.length) {
    const lo = Math.min(...fi), hi = Math.max(...fi), Y = (v) => c1 - ((v - lo) / (hi - lo + 1e-9)) * (c1 - c0);
    ctx.strokeStyle = accent; ctx.lineWidth = 1.5; ctx.beginPath();
    fi.forEach((v, i) => { const x = X(i / res.feature_fps); i ? ctx.lineTo(x, Y(v)) : ctx.moveTo(x, Y(v)); });
    ctx.stroke();
  }
  ctx.strokeStyle = sel; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.beginPath();
  for (const s of res.scenes) { const y = c1 - s._score * (c1 - c0); ctx.moveTo(X(s.start), y); ctx.lineTo(X(s.end), y); }
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = muted; ctx.fillText("importance", 4, c0 + 10);
  // speech
  ctx.fillStyle = css("--speech");
  for (const g of res.transcript) ctx.fillRect(X(g.start), LANES.speech[0], Math.max(1, X(g.end) - X(g.start)), LANES.speech[1] - LANES.speech[0]);
  // playhead
  const t = $("player").currentTime || 0;
  ctx.strokeStyle = css("--text"); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(t), 0); ctx.lineTo(X(t), 200); ctx.stroke();
}

function hitTest(e) {
  const res = state.result, canvas = $("timeline"), r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top, t = (x / r.width) * res.video.duration;
  const scene = res.scenes.find((s) => t >= s.start && t < s.end);
  const shotIdx = res.shots.findIndex((s) => t >= s.start && t < s.end);
  const seg = res.transcript.find((g) => t >= g.start && t < g.end);
  let html = `<b>${fmt(t)}</b>`;
  if (y < LANES.scenes[0]) html += ` · shot ${shotIdx + 1} (${fmt(res.shots[shotIdx]?.start)} to ${fmt(res.shots[shotIdx]?.end)})`;
  else if (y >= LANES.speech[0] - 4) html += seg ? `<br>${esc(seg.text)}` : " · no speech";
  else if (scene) html += ` · scene ${scene.id}, score ${scene._score.toFixed(2)}, rank #${scene._rank}${state.summary.some((g) => t >= g.start && t < g.end) ? " · in reel" : ""}`;
  return [t, html, x, y];
}
$("timeline").onmousemove = (e) => {
  const [, html, x, y] = hitTest(e), tip = $("tooltip");
  tip.hidden = false; tip.innerHTML = html;
  tip.style.left = Math.min(x + 28, $("timeline").clientWidth - 240) + "px"; tip.style.top = y + 50 + "px";
};
$("timeline").onmouseleave = () => ($("tooltip").hidden = true);
$("timeline").onclick = (e) => seek(hitTest(e)[0], false);

$("player").ontimeupdate = () => {
  const res = state.result, t = $("player").currentTime;
  if (!res) return;
  const sp = state.summaryPlay;
  if (sp) {
    const cur = sp.scenes[sp.i];
    if (t >= cur.end - 0.05 || t < cur.start - 0.5) {
      sp.i++;
      if (sp.i >= sp.scenes.length) { state.summaryPlay = null; $("player").pause(); }
      else $("player").currentTime = sp.scenes[sp.i].start;
    }
  }
  const scene = res.scenes.find((s) => t >= s.start && t < s.end);
  $("nowPlaying").textContent = (state.summaryPlay ? `Reel ${state.summaryPlay.i + 1}/${state.summaryPlay.scenes.length} · ` : "")
    + (scene ? `Scene ${scene.id} · score ${scene._score.toFixed(2)} · rank #${scene._rank}` : "");
  drawTimeline();
};
$("playSummary").onclick = () => {
  const scenes = state.summary;
  if (!scenes.length) return;
  $("player").currentTime = scenes[0].start;
  state.summaryPlay = { scenes, i: 0 };
  $("player").play();
};
$("renderSummary").onclick = async () => {
  const btn = $("renderSummary");
  btn.disabled = true; btn.textContent = "Rendering...";
  try {
    const r = await api(`/api/runs/${state.current}/render`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ segments: state.summary }) });
    $("summaryLink").href = url(r.url); $("summaryLink").hidden = false;
  } catch (e) { alert(e.message); }
  btn.disabled = false; btn.textContent = "Export MP4";
};
$("deleteRun").onclick = async () => {
  if (!confirm("Delete this video and its clips from the server?")) return;
  await api(`/api/runs/${state.current}`, { method: "DELETE" });
  forget(state.current);
  state.current = null; state.result = null; history.replaceState(null, "", "#");
  show("emptyView"); loadRuns();
};
window.onresize = () => { drawTimeline(); if (!$("benchView").hidden) drawBench(); };

function renderDetails() {
  const res = state.result;
  const timing = Object.entries(res.timings).map(([k, v]) => `<tr><td>${STAGE_LABELS[k] || k}</td><td>${v.toFixed(1)}s</td></tr>`).join("");
  const models = Object.entries(res.models).map(([k, v]) => `<tr><td>${k}</td><td>${esc(v)}</td></tr>`).join("");
  const chosen = res.boundaries.filter((b) => b.chosen).length;
  $("details").innerHTML = `
    <div><h4>Stage timings</h4><table>${timing}</table></div>
    <div><h4>Models</h4><table>${models}</table></div>
    <div><h4>Segmentation</h4><table>
      <tr><td>Video</td><td>${res.video.width}×${res.video.height} @ ${res.video.fps.toFixed(2)} fps</td></tr>
      <tr><td>Shots / sub-shot units</td><td>${res.shots.length} / ${res.units.length}</td></tr>
      <tr><td>Candidate cuts, scene boundaries</td><td>${res.boundaries.length}, ${chosen}</td></tr>
      <tr><td>Cuts inside a sentence</td><td>${res.boundaries.filter((b) => b.speech_continues).length}</td></tr>
      <tr><td>Scene boundaries inside a sentence</td><td>${res.boundaries.filter((b) => b.chosen && b.speech_continues).length}</td></tr>
    </table></div>`;
}

/* ---------------- benchmark ---------------- */
let bench = null;
async function showBench() {
  clearInterval(state.poll);
  state.current = null; history.replaceState(null, "", "#benchmark"); loadRuns();
  show("benchView");
  if (!bench) {
    try { bench = STATIC ? await getJSON(STATIC.benchmark) : await api("/api/benchmark"); }
    catch (e) { $("benchProtocol").textContent = "Benchmark results not found: " + e.message; return; }
  }
  $("benchProtocol").textContent = bench.protocol;
  const row = (r) => `<tr class="${r.kind}"><td>${esc(r.label)}<span class="desc">${esc(r.description)}</span></td>
    ${["kendall", "spearman", "f1"].map((k) => `<td>${r.mean[k].toFixed(3)}${r.std ? ` <span class="muted">± ${r.std[k].toFixed(3)}</span>` : ""}</td>`).join("")}</tr>`;
  $("benchTable").innerHTML = `<tr><th>Model</th><th>Kendall τ</th><th>Spearman ρ</th><th>F1 at 15%</th></tr>` + bench.rows.map(row).join("");
  $("benchVideo").innerHTML = bench.videos.map((v, i) => `<option value="${i}">${esc(v.title)} (${esc(v.category)}) · τ=${v.kendall.toFixed(2)}</option>`).join("");
  $("benchVideo").onchange = drawBench;
  drawBench();
}
function drawBench() {
  if (!bench) return;
  const v = bench.videos[+$("benchVideo").value || 0];
  const [ctx, W, H] = setupCanvas($("benchChart"));
  const pad = 24, X = (i) => pad + (i / (v.gt.length - 1)) * (W - 2 * pad), Y = (y) => H - pad - y * (H - 2 * pad);
  const norm = (a) => { const lo = Math.min(...a), hi = Math.max(...a); return a.map((x) => (x - lo) / (hi - lo + 1e-9)); };
  ctx.strokeStyle = css("--border"); ctx.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);
  for (const [series, color, width] of [[norm(v.gt), css("--gt"), 1.5], [norm(v.pred), css("--accent"), 2]]) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    series.forEach((y, i) => (i ? ctx.lineTo(X(i), Y(y)) : ctx.moveTo(X(i), Y(y))));
    ctx.stroke();
  }
  ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui";
  ctx.fillText("0:00", pad, H - 6); ctx.fillText(fmt(v.gt.length / 2), W - pad - 30, H - 6);
}
$("navBenchmark").onclick = showBench;

/* ---------------- run it yourself ---------------- */
function showHow() {
  clearInterval(state.poll);
  state.current = null; history.replaceState(null, "", "#run"); loadRuns();
  show("howView");
}
if (STATIC) {
  const box = $("upload");
  box.classList.add("as-link");
  box.querySelector(".upload-title").textContent = "Run it on your own video";
  $("uploadHint").textContent = "Free on Google Colab, or on your computer";
  $("fileInput").disabled = true;
  box.onclick = (e) => { e.preventDefault(); showHow(); };
  box.ondragover = box.ondrop = (e) => e.preventDefault();
  $("colabLink").href = STATIC.colab;
  $("repoLink").href = STATIC.repo;
  $("repoUrl").textContent = STATIC.repo + ".git";
  $("emptyHint").textContent = "Pick an example on the left to see its ranked clips, or run ClipRanker on your own video.";
}

/* ---------------- server status ---------------- */
async function checkServer() {
  const el = $("server"), label = el.querySelector("span"), started = Date.now();
  if (STATIC) { el.hidden = true; return; }
  for (;;) {
    try {
      const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 15000);
      const h = await (await fetch(url("/api/health"), { signal: ctl.signal })).json();
      clearTimeout(timer);
      state.health = h;
      if (!h.public) $("emptyHint").textContent = "Results are saved on this machine in the runs/ folder.";
      else $("maxDur").textContent = Math.round(h.max_duration / 60);
      if (h.models_loaded) {
        el.className = "server ok";
        label.textContent = h.queue ? `Server ready, ${h.queue} in queue` : `Server ready (${h.device.toUpperCase()})`;
        return;
      }
      el.className = "server waking"; label.textContent = "Server is loading its models";
    } catch {
      const late = Date.now() - started > 240000;
      el.className = late ? "server down" : "server waking";
      label.textContent = late ? "Server is not responding" : "Waking the server up (about a minute)";
    }
    await new Promise((r) => setTimeout(r, 4000));
  }
}

/* ---------------- boot ---------------- */
(async () => {
  checkServer();
  for (;;) {
    try { await loadRuns(); break; } catch { await new Promise((r) => setTimeout(r, 4000)); }
  }
  const h = location.hash.slice(1);
  if (h === "benchmark") showBench();
  else if (h === "run" && STATIC) showHow();
  else if (h && h.length > 8) openRun(h).catch(() => show("emptyView"));
  else if (state.runs.find((r) => r.status === "done")) openRun(state.runs.find((r) => r.status === "done").id);
  setInterval(() => { if ($("runView").hidden && $("progressView").hidden) loadRuns(); }, 5000);
})();
