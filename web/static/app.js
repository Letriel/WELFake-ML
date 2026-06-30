"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---- Tabs ----
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll(".tab").forEach((b) => {
      const on = b.dataset.tab === tab;
      b.classList.toggle("bg-slate-700", on);
      b.classList.toggle("text-white", on);
      b.classList.toggle("text-slate-400", !on);
    });
    $("panel-single").classList.toggle("hidden", tab !== "single");
    $("panel-batch").classList.toggle("hidden", tab !== "batch");
  });
});

// ---- Health badge ----
fetch("/health")
  .then((r) => r.json())
  .then((h) => {
    const ok = h.model_loaded && h.tokenizer_loaded;
    $("health").innerHTML = ok
      ? `<span class="text-emerald-400">●</span> model + tokenizer ready · vocab ${h.vocab_size.toLocaleString()}`
      : `<span class="text-amber-400">●</span> not ready — model_loaded=${h.model_loaded}, tokenizer_loaded=${h.tokenizer_loaded}`;
  })
  .catch(() => ($("health").textContent = "health check failed"));

// ---- Render a single result ----
function resultCard(d) {
  const fake = d.label === "FAKE";
  const accent = fake ? "rose" : "emerald";
  const pct = (d.confidence * 100).toFixed(1);
  const probPct = (d.probability * 100).toFixed(1);

  let translated = "";
  if (d.input_was_translated) {
    translated = `
      <details class="mt-3 text-xs text-slate-400">
        <summary class="cursor-pointer hover:text-slate-200">Translated from
          <span class="font-semibold uppercase">${esc(d.detected_language || "auto")}</span> → English</summary>
        <p class="mt-2 p-3 bg-slate-950/60 border border-slate-800 rounded-lg leading-relaxed">${esc(d.translated_text)}</p>
      </details>`;
  }

  let verify = "";
  const v = d.verification;
  if (v) {
    const sources = (v.sources || [])
      .map(
        (s) => `
        <a href="${esc(s.url)}" target="_blank" rel="noopener"
           class="block p-3 rounded-lg bg-slate-950/50 border border-slate-800 hover:border-cyan-500/50 transition">
          <div class="text-sm font-medium text-cyan-300">${esc(s.title) || "(untitled)"}</div>
          <div class="text-xs text-slate-400 mt-0.5 line-clamp-2">${esc(s.snippet)}</div>
        </a>`
      )
      .join("");
    verify = `
      <div class="mt-4 pt-4 border-t border-slate-800">
        <div class="text-xs font-semibold text-slate-400 mb-2">🔎 VERIFICATION · ${esc(v.method)}</div>
        <p class="text-sm text-slate-300 mb-3">${esc(v.summary || "")}</p>
        <div class="space-y-2">${sources || '<p class="text-xs text-slate-500">No sources returned.</p>'}</div>
      </div>`;
  } else {
    const msg = fake ? "Want to verify?" : "Having trust issue?";
    verify = `
      <div id="verify-section" class="mt-4 pt-4 border-t border-slate-800 flex flex-col items-center">
        <div class="text-sm text-slate-400 mb-3">${msg}</div>
        <button id="btn-manual-verify" class="bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold px-4 py-2 rounded-lg transition border border-slate-700">
          Verify this claim
        </button>
      </div>`;
  }

  return `
    <div class="fade-in bg-slate-900/60 border border-${accent}-500/30 rounded-2xl p-5 shadow-xl">
      <div class="flex items-center gap-3">
        <span class="text-3xl">${fake ? "🚫" : "✅"}</span>
        <div>
          <div class="text-2xl font-extrabold text-${accent}-400 tracking-tight">${d.label}</div>
          <div class="text-xs text-slate-400">${pct}% confidence · raw sigmoid ${probPct}%</div>
        </div>
        <div class="ml-auto text-right">
          <div class="text-[10px] uppercase tracking-wider text-slate-500">Fake-ness</div>
          <div class="text-lg font-bold text-${accent}-300">${probPct}%</div>
        </div>
      </div>
      <div class="mt-4 h-2 w-full bg-slate-800 rounded-full overflow-hidden">
        <div class="bar h-full bg-${accent}-500" style="width:${probPct}%"></div>
      </div>
      ${translated}
      ${verify}
    </div>`;
}

// ---- Single analyze ----
$("analyze").addEventListener("click", async () => {
  const btn = $("analyze");
  const box = $("result");
  btn.disabled = true;
  btn.textContent = "Analyzing…";
  box.classList.remove("hidden");
  box.innerHTML = `<div class="text-sm text-slate-400 animate-pulse p-4">Running model…</div>`;
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: $("title").value,
        text: $("text").value,
        translate: $("translate").checked,
        verify: $("verify").checked,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    box.innerHTML = resultCard(data);

    const btnVerify = $("btn-manual-verify");
    if (btnVerify) {
      btnVerify.addEventListener("click", async () => {
        const vSec = $("verify-section");
        vSec.innerHTML = `<div class="text-sm text-slate-400 animate-pulse text-center">Running verification…</div>`;
        try {
          const vRes = await fetch("/api/verify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              title: $("title").value,
              text: $("text").value,
              translate: $("translate").checked
            })
          });
          const vData = await vRes.json();
          if (!vRes.ok) throw new Error(vData.detail || "Verify failed");
          data.verification = vData;
          box.innerHTML = resultCard(data);
        } catch (ve) {
          vSec.innerHTML = `<div class="text-sm text-rose-400 text-center">⚠️ ${esc(ve.message)}</div>`;
        }
      });
    }

  } catch (e) {
    box.innerHTML = `<div class="text-sm text-rose-400 p-4 border border-rose-500/30 rounded-xl bg-rose-500/5">⚠️ ${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
  }
});

// ---- Batch ----
$("batch-json").value = JSON.stringify(
  [
    { title: "The Great Moon Hoax", text: "In 1835 a newspaper claimed an astronomer observed winged humanoids living on the moon." },
    { title: "Reuters: central bank holds rates", text: "The central bank kept its benchmark interest rate unchanged on Thursday, citing stable inflation." },
  ],
  null,
  2
);

$("run-batch").addEventListener("click", async () => {
  const btn = $("run-batch");
  const box = $("batch-result");
  let payload;
  try {
    payload = JSON.parse($("batch-json").value);
    if (!Array.isArray(payload)) throw new Error("JSON must be an array");
  } catch (e) {
    box.innerHTML = `<div class="text-sm text-rose-400 p-4">Invalid JSON: ${esc(e.message)}</div>`;
    return;
  }
  btn.disabled = true;
  btn.textContent = "Running…";
  box.innerHTML = `<div class="text-sm text-slate-400 animate-pulse p-4">Processing ${payload.length} item(s)…</div>`;
  try {
    const res = await fetch(`/api/predict/batch?translate=${$("batch-translate").checked}&verify=false`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    const rows = data.results
      .map((d, i) => {
        const fake = d.label === "FAKE";
        const accent = fake ? "rose" : "emerald";
        return `<tr class="border-t border-slate-800">
          <td class="py-2 pr-3 text-slate-500">${i + 1}</td>
          <td class="py-2 pr-3 truncate max-w-[280px]">${esc(payload[i].title || payload[i].text || "")}</td>
          <td class="py-2 pr-3 font-bold text-${accent}-400">${d.label}</td>
          <td class="py-2 text-slate-400">${(d.confidence * 100).toFixed(1)}%</td>
        </tr>`;
      })
      .join("");
    box.innerHTML = `
      <div class="fade-in bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <table class="w-full text-sm">
          <thead><tr class="text-xs text-slate-500 text-left">
            <th class="pb-2 pr-3">#</th><th class="pb-2 pr-3">Item</th><th class="pb-2 pr-3">Label</th><th class="pb-2">Conf.</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  } catch (e) {
    box.innerHTML = `<div class="text-sm text-rose-400 p-4">⚠️ ${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run batch";
  }
});
