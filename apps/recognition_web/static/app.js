const state = {
  models: [],
  source: "dataset",
  selectedSample: null,
  selectedFile: null,
};

const els = {
  primaryModel: document.querySelector("#primaryModel"),
  compareModels: document.querySelector("#compareModels"),
  modelMeta: document.querySelector("#modelMeta"),
  modelCount: document.querySelector("#modelCount"),
  deviceText: document.querySelector("#deviceText"),
  inputMode: document.querySelector("#inputMode"),
  sampleSearch: document.querySelector("#sampleSearch"),
  sampleList: document.querySelector("#sampleList"),
  datasetPane: document.querySelector("#datasetPane"),
  uploadPane: document.querySelector("#uploadPane"),
  fileInput: document.querySelector("#fileInput"),
  fileName: document.querySelector("#fileName"),
  runPredict: document.querySelector("#runPredict"),
  predictionCards: document.querySelector("#predictionCards"),
  compareTable: document.querySelector("#compareTable"),
  channelGrid: document.querySelector("#channelGrid"),
  statsTable: document.querySelector("#statsTable"),
  sampleMeta: document.querySelector("#sampleMeta"),
  resultHint: document.querySelector("#resultHint"),
  toast: document.querySelector("#toast"),
};

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.classList.toggle("error", isError);
  els.toast.classList.remove("hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => els.toast.classList.add("hidden"), 3600);
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

function metricText(summary, key) {
  const item = summary?.metrics?.[key];
  if (!item) return "-";
  return `${item.mean.toFixed(4)} ± ${item.std.toFixed(4)}`;
}

function modelLabel(model) {
  return `${model.run_name}`;
}

function currentModel() {
  return state.models.find((item) => item.run_name === els.primaryModel.value);
}

function selectedModels() {
  const checked = [...document.querySelectorAll("input[name='compareModel']:checked")].map((item) => item.value);
  return Array.from(new Set([els.primaryModel.value, ...checked].filter(Boolean)));
}

async function loadModels() {
  const data = await fetchJson("/api/models");
  state.models = data.models;
  els.deviceText.textContent = data.device;
  els.modelCount.textContent = String(state.models.length);
  renderModelOptions();
  renderCompareOptions();
  renderModelMeta();
  await loadSamples();
}

function renderModelOptions() {
  const preferred = state.models.find((item) => item.run_name === "uvd4_attention_5fold") || state.models[0];
  els.primaryModel.innerHTML = state.models
    .map((model) => `<option value="${model.run_name}">${modelLabel(model)}</option>`)
    .join("");
  if (preferred) els.primaryModel.value = preferred.run_name;
}

function renderCompareOptions() {
  els.compareModels.innerHTML = state.models
    .map(
      (model) => `
      <label class="check-item">
        <input type="checkbox" name="compareModel" value="${model.run_name}">
        <span>
          <strong>${model.run_name}</strong><br>
          <small>${model.label_mode} · ${model.input_mode} · F1 ${metricText(model.summary, "macro_f1")}</small>
        </span>
      </label>
    `,
    )
    .join("");
}

function renderModelMeta() {
  const model = currentModel();
  if (!model) return;
  els.inputMode.textContent = model.input_mode;
  els.modelMeta.innerHTML = `
    <div><span>类别任务</span><strong>${model.label_mode}</strong></div>
    <div><span>输入模式</span><strong>${model.input_mode}</strong></div>
    <div><span>Macro-F1</span><strong>${metricText(model.summary, "macro_f1")}</strong></div>
    <div><span>UAR</span><strong>${metricText(model.summary, "uar")}</strong></div>
  `;
}

async function loadSamples() {
  const model = currentModel();
  const labelMode = model?.label_mode || "4class";
  const q = encodeURIComponent(els.sampleSearch.value.trim());
  const data = await fetchJson(`/api/samples?label_mode=${labelMode}&q=${q}`);
  renderSamples(data.samples);
}

function renderSamples(samples) {
  if (!samples.length) {
    els.sampleList.innerHTML = `<div class="empty">没有匹配样本</div>`;
    return;
  }
  els.sampleList.innerHTML = samples
    .map(
      (sample) => `
      <div class="sample-item ${sample.sample_id === state.selectedSample ? "active" : ""}" data-id="${sample.sample_id}">
        <div>
          <strong>${sample.sample_id}</strong>
          <small>${sample.subject} · 4类 ${sample.emotion_4} · 7类 ${sample.emotion_7}</small>
        </div>
        <span class="badge">${currentModel()?.label_mode === "7class" ? sample.emotion_7 : sample.emotion_4}</span>
      </div>
    `,
    )
    .join("");
  document.querySelectorAll(".sample-item").forEach((item) => {
    item.addEventListener("click", () => {
      state.selectedSample = item.dataset.id;
      renderSamples(samples);
    });
  });
  if (!state.selectedSample && samples[0]) {
    state.selectedSample = samples[0].sample_id;
    renderSamples(samples);
  }
}

function setSource(source) {
  state.source = source;
  document.querySelectorAll(".segment").forEach((button) => {
    button.classList.toggle("active", button.dataset.source === source);
  });
  els.datasetPane.classList.toggle("hidden", source !== "dataset");
  els.uploadPane.classList.toggle("hidden", source !== "upload");
}

async function runPrediction() {
  const models = selectedModels();
  if (!models.length) {
    showToast("请至少选择一个模型", true);
    return;
  }
  els.runPredict.disabled = true;
  els.resultHint.textContent = "识别中...";
  try {
    let data;
    if (state.source === "dataset") {
      if (!state.selectedSample) throw new Error("请选择待识别样本");
      data = await fetchJson(`/api/predict?sample_id=${encodeURIComponent(state.selectedSample)}&models=${models.join(",")}`);
    } else {
      if (!state.selectedFile) throw new Error("请上传 .npy 文件");
      const form = new FormData();
      form.append("models", models.join(","));
      form.append("file", state.selectedFile);
      data = await fetchJson("/api/predict_upload", { method: "POST", body: form });
    }
    renderResult(data);
  } catch (err) {
    showToast(err.message, true);
    els.resultHint.textContent = "识别失败";
  } finally {
    els.runPredict.disabled = false;
  }
}

function renderResult(data) {
  els.resultHint.textContent = `${data.predictions.length} 个模型完成`;
  const sample = data.sample || {};
  els.sampleMeta.textContent = sample.source === "uploaded"
    ? `上传文件：${sample.sample_id}`
    : `${sample.sample_id} · subject ${sample.subject} · 4类 ${sample.emotion_4} · 7类 ${sample.emotion_7}`;
  renderPredictions(data.predictions);
  renderComparison(data.predictions);
  renderVisual(data.visual);
}

function renderPredictions(predictions) {
  els.predictionCards.classList.remove("empty");
  els.predictionCards.innerHTML = predictions
    .map((pred) => {
      const rows = [...pred.probabilities]
        .sort((a, b) => b.probability - a.probability)
        .map(
          (item) => `
          <div class="prob-row">
            <span>${item.label}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${(item.probability * 100).toFixed(1)}%"></div></div>
            <strong>${(item.probability * 100).toFixed(1)}%</strong>
          </div>
        `,
        )
        .join("");
      return `
        <article class="prediction-card">
          <div class="prediction-head">
            <div>
              <h4>${pred.run_name}</h4>
              <span class="muted">${pred.label_mode} · ${pred.input_mode}</span>
            </div>
            <div>
              <div class="confidence">${pred.predicted_label}</div>
              <span class="muted">${(pred.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
          ${rows}
        </article>
      `;
    })
    .join("");
}

function renderComparison(predictions) {
  if (!predictions.length) {
    els.compareTable.innerHTML = "";
    return;
  }
  els.compareTable.innerHTML = `
    <table>
      <thead><tr><th>模型</th><th>任务</th><th>输入</th><th>预测</th><th>置信度</th></tr></thead>
      <tbody>
        ${predictions
          .map(
            (pred) => `
            <tr>
              <td>${pred.run_name}</td>
              <td>${pred.label_mode}</td>
              <td>${pred.input_mode}</td>
              <td><strong>${pred.predicted_label}</strong></td>
              <td>${(pred.confidence * 100).toFixed(1)}%</td>
            </tr>
          `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderVisual(visual) {
  els.channelGrid.classList.remove("empty");
  els.channelGrid.innerHTML = visual.channels
    .map(
      (item) => `
      <div class="channel-card">
        <img src="${item.src}" alt="${item.name}">
        <span>${item.name}</span>
      </div>
    `,
    )
    .join("");
  els.statsTable.innerHTML = `
    <table>
      <thead><tr><th>通道</th><th>均值</th><th>标准差</th><th>|均值|</th><th>范围</th></tr></thead>
      <tbody>
        ${visual.stats
          .map(
            (row) => `
            <tr>
              <td>${row.name}</td>
              <td>${row.mean.toFixed(4)}</td>
              <td>${row.std.toFixed(4)}</td>
              <td>${row.abs_mean.toFixed(4)}</td>
              <td>${row.min.toFixed(3)} ~ ${row.max.toFixed(3)}</td>
            </tr>
          `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function selectRecommended() {
  const names = new Set([
    "uv4_baseline_5fold",
    "uvd4_attention_5fold",
    "uvd4_residual_masked_attention_5fold",
    "uv7_baseline_5fold",
    "uvd7_residual_masked_attention_5fold",
  ]);
  document.querySelectorAll("input[name='compareModel']").forEach((item) => {
    item.checked = names.has(item.value);
  });
}

document.querySelectorAll(".segment").forEach((button) => {
  button.addEventListener("click", () => setSource(button.dataset.source));
});

els.primaryModel.addEventListener("change", async () => {
  renderModelMeta();
  state.selectedSample = null;
  await loadSamples();
});
document.querySelector("#refreshModels").addEventListener("click", loadModels);
document.querySelector("#searchSamples").addEventListener("click", loadSamples);
document.querySelector("#selectRecommended").addEventListener("click", selectRecommended);
els.sampleSearch.addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadSamples();
});
els.fileInput.addEventListener("change", () => {
  state.selectedFile = els.fileInput.files[0] || null;
  els.fileName.textContent = state.selectedFile ? state.selectedFile.name : "选择 .npy 张量文件";
});
els.runPredict.addEventListener("click", runPrediction);

loadModels().catch((err) => showToast(err.message, true));
