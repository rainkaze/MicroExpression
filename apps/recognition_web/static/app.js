const state = {
  metadata: null,
  history: [],
  mode: "sample",
};

const sourceNames = {
  dataset_sample: "数据集样本",
  computed_flow: "图片计算光流",
  uploaded_tensor: "上传张量",
};

const labelNames = {
  negative: "消极",
  positive: "积极",
  surprise: "惊讶",
  others: "其他",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function displayLabel(label) {
  return labelNames[label] ? `${labelNames[label]} (${label})` : label;
}

function formatPct(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 4200);
}

async function loadMetadata() {
  const response = await fetch("/api/metadata");
  const metadata = await response.json();
  if (!response.ok || metadata.error) {
    throw new Error(metadata.error || "模型信息加载失败。");
  }
  state.metadata = metadata;
  $("#deviceBadge").textContent = metadata.device;
  $("#runName").textContent = metadata.run_name;
  $("#modelName").textContent = metadata.model_name;
  $("#inputMode").textContent = metadata.input_mode;
  $("#labelMode").textContent = metadata.label_mode;

  const select = $("#foldSelect");
  select.innerHTML = "";
  const ensemble = document.createElement("option");
  ensemble.value = "ensemble";
  ensemble.textContent = `五折集成 (${metadata.folds.length} 个模型)`;
  select.appendChild(ensemble);
  metadata.folds.forEach((fold) => {
    const option = document.createElement("option");
    option.value = fold;
    option.textContent = `只用 ${fold}`;
    select.appendChild(option);
  });
}

async function loadSamples() {
  const response = await fetch("/api/samples");
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || "样本列表加载失败。");
  }
  const select = $("#sampleSelect");
  select.innerHTML = "";
  payload.samples.forEach((sample) => {
    const option = document.createElement("option");
    option.value = sample.sample_id;
    option.textContent = `${sample.sample_id} · ${displayLabel(sample.label)} · ${sample.subject}`;
    select.appendChild(option);
  });
}

function bindTabs() {
  $$(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      $$(".tab").forEach((tab) => tab.classList.toggle("active", tab === button));
      $("#sampleForm").classList.toggle("active-form", state.mode === "sample");
      $("#imageForm").classList.toggle("active-form", state.mode === "images");
      $("#tensorForm").classList.toggle("active-form", state.mode === "tensor");
    });
  });
}

function bindFileNames() {
  [
    ["#onsetInput", "#onsetName", "选择图片"],
    ["#apexInput", "#apexName", "选择图片"],
    ["#tensorInput", "#tensorName", "选择 .npy 文件"],
  ].forEach(([inputSelector, labelSelector, fallback]) => {
    $(inputSelector).addEventListener("change", (event) => {
      const file = event.target.files[0];
      $(labelSelector).textContent = file ? file.name : fallback;
    });
  });
}

function currentFold() {
  return $("#foldSelect").value || "ensemble";
}

async function postForm(url, formData, button) {
  button.disabled = true;
  button.querySelector("span").textContent = "识别中...";
  try {
    formData.append("fold", currentFold());
    const response = await fetch(url, { method: "POST", body: formData });
    const result = await response.json();
    if (!response.ok || result.error) {
      throw new Error(result.error || "识别失败。");
    }
    renderResult(result);
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "开始识别";
    if (state.mode === "sample") {
      button.querySelector("span").textContent = "识别该样本";
    }
  }
}

function bindForms() {
  $("#sampleForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const sampleId = $("#sampleSelect").value;
    if (!sampleId) {
      showToast("请先选择一个数据集样本。");
      return;
    }
    const data = new FormData();
    data.append("sample_id", sampleId);
    await postForm("/api/predict-sample", data, event.submitter);
  });

  $("#imageForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const onset = $("#onsetInput").files[0];
    const apex = $("#apexInput").files[0];
    if (!onset || !apex) {
      showToast("请同时上传 onset 开始帧和 apex 峰值帧。");
      return;
    }
    const data = new FormData();
    data.append("onset", onset);
    data.append("apex", apex);
    await postForm("/api/predict-images", data, event.submitter);
  });

  $("#tensorForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const tensor = $("#tensorInput").files[0];
    if (!tensor) {
      showToast("请上传 .npy 光流张量。");
      return;
    }
    const data = new FormData();
    data.append("tensor", tensor);
    await postForm("/api/predict-tensor", data, event.submitter);
  });
}

function renderResult(result) {
  $("#predictionTitle").textContent = displayLabel(result.prediction);
  $("#predictionSubtitle").textContent = `${currentFold()} · ${result.model_count} 个模型 · ${sourceNames[result.source] || result.source}`;
  $("#confidenceValue").textContent = formatPct(result.confidence);
  $("#modelCount").textContent = `${result.model_count} 个模型`;
  $("#sourceType").textContent = sourceNames[result.source] || result.source;

  const probabilities = $("#probabilities");
  probabilities.classList.remove("empty");
  probabilities.innerHTML = "";
  result.ranking.forEach((item) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <strong>${displayLabel(item.label)}</strong>
      <div class="bar"><span style="width: ${Math.max(2, item.probability * 100)}%"></span></div>
      <span>${formatPct(item.probability)}</span>
    `;
    probabilities.appendChild(row);
  });

  $("#statShape").textContent = result.stats.shape.join(" x ");
  $("#statMean").textContent = result.stats.mean.toFixed(4);
  $("#statStd").textContent = result.stats.std.toFixed(4);
  $("#statRange").textContent = `${result.stats.min.toFixed(3)} 到 ${result.stats.max.toFixed(3)}`;

  const frame = $(".visual-frame");
  $("#motionImage").src = result.visualization;
  frame.classList.add("has-image");

  renderSampleInfo(result.sample);
  addHistory(result);
}

function renderSampleInfo(sample) {
  const panel = $("#sampleInfo");
  if (!sample) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  $("#sampleTruth").textContent = `真实标签：${displayLabel(sample.label)}`;
  $("#sampleMeta").innerHTML = `
    <span>样本：${sample.sample_id}</span>
    <span>主体：${sample.subject}</span>
    <span>视频：${sample.video_code}</span>
    <span>onset/apex/offset：${sample.onset}/${sample.apex}/${sample.offset}</span>
    <span>7类标签：${sample.emotion_7}</span>
  `;
}

function addHistory(result) {
  const item = {
    prediction: result.prediction,
    confidence: result.confidence,
    fold: currentFold(),
    source: sourceNames[result.source] || result.source,
    time: new Date().toLocaleTimeString(),
  };
  state.history.unshift(item);
  state.history = state.history.slice(0, 8);
  renderHistory();
}

function renderHistory() {
  const list = $("#historyList");
  list.innerHTML = "";
  if (!state.history.length) {
    list.innerHTML = `<p class="muted">暂无识别记录。</p>`;
    return;
  }
  state.history.forEach((item) => {
    const row = document.createElement("div");
    row.className = "history-item";
    row.innerHTML = `
      <div>
        <strong>${displayLabel(item.prediction)}</strong>
        <span class="muted"> · ${item.source} · ${item.time}</span>
      </div>
      <span>${formatPct(item.confidence)}</span>
    `;
    list.appendChild(row);
  });
}

async function init() {
  bindTabs();
  bindFileNames();
  bindForms();
  $("#clearHistory").addEventListener("click", () => {
    state.history = [];
    renderHistory();
  });
  renderHistory();
  try {
    await loadMetadata();
    await loadSamples();
  } catch (error) {
    showToast(error.message);
  }
}

init();
