const headers = { "Content-Type": "application/json" };

const jsonBox = (id) => document.getElementById(id);

const prettyPrint = (target, data) => {
  const box = jsonBox(target);
  box.textContent = JSON.stringify(data, null, 2);
};

const formatBytes = (bytes) => {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + " " + sizes[i];
};

const formatDate = (dateString) => {
  if (!dateString) return "-";
  try {
    const date = new Date(dateString);
    return date.toLocaleString("ko-KR");
  } catch {
    return dateString;
  }
};

// 모델 리스트 로드 및 드롭다운 업데이트
let modelsCache = [];

const loadModels = async () => {
  try {
    const response = await fetch("/models");
    if (!response.ok) throw new Error("모델 리스트 조회 실패");
    const models = await response.json();
    modelsCache = models;
    
    // 모델 드롭다운 업데이트
    const modelSelect = document.getElementById("model-select");
    const indexSelect = document.getElementById("index-select");
    
    modelSelect.innerHTML = '<option value="">모델을 선택하세요</option>';
    indexSelect.innerHTML = '<option value="">인덱스 파일 없음</option>';
    
    models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.model_id;
      option.textContent = `${model.model_id} (${model.model_files.length}개 파일)`;
      modelSelect.appendChild(option);
    });
    
    return models;
  } catch (error) {
    console.error("모델 로드 실패:", error);
    return [];
  }
};

// 모델 선택 시 모델 파일 및 인덱스 파일 업데이트
document.getElementById("model-select").addEventListener("change", (e) => {
  const modelId = e.target.value;
  const modelPathInput = document.getElementById("model-path-input");
  const indexSelect = document.getElementById("index-select");
  const indexPathInput = document.getElementById("index-path-input");
  
  if (!modelId) {
    modelPathInput.value = "";
    indexSelect.innerHTML = '<option value="">인덱스 파일 없음</option>';
    indexPathInput.value = "";
    return;
  }
  
  const model = modelsCache.find((m) => m.model_id === modelId);
  if (!model) return;
  
  // 첫 번째 모델 파일 자동 선택 (전체 경로 구성: logs/{model_id}/{filename})
  if (model.model_files.length > 0) {
    const modelFileName = model.model_files[0];
    modelPathInput.value = `logs/${modelId}/${modelFileName}`;
  }
  
  // 인덱스 파일 드롭다운 업데이트
  indexSelect.innerHTML = '<option value="">인덱스 파일 없음</option>';
  model.index_files.forEach((indexFile) => {
    const option = document.createElement("option");
    // 전체 경로 구성: logs/{model_id}/{filename}
    const fullIndexPath = `logs/${modelId}/${indexFile}`;
    option.value = fullIndexPath;
    option.textContent = indexFile;
    indexSelect.appendChild(option);
  });
});

// 인덱스 파일 선택 시 인덱스 경로 업데이트
document.getElementById("index-select").addEventListener("change", (e) => {
  const indexPath = e.target.value;
  const indexPathInput = document.getElementById("index-path-input");
  indexPathInput.value = indexPath;
});

// 모델 리스트를 테이블로 표시
const renderModelsTable = (models) => {
  const container = document.getElementById("models-list-container");
  
  if (!models || models.length === 0) {
    container.innerHTML = '<div class="result empty-state">모델이 없습니다</div>';
    return;
  }
  
  let html = `
    <table class="data-table">
      <thead>
        <tr>
          <th>모델 ID</th>
          <th>모델 파일</th>
          <th>인덱스 파일</th>
          <th>생성 시간</th>
        </tr>
      </thead>
      <tbody>
  `;
  
  models.forEach((model) => {
    const modelFiles = model.model_files.map(f => f.split("/").pop()).join(", ");
    const indexFiles = model.index_files.map(f => f.split("/").pop()).join(", ") || "-";
    html += `
      <tr>
        <td><strong>${model.model_id}</strong></td>
        <td>${modelFiles || "-"}</td>
        <td>${indexFiles}</td>
        <td>${formatDate(model.created_at)}</td>
      </tr>
    `;
  });
  
  html += `
      </tbody>
    </table>
  `;
  
  container.innerHTML = html;
};

// 추론 결과 리스트를 테이블로 표시
const renderOutputsTable = (outputs) => {
  const container = document.getElementById("outputs-list-container");
  
  if (!outputs || outputs.length === 0) {
    container.innerHTML = '<div class="result empty-state">추론 결과가 없습니다</div>';
    return;
  }
  
  let html = `
    <table class="data-table">
      <thead>
        <tr>
          <th>파일명</th>
          <th>크기</th>
          <th>생성 시간</th>
          <th>작업</th>
        </tr>
      </thead>
      <tbody>
  `;
  
  outputs.forEach((output) => {
    const fileName = output.output_id;
    const fileSize = formatBytes(output.file_size);
    const createdAt = formatDate(output.created_at);
    const outputId = encodeURIComponent(output.output_id);
    
    html += `
      <tr>
        <td>
          <span class="file-link" data-output-id="${outputId}" data-file-name="${fileName}">
            ${fileName}
          </span>
        </td>
        <td>${fileSize}</td>
        <td>${createdAt}</td>
        <td>
          <div class="file-actions">
            <button onclick="downloadOutput('${outputId}', '${fileName}')">다운로드</button>
            <button onclick="playOutput('${outputId}', '${fileName}')">재생</button>
          </div>
        </td>
      </tr>
    `;
  });
  
  html += `
      </tbody>
    </table>
    <div id="audio-player-container" class="hidden" style="margin-top: 1rem;">
      <audio id="audio-player" class="audio-player" controls></audio>
    </div>
  `;
  
  container.innerHTML = html;
  
  // 파일명 클릭 이벤트 (다운로드)
  container.querySelectorAll(".file-link").forEach((link) => {
    link.addEventListener("click", (e) => {
      const outputId = e.target.getAttribute("data-output-id");
      const fileName = e.target.getAttribute("data-file-name");
      downloadOutput(outputId, fileName);
    });
  });
};

// 파일 다운로드 함수
window.downloadOutput = (outputId, fileName) => {
  const url = `/outputs/${outputId}/download`;
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// 파일 재생 함수
window.playOutput = (outputId, fileName) => {
  const container = document.getElementById("audio-player-container");
  const player = document.getElementById("audio-player");
  
  if (container && player) {
    const url = `/outputs/${outputId}/download`;
    player.src = url;
    container.classList.remove("hidden");
    player.play().catch((err) => {
      console.error("재생 실패:", err);
      alert("오디오 재생에 실패했습니다.");
    });
  }
};

// 헬스 체크
document.getElementById("health-btn").addEventListener("click", async () => {
  const target = "health-result";
  prettyPrint(target, { status: "요청 중..." });
  try {
    const response = await fetch("/");
    const data = await response.json();
    prettyPrint(target, data);
  } catch (error) {
    prettyPrint(target, { error: error.message || error });
  }
});

// 모델 리스트 조회
const refreshModels = async () => {
  const target = "models-list";
  prettyPrint(target, { status: "요청 중..." });
  try {
    const models = await loadModels();
    renderModelsTable(models);
    prettyPrint(target, models);
  } catch (error) {
    prettyPrint(target, { error: error.message || error });
  }
};

document.getElementById("models-btn").addEventListener("click", refreshModels);
document.getElementById("models-refresh-btn").addEventListener("click", refreshModels);

// 페이지 로드 시 모델 리스트 자동 로드
loadModels();

// 추론 결과 리스트 조회
const refreshOutputs = async () => {
  const target = "outputs-list";
  prettyPrint(target, { status: "요청 중..." });
  try {
    const response = await fetch("/outputs");
    const data = await response.json();
    renderOutputsTable(data);
    prettyPrint(target, data);
  } catch (error) {
    prettyPrint(target, { error: error.message || error });
  }
};

document.getElementById("outputs-btn").addEventListener("click", refreshOutputs);
document.getElementById("outputs-refresh-btn").addEventListener("click", refreshOutputs);

// 모델 삭제
document.getElementById("delete-model-btn").addEventListener("click", async () => {
  const modelId = document.getElementById("delete-model-id").value.trim();
  const target = "models-list";
  
  if (!modelId) {
    prettyPrint(target, { error: "모델 ID를 입력해주세요." });
    return;
  }
  
  prettyPrint(target, { status: "삭제 중..." });
  try {
    const response = await fetch(`/models/${modelId}`, {
      method: "DELETE",
    });
    const data = await response.json();
    
    if (!response.ok) {
      throw data;
    }
    
    prettyPrint(target, data);
    
    // 삭제 후 리스트 새로고침
    setTimeout(() => {
      refreshModels();
    }, 500);
  } catch (error) {
    prettyPrint(target, { error: error.detail || error.message || error });
  }
});

// 출력 파일 삭제
document.getElementById("delete-output-btn").addEventListener("click", async () => {
  const outputId = document.getElementById("delete-output-id").value.trim();
  const target = "outputs-list";
  
  if (!outputId) {
    prettyPrint(target, { error: "출력 파일 ID를 입력해주세요." });
    return;
  }
  
  prettyPrint(target, { status: "삭제 중..." });
  try {
    const response = await fetch(`/outputs/${outputId}`, {
      method: "DELETE",
    });
    const data = await response.json();
    
    if (!response.ok) {
      throw data;
    }
    
    prettyPrint(target, data);
    
    // 삭제 후 리스트 새로고침
    setTimeout(() => {
      refreshOutputs();
    }, 500);
  } catch (error) {
    prettyPrint(target, { error: error.detail || error.message || error });
  }
});

// 학습 파일 업로드
document
  .getElementById("train-upload-form")
  .addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const target = "train-upload-result";

    prettyPrint(target, { status: "요청 중..." });

    try {
      const formData = new FormData();

      if (form.sample_rate.value.trim().length > 0) {
        formData.append("sample_rate", form.sample_rate.value);
      }
      if (form.total_epoch.value.trim().length > 0) {
        formData.append("total_epoch", form.total_epoch.value);
      }
      if (form.batch_size.value.trim().length > 0) {
        formData.append("batch_size", form.batch_size.value);
      }
      if (form.embedder_model.value.trim().length > 0) {
        formData.append("embedder_model", form.embedder_model.value);
      }

      // multiple 파일들 추가
      const files = form.files.files;
      if (files.length === 0) {
        throw new Error("최소 하나의 파일을 선택해주세요.");
      }
      for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
      }

      const response = await fetch("/train-files", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw data;
      }

      prettyPrint(target, data);

      // Job ID가 있으면 자동으로 상태 조회 섹션에 표시
      if (data.job_id) {
        document.querySelector('input[name="job_id"]').value = data.job_id;
        document.querySelector('select[name="queue_name"]').value = "train";
        // 자동으로 상태 조회
        setTimeout(() => {
          document.getElementById("job-status-form").dispatchEvent(
            new Event("submit", { bubbles: true, cancelable: true })
          );
        }, 500);
      }
    } catch (error) {
      prettyPrint(target, { error: error.detail || error.message || error });
    }
  });

// 추론 파일 업로드
document
  .getElementById("inference-upload-form")
  .addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const target = "inference-upload-result";

    prettyPrint(target, { status: "요청 중..." });

    try {
      const formData = new FormData();
      formData.append("target_audio", form.target_audio.files[0]);
      
      // 모델 경로: 드롭다운에서 선택한 값 또는 직접 입력한 값
      const modelPath = form.model_path.value.trim();
      if (!modelPath) {
        throw new Error("모델 경로를 입력하거나 모델을 선택해주세요.");
      }
      formData.append("model_path", modelPath);

      // 인덱스 경로: 드롭다운에서 선택한 값 또는 직접 입력한 값
      if (form.index_path.value.trim().length > 0) {
        formData.append("index_path", form.index_path.value);
      }
      if (form.output_dir.value.trim().length > 0) {
        formData.append("output_dir", form.output_dir.value);
      }
      if (form.volume_envelope.value.trim().length > 0) {
        formData.append("volume_envelope", form.volume_envelope.value);
      }
      if (form.protect.value.trim().length > 0) {
        formData.append("protect", form.protect.value);
      }
      if (form.f0_autotune.checked) {
        formData.append("f0_autotune", "true");
      }
      if (form.embedder_model.value.trim().length > 0) {
        formData.append("embedder_model", form.embedder_model.value);
      }

      const response = await fetch("/inference-files", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw data;
      }

      prettyPrint(target, data);

      // Job ID가 있으면 자동으로 상태 조회 섹션에 표시
      if (data.job_id) {
        document.querySelector('input[name="job_id"]').value = data.job_id;
        document.querySelector('select[name="queue_name"]').value = "inference";
        // 자동으로 상태 조회
        setTimeout(() => {
          document.getElementById("job-status-form").dispatchEvent(
            new Event("submit", { bubbles: true, cancelable: true })
          );
        }, 500);
      }
      
      // 추론 완료 후 결과 리스트 새로고침
      setTimeout(() => {
        refreshOutputs();
      }, 2000);
    } catch (error) {
      prettyPrint(target, { error: error.detail || error.message || error });
    }
  });

// 작업 상태 조회
let autoRefreshInterval = null;

const checkJobStatus = async (queueName, jobId) => {
  const target = "job-status-result";
  try {
    const response = await fetch(`/jobs/${queueName}/${jobId}`);
    const data = await response.json();

    if (!response.ok) {
      throw data;
    }

    // 상태에 따른 스타일링
    const status = data.status;
    const statusColors = {
      pending: "#3b82f6",
      running: "#f59e0b",
      completed: "#10b981",
      failed: "#ef4444",
    };

    const resultBox = jsonBox(target);
    resultBox.style.borderLeft = `4px solid ${statusColors[status] || "#6b7280"}`;
    prettyPrint(target, data);

    // 완료되면 자동 새로고침 중지
    if (status === "completed" || status === "failed") {
      const checkbox = document.getElementById("auto-refresh-checkbox");
      if (checkbox.checked) {
        checkbox.checked = false;
        if (autoRefreshInterval) {
          clearInterval(autoRefreshInterval);
          autoRefreshInterval = null;
        }
      }
      
      // 완료 시 추론 결과 리스트 새로고침
      if (status === "completed" && queueName === "inference") {
        setTimeout(() => {
          refreshOutputs();
        }, 1000);
      }
    }
  } catch (error) {
    prettyPrint(target, { error: error.detail || error.message || error });
  }
};

document.getElementById("job-status-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const queueName = form.queue_name.value;
  const jobId = form.job_id.value.trim();

  if (!jobId) {
    prettyPrint("job-status-result", { error: "Job ID를 입력해주세요." });
    return;
  }

  checkJobStatus(queueName, jobId);
});

// 자동 새로고침 기능
document.getElementById("auto-refresh-checkbox").addEventListener("change", (event) => {
  const checkbox = event.target;
  const form = document.getElementById("job-status-form");
  const queueName = form.queue_name.value;
  const jobId = form.job_id.value.trim();

  if (checkbox.checked) {
    if (!jobId) {
      checkbox.checked = false;
      alert("Job ID를 먼저 입력해주세요.");
      return;
    }

    // 즉시 한 번 조회
    checkJobStatus(queueName, jobId);

    // 5초마다 자동 조회
    autoRefreshInterval = setInterval(() => {
      const currentQueueName = form.queue_name.value;
      const currentJobId = form.job_id.value.trim();
      if (currentJobId) {
        checkJobStatus(currentQueueName, currentJobId);
      } else {
        checkbox.checked = false;
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
      }
    }, 5000);
  } else {
    if (autoRefreshInterval) {
      clearInterval(autoRefreshInterval);
      autoRefreshInterval = null;
    }
  }
});
