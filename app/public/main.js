const headers = { "Content-Type": "application/json" };

const jsonBox = (id) => document.getElementById(id);

const prettyPrint = (target, data) => {
  const box = jsonBox(target);
  if (!box) {
    console.warn(`Element with id "${target}" not found`);
    return;
  }
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
    
    // 사전 학습 모델 드롭다운 업데이트
    updatePretrainedModelDropdowns();
    
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

// 사전 학습 모델 드롭다운 업데이트
const updatePretrainedModelDropdowns = () => {
  const gSelect = document.getElementById("g-pretrained-select");
  const dSelect = document.getElementById("d-pretrained-select");
  
  if (!gSelect || !dSelect) return;
  
  // 기존 옵션 초기화 (첫 번째 옵션 제외)
  gSelect.innerHTML = '<option value="">모델을 선택하세요</option>';
  dSelect.innerHTML = '<option value="">모델을 선택하세요</option>';
  
  // 모델 리스트를 순회하며 .pth 파일들을 드롭다운에 추가
  modelsCache.forEach((model) => {
    // 절대 경로 사용
    if (model.model_files_absolute && model.model_files_absolute.length > 0) {
      model.model_files_absolute.forEach((absolutePath) => {
        const fileName = absolutePath.split(/[/\\]/).pop(); // 파일명만 추출
        // G 모델 파일들 (G_로 시작하는 파일)
        if (fileName.startsWith("G_") && fileName.endsWith(".pth")) {
          const option = document.createElement("option");
          option.value = absolutePath; // 절대 경로 사용
          option.textContent = `${model.model_id} - ${fileName}`;
          gSelect.appendChild(option);
        }
        // D 모델 파일들 (D_로 시작하는 파일)
        if (fileName.startsWith("D_") && fileName.endsWith(".pth")) {
          const option = document.createElement("option");
          option.value = absolutePath; // 절대 경로 사용
          option.textContent = `${model.model_id} - ${fileName}`;
          dSelect.appendChild(option);
        }
      });
    }
  });
};

// Custom Pretrained 선택 시 사전 학습 모델 그룹 표시/숨김
document.getElementById("custom-pretrained-select")?.addEventListener("change", (e) => {
  const pretrainedGroup = document.getElementById("pretrained-model-group");
  if (pretrainedGroup) {
    pretrainedGroup.style.display = e.target.value === "true" ? "block" : "none";
  }
});

// G 사전 학습 모델 선택 시 경로 업데이트
document.getElementById("g-pretrained-select")?.addEventListener("change", (e) => {
  const pathInput = document.getElementById("g-pretrained-path-input");
  if (pathInput && e.target.value) {
    pathInput.value = e.target.value;
  }
});

// D 사전 학습 모델 선택 시 경로 업데이트
document.getElementById("d-pretrained-select")?.addEventListener("change", (e) => {
  const pathInput = document.getElementById("d-pretrained-path-input");
  if (pathInput && e.target.value) {
    pathInput.value = e.target.value;
  }
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
    container.innerHTML = '<div class="empty-state">모델이 없습니다</div>';
    return;
  }
  
  let html = `
    <table class="data-table">
      <thead>
        <tr>
          <th>모델 ID</th>
          <th>모델 이름</th>
          <th>임베더</th>
          <th>샘플레이트</th>
          <th>Epoch</th>
          <th>보코더</th>
          <th>모델 파일 (절대 경로)</th>
          <th>인덱스 파일 (절대 경로)</th>
          <th>생성 시간</th>
          <th>작업</th>
        </tr>
      </thead>
      <tbody>
  `;
  
  models.forEach((model) => {
    const modelId = encodeURIComponent(model.model_id);
    const modelName = model.model_name || "-";
    const embedder = model.embedder_model || "-";
    const sampleRate = model.sample_rate ? `${model.sample_rate}Hz` : "-";
    const totalEpoch = model.total_epoch || "-";
    const vocoder = model.vocoder || "-";
    
    // 절대 경로 표시 (없으면 상대 경로 파일명)
    const modelFilesAbsolute = model.model_files_absolute && model.model_files_absolute.length > 0
      ? model.model_files_absolute.join(", ")
      : model.model_files.map(f => f.split("/").pop()).join(", ") || "-";
    const indexFilesAbsolute = model.index_files_absolute && model.index_files_absolute.length > 0
      ? model.index_files_absolute.join(", ")
      : model.index_files.map(f => f.split("/").pop()).join(", ") || "-";
    
    html += `
      <tr>
        <td><strong>${model.model_id}</strong></td>
        <td>${modelName}</td>
        <td>${embedder}</td>
        <td>${sampleRate}</td>
        <td>${totalEpoch}</td>
        <td>${vocoder}</td>
        <td style="font-size: 0.85rem; max-width: 300px; word-break: break-all;">${modelFilesAbsolute}</td>
        <td style="font-size: 0.85rem; max-width: 300px; word-break: break-all;">${indexFilesAbsolute}</td>
        <td>${formatDate(model.created_at)}</td>
        <td>
          <div class="file-actions">
            <button onclick="deleteModel('${modelId}', '${model.model_id}')" style="background: #ef4444;">삭제</button>
          </div>
        </td>
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
    container.innerHTML = '<div class="empty-state">추론 결과가 없습니다</div>';
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
            <button onclick="deleteOutput('${outputId}', '${fileName}')" style="background: #ef4444;">삭제</button>
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

// 모델 삭제 함수
window.deleteModel = async (modelId, modelName) => {
  if (!confirm(`모델 "${modelName}"을(를) 삭제하시겠습니까?`)) {
    return;
  }
  
  try {
    const response = await fetch(`/models/${modelId}`, {
      method: "DELETE",
    });
    
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || "모델 삭제 실패");
    }
    
    alert("모델이 삭제되었습니다.");
    
    // 삭제 후 리스트 새로고침
    setTimeout(() => {
      refreshModels();
      // 사전 학습 모델 드롭다운도 업데이트
      updatePretrainedModelDropdowns();
    }, 500);
  } catch (error) {
    alert(`모델 삭제 실패: ${error.message || error}`);
  }
};

// 출력 파일 삭제 함수
window.deleteOutput = async (outputId, fileName) => {
  if (!confirm(`파일 "${fileName}"을(를) 삭제하시겠습니까?`)) {
    return;
  }
  
  try {
    const response = await fetch(`/outputs/${outputId}`, {
      method: "DELETE",
    });
    
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || "출력 파일 삭제 실패");
    }
    
    alert("출력 파일이 삭제되었습니다.");
    
    // 삭제 후 리스트 새로고침
    setTimeout(() => {
      refreshOutputs();
    }, 500);
  } catch (error) {
    alert(`출력 파일 삭제 실패: ${error.message || error}`);
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
  try {
    const models = await loadModels();
    renderModelsTable(models);
  } catch (error) {
    alert(`모델 리스트 조회 실패: ${error.message || error}`);
  }
};

document.getElementById("models-btn").addEventListener("click", refreshModels);
document.getElementById("models-refresh-btn").addEventListener("click", refreshModels);

// 페이지 로드 시 모델 리스트 자동 로드
loadModels();

// 추론 결과 리스트 조회
const refreshOutputs = async () => {
  try {
    const response = await fetch("/outputs");
    if (!response.ok) throw new Error("추론 결과 조회 실패");
    const outputs = await response.json();
    renderOutputsTable(outputs);
  } catch (error) {
    alert(`추론 결과 조회 실패: ${error.message || error}`);
  }
};

document.getElementById("outputs-btn").addEventListener("click", refreshOutputs);
document.getElementById("outputs-refresh-btn").addEventListener("click", refreshOutputs);


// 학습 파일 업로드
document
  .getElementById("train-upload-form")
  .addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;

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
      if (form.vocoder.value.trim().length > 0) {
        formData.append("vocoder", form.vocoder.value);
      }
      if (form.overtraining_detector.value.trim().length > 0) {
        formData.append("overtraining_detector", form.overtraining_detector.value === "true");
      }
      if (form.custom_pretrained.value.trim().length > 0) {
        formData.append("custom_pretrained", form.custom_pretrained.value === "true");
      }
      // 사전 학습 모델 경로: 드롭다운에서 선택한 값 또는 직접 입력한 값
      const gPretrainedPath = form.g_pretrained_select?.value || form.g_pretrained_path?.value || "";
      if (gPretrainedPath.trim().length > 0) {
        formData.append("g_pretrained_path", gPretrainedPath);
      }
      const dPretrainedPath = form.d_pretrained_select?.value || form.d_pretrained_path?.value || "";
      if (dPretrainedPath.trim().length > 0) {
        formData.append("d_pretrained_path", dPretrainedPath);
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

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "학습 요청 실패");
      }

      const data = await response.json();
      alert(`학습 작업이 등록되었습니다. Job ID: ${data.job_id}`);

      // Job ID가 있으면 자동으로 작업 리스트 새로고침
      if (data.job_id) {
        document.getElementById("queue-select").value = "train";
        setTimeout(() => {
          refreshJobsList();
        }, 500);
      }
    } catch (error) {
      alert(`학습 요청 실패: ${error.message || error}`);
    }
  });

// 추론 파일 업로드
document
  .getElementById("inference-upload-form")
  .addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;

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
      if (form.f0_autotune_strength.value.trim().length > 0) {
        formData.append("f0_autotune_strength", form.f0_autotune_strength.value);
      }
      if (form.index_rate.value.trim().length > 0) {
        formData.append("index_rate", form.index_rate.value);
      }
      if (form.embedder_model.value.trim().length > 0) {
        formData.append("embedder_model", form.embedder_model.value);
      }

      const response = await fetch("/inference-files", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "추론 요청 실패");
      }

      const data = await response.json();
      alert(`추론 작업이 등록되었습니다. Job ID: ${data.job_id}`);

      // Job ID가 있으면 자동으로 작업 리스트 새로고침
      if (data.job_id) {
        document.getElementById("queue-select").value = "inference";
        setTimeout(() => {
          refreshJobsList();
        }, 500);
      }
      
      // 추론 완료 후 결과 리스트 새로고침
      setTimeout(() => {
        refreshOutputs();
      }, 2000);
    } catch (error) {
      alert(`추론 요청 실패: ${error.message || error}`);
    }
  });

// 작업 리스트를 테이블로 표시
const renderJobsTable = (jobs) => {
  const container = document.getElementById("jobs-list-container");
  const queueName = document.getElementById("queue-select").value;
  
  if (!jobs || jobs.length === 0) {
    container.innerHTML = '<div class="empty-state">작업이 없습니다</div>';
    return;
  }
  
  let html = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Job ID</th>
          <th>상태</th>
          <th>진행률</th>
          <th>생성 시간</th>
          <th>시작 시간</th>
          <th>완료 시간</th>
          <th>결과</th>
          <th>오류</th>
          <th>작업</th>
        </tr>
      </thead>
      <tbody>
  `;
  
  jobs.forEach((job) => {
    const status = job.status;
    const statusClass = `status-${status}`;
    const statusColors = {
      pending: "#3b82f6",
      running: "#f59e0b",
      completed: "#10b981",
      failed: "#ef4444",
    };
    
    const progress = job.progress 
      ? `${job.progress.current_epoch}/${job.progress.total_epoch} (${job.progress.progress_percent}%)`
      : "-";
    
    const result = job.result ? JSON.stringify(job.result).substring(0, 100) + (JSON.stringify(job.result).length > 100 ? "..." : "") : "-";
    const error = job.error || "-";
    
    // 취소 가능한 상태인지 확인 (pending 또는 running)
    const canCancel = status === "pending" || status === "running";
    const cancelButton = canCancel 
      ? `<button class="btn btn-danger" onclick="cancelJob('${queueName}', '${job.job_id}')" style="padding: 4px 8px; font-size: 0.8rem;">취소</button>`
      : "-";
    
    html += `
      <tr style="border-left: 4px solid ${statusColors[status] || "#6b7280"}">
        <td><strong>${job.job_id.substring(0, 8)}...</strong></td>
        <td><span class="status-badge ${statusClass}">${status}</span></td>
        <td>${progress}</td>
        <td>${formatDate(job.created_at)}</td>
        <td>${formatDate(job.started_at)}</td>
        <td>${formatDate(job.completed_at)}</td>
        <td style="font-size: 0.8rem; max-width: 200px; word-break: break-all;">${result}</td>
        <td style="font-size: 0.8rem; max-width: 200px; word-break: break-all; color: #ef4444;">${error}</td>
        <td>${cancelButton}</td>
      </tr>
    `;
  });
  
  html += `
      </tbody>
    </table>
  `;
  
  container.innerHTML = html;
};

// 작업 리스트 조회
let autoRefreshInterval = null;

const refreshJobsList = async () => {
  const queueName = document.getElementById("queue-select").value;
  
  try {
    const response = await fetch(`/jobs/${queueName}`);
    if (!response.ok) {
      throw new Error("작업 리스트 조회 실패");
    }
    const jobs = await response.json();
    renderJobsTable(jobs);
  } catch (error) {
    alert(`작업 리스트 조회 실패: ${error.message || error}`);
  }
};

// 새로고침 버튼
document.getElementById("jobs-refresh-btn").addEventListener("click", refreshJobsList);

// 큐 선택 변경 시 자동 새로고침
document.getElementById("queue-select").addEventListener("change", refreshJobsList);

// 자동 새로고침 기능
document.getElementById("auto-refresh-checkbox").addEventListener("change", (event) => {
  const checkbox = event.target;
  
  if (checkbox.checked) {
    // 즉시 한 번 조회
    refreshJobsList();
    
    // 5초마다 자동 조회
    autoRefreshInterval = setInterval(() => {
      refreshJobsList();
    }, 5000);
  } else {
    if (autoRefreshInterval) {
      clearInterval(autoRefreshInterval);
      autoRefreshInterval = null;
    }
  }
});

// 작업 취소 함수
window.cancelJob = async (queueName, jobId) => {
  if (!confirm(`작업을 취소하시겠습니까? (Job ID: ${jobId.substring(0, 8)}...)`)) {
    return;
  }
  
  try {
    const response = await fetch(`/jobs/${queueName}/${jobId}`, {
      method: "DELETE",
    });
    
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || "작업 취소 실패");
    }
    
    alert("작업이 취소되었습니다.");
    
    // 취소 후 리스트 새로고침
    setTimeout(() => {
      refreshJobsList();
    }, 500);
  } catch (error) {
    alert(`작업 취소 실패: ${error.message || error}`);
  }
};

// 페이지 로드 시 초기 조회
refreshJobsList();
