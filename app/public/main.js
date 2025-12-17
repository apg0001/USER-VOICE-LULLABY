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
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
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
let outputsCache = [];

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
  // G_와 D_로 시작하는 파일은 제외
  const validModelFiles = model.model_files.filter(f => !f.startsWith("G_") && !f.startsWith("D_"));
  if (validModelFiles.length > 0) {
    const modelFileName = validModelFiles[0];
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
document
  .getElementById("custom-pretrained-select")
  ?.addEventListener("change", (e) => {
    const pretrainedGroup = document.getElementById("pretrained-model-group");
    if (pretrainedGroup) {
      pretrainedGroup.style.display =
        e.target.value === "true" ? "block" : "none";
    }
  });

// G 사전 학습 모델 선택 시 경로 업데이트
document
  .getElementById("g-pretrained-select")
  ?.addEventListener("change", (e) => {
    const pathInput = document.getElementById("g-pretrained-path-input");
    if (pathInput && e.target.value) {
      pathInput.value = e.target.value;
    }
  });

// D 사전 학습 모델 선택 시 경로 업데이트
document
  .getElementById("d-pretrained-select")
  ?.addEventListener("change", (e) => {
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

// 페이지네이션 상태 관리
let modelsPaginationState = {
  currentPage: 1,
  itemsPerPage: 10,
  totalItems: 0,
  totalPages: 0
};

let outputsPaginationState = {
  currentPage: 1,
  itemsPerPage: 10,
  totalItems: 0,
  totalPages: 0
};

// 모델 리스트를 테이블로 표시 (페이지네이션 지원)
const renderModelsTable = (models) => {
  const container = document.getElementById("models-list-container");

  if (!models || models.length === 0) {
    container.innerHTML = '<div class="empty-state">모델이 없습니다</div>';
    return;
  }

  // 페이지네이션 계산
  modelsPaginationState.totalItems = models.length;
  modelsPaginationState.totalPages = Math.ceil(models.length / modelsPaginationState.itemsPerPage);
  if (modelsPaginationState.currentPage > modelsPaginationState.totalPages) {
    modelsPaginationState.currentPage = Math.max(1, modelsPaginationState.totalPages);
  }

  // 현재 페이지에 표시할 항목들
  const startIndex = (modelsPaginationState.currentPage - 1) * modelsPaginationState.itemsPerPage;
  const endIndex = startIndex + modelsPaginationState.itemsPerPage;
  const paginatedModels = models.slice(startIndex, endIndex);

  // let html = `
  //   <table class="data-table">
  //     <thead>
  //       <tr>
  //         <th>모델 ID</th>
  //         <th>모델 이름</th>
  //         <th>설명</th>
  //         <th>임베더</th>
  //         <th>샘플레이트</th>
  //         <th>Epoch</th>
  //         <th>보코더</th>
  //         <th>모델 파일 (절대 경로)</th>
  //         <th>인덱스 파일 (절대 경로)</th>
  //         <th>생성 시간</th>
  //         <th>작업</th>
  //       </tr>
  //     </thead>
  //     <tbody>
  // `;
  let html = `
    <table class="data-table">
      <thead>
        <tr>
          <th>모델 ID</th>
          <th>설명</th>
          <th>모델 파일 (절대 경로)</th>
          <th>인덱스 파일 (절대 경로)</th>
          <th>생성 시간</th>
          <th>작업</th>
        </tr>
      </thead>
      <tbody>
  `;

  paginatedModels.forEach((model) => {
    const modelId = encodeURIComponent(model.model_id);
    // 주석 처리된 필드들 (나중에 쉽게 복구 가능)
    // const modelName = model.model_name || "-";
    // const embedder = model.embedder_model || "-";
    // const sampleRate = model.sample_rate ? `${model.sample_rate}Hz` : "-";
    // const totalEpoch = model.total_epoch || "-";
    // const vocoder = model.vocoder || "-";

    // 절대 경로 표시 (여러 개일 경우 줄바꿈으로 표시)
    // G_와 D_로 시작하는 파일 제외
    const filteredModelFiles = model.model_files.filter(f => !f.startsWith("G_") && !f.startsWith("D_"));
    const filteredModelFilesAbsolute = (model.model_files_absolute || [])
      .filter(path => {
        const fileName = path.split(/[/\\]/).pop();
        return !fileName.startsWith("G_") && !fileName.startsWith("D_");
      });
    
    const modelFilesAbsolute =
      filteredModelFilesAbsolute.length > 0
        ? filteredModelFilesAbsolute.join("<br>")
        : filteredModelFiles.map((f) => f.split("/").pop()).join("<br>") || "-";
    const indexFilesAbsolute =
      model.index_files_absolute && model.index_files_absolute.length > 0
        ? model.index_files_absolute.join("<br>")
        : model.index_files.map((f) => f.split("/").pop()).join("<br>") || "-";

    const modelDescription = model.model_description || "-";

    // html += `
    //   <tr>
    //     <td><strong>${model.model_id}</strong></td>
    //     <!-- 주석 처리된 필드들 (나중에 쉽게 복구 가능) -->
    //     <td>${modelName}</td>
    //     <td style="max-width: 200px; word-break: break-word; white-space: pre-wrap;">${modelDescription}</td>
    //     <td>${embedder}</td>
    //     <td>${sampleRate}</td>
    //     <td>${totalEpoch}</td>
    //     <td>${vocoder}</td>
    //     <td style="font-size: 0.85rem; max-width: 300px; word-break: break-all;">${modelFilesAbsolute}</td>
    //     <td style="font-size: 0.85rem; max-width: 300px; word-break: break-all;">${indexFilesAbsolute}</td>
    //     <td>${formatDate(model.created_at)}</td>
    //     <td>
    //       <div class="file-actions">
    //         <button onclick="deleteModel('${modelId}', '${
    //   model.model_id
    // }')" style="background: #ef4444;">삭제</button>
    //       </div>
    //     </td>
    //   </tr>
    // `;
    html += `
      <tr>
        <td><strong>${model.model_id}</strong></td>
        <td style="max-width: 200px; word-break: break-word; white-space: pre-wrap;">${modelDescription}</td>
        <td style="font-size: 0.85rem; max-width: 300px; word-break: break-all;">${modelFilesAbsolute}</td>
        <td style="font-size: 0.85rem; max-width: 300px; word-break: break-all;">${indexFilesAbsolute}</td>
        <td>${formatDate(model.created_at)}</td>
        <td>
          <div class="file-actions">
            <button onclick="deleteModel('${modelId}', '${
      model.model_id
    }')" style="background: #ef4444;">삭제</button>
          </div>
        </td>
      </tr>
    `;
  });

  html += `
      </tbody>
    </table>
  `;

  // 페이지네이션 컨트롤 추가
  html += `
    <div class="pagination-controls" style="margin-top: 1rem; display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
      <div style="display: flex; align-items: center; gap: 0.5rem;">
        <label style="margin: 0;">한 페이지에 표시할 개수:</label>
        <select id="models-items-per-page" style="padding: 0.25rem 0.5rem;">
          <option value="5" ${modelsPaginationState.itemsPerPage === 5 ? 'selected' : ''}>5</option>
          <option value="10" ${modelsPaginationState.itemsPerPage === 10 ? 'selected' : ''}>10</option>
          <option value="20" ${modelsPaginationState.itemsPerPage === 20 ? 'selected' : ''}>20</option>
          <option value="50" ${modelsPaginationState.itemsPerPage === 50 ? 'selected' : ''}>50</option>
          <option value="100" ${modelsPaginationState.itemsPerPage === 100 ? 'selected' : ''}>100</option>
        </select>
      </div>
      <div style="display: flex; align-items: center; gap: 0.5rem;">
        <button onclick="modelsPaginationPrev()" ${modelsPaginationState.currentPage === 1 ? 'disabled' : ''} style="padding: 0.25rem 0.5rem;">이전</button>
        <span>페이지 ${modelsPaginationState.currentPage} / ${modelsPaginationState.totalPages} (총 ${modelsPaginationState.totalItems}개)</span>
        <button onclick="modelsPaginationNext()" ${modelsPaginationState.currentPage >= modelsPaginationState.totalPages ? 'disabled' : ''} style="padding: 0.25rem 0.5rem;">다음</button>
      </div>
    </div>
  `;

  container.innerHTML = html;

  // 페이지네이션 이벤트 리스너
  const itemsPerPageSelect = document.getElementById("models-items-per-page");
  if (itemsPerPageSelect) {
    itemsPerPageSelect.addEventListener("change", (e) => {
      modelsPaginationState.itemsPerPage = parseInt(e.target.value);
      modelsPaginationState.currentPage = 1;
      renderModelsTable(modelsCache);
    });
  }
};

// 모델 리스트 페이지네이션 함수
window.modelsPaginationPrev = () => {
  if (modelsPaginationState.currentPage > 1) {
    modelsPaginationState.currentPage--;
    renderModelsTable(modelsCache);
  }
};

window.modelsPaginationNext = () => {
  if (modelsPaginationState.currentPage < modelsPaginationState.totalPages) {
    modelsPaginationState.currentPage++;
    renderModelsTable(modelsCache);
  }
};

// 추론 결과 리스트를 테이블로 표시 (페이지네이션 지원)
const renderOutputsTable = (outputs) => {
  const container = document.getElementById("outputs-list-container");

  if (!outputs || outputs.length === 0) {
    container.innerHTML = '<div class="empty-state">추론 결과가 없습니다</div>';
    return;
  }

  // 페이지네이션 계산
  outputsPaginationState.totalItems = outputs.length;
  outputsPaginationState.totalPages = Math.ceil(outputs.length / outputsPaginationState.itemsPerPage);
  if (outputsPaginationState.currentPage > outputsPaginationState.totalPages) {
    outputsPaginationState.currentPage = Math.max(1, outputsPaginationState.totalPages);
  }

  // 현재 페이지에 표시할 항목들
  const startIndex = (outputsPaginationState.currentPage - 1) * outputsPaginationState.itemsPerPage;
  const endIndex = startIndex + outputsPaginationState.itemsPerPage;
  const paginatedOutputs = outputs.slice(startIndex, endIndex);

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

  paginatedOutputs.forEach((output) => {
    const fileName = output.output_id;
    const fileSize = formatBytes(output.file_size);
    const createdAt = formatDate(output.created_at);
    const outputId = encodeURIComponent(output.output_id);
    const rowId = `output-row-${outputId.replace(/[^a-zA-Z0-9]/g, '_')}`;

    html += `
      <tr id="${rowId}">
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
            <button onclick="playOutput('${outputId}', '${fileName}', '${rowId}')">재생</button>
            <button onclick="deleteOutput('${outputId}', '${fileName}')" style="background: #ef4444;">삭제</button>
          </div>
        </td>
      </tr>
      <tr id="${rowId}-player-row" class="audio-player-row hidden">
        <td colspan="4" style="padding: 0.5rem 0.75rem; background: #1a1f2e;">
          <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
            <span style="font-size: 0.85rem; color: #a0aec0; font-weight: 500;">재생 중: ${fileName}</span>
          </div>
          <audio id="audio-player-${outputId.replace(/[^a-zA-Z0-9]/g, '_')}" class="audio-player" controls style="width: 100%;"></audio>
        </td>
      </tr>
    `;
  });

  html += `
      </tbody>
    </table>
  `;

  // 페이지네이션 컨트롤 추가
  html += `
    <div class="pagination-controls" style="margin-top: 1rem; display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
      <div style="display: flex; align-items: center; gap: 0.5rem;">
        <label style="margin: 0;">한 페이지에 표시할 개수:</label>
        <select id="outputs-items-per-page" style="padding: 0.25rem 0.5rem;">
          <option value="5" ${outputsPaginationState.itemsPerPage === 5 ? 'selected' : ''}>5</option>
          <option value="10" ${outputsPaginationState.itemsPerPage === 10 ? 'selected' : ''}>10</option>
          <option value="20" ${outputsPaginationState.itemsPerPage === 20 ? 'selected' : ''}>20</option>
          <option value="50" ${outputsPaginationState.itemsPerPage === 50 ? 'selected' : ''}>50</option>
          <option value="100" ${outputsPaginationState.itemsPerPage === 100 ? 'selected' : ''}>100</option>
        </select>
      </div>
      <div style="display: flex; align-items: center; gap: 0.5rem;">
        <button onclick="outputsPaginationPrev()" ${outputsPaginationState.currentPage === 1 ? 'disabled' : ''} style="padding: 0.25rem 0.5rem;">이전</button>
        <span>페이지 ${outputsPaginationState.currentPage} / ${outputsPaginationState.totalPages} (총 ${outputsPaginationState.totalItems}개)</span>
        <button onclick="outputsPaginationNext()" ${outputsPaginationState.currentPage >= outputsPaginationState.totalPages ? 'disabled' : ''} style="padding: 0.25rem 0.5rem;">다음</button>
      </div>
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

  // 페이지네이션 이벤트 리스너
  const itemsPerPageSelect = document.getElementById("outputs-items-per-page");
  if (itemsPerPageSelect) {
    itemsPerPageSelect.addEventListener("change", (e) => {
      outputsPaginationState.itemsPerPage = parseInt(e.target.value);
      outputsPaginationState.currentPage = 1;
      renderOutputsTable(outputsCache);
    });
  }
};

// 추론 결과 리스트 페이지네이션 함수
window.outputsPaginationPrev = () => {
  if (outputsPaginationState.currentPage > 1) {
    outputsPaginationState.currentPage--;
    renderOutputsTable(outputsCache);
  }
};

window.outputsPaginationNext = () => {
  if (outputsPaginationState.currentPage < outputsPaginationState.totalPages) {
    outputsPaginationState.currentPage++;
    renderOutputsTable(outputsCache);
  }
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
window.playOutput = (outputId, fileName, rowId) => {
  // 모든 재생 바 숨기기
  document.querySelectorAll('.audio-player-row').forEach(row => {
    row.classList.add('hidden');
  });
  
  // 모든 오디오 플레이어 일시정지
  document.querySelectorAll('.audio-player').forEach(player => {
    player.pause();
    player.currentTime = 0;
  });

  // 해당 행의 재생 바 표시
  const playerRow = document.getElementById(`${rowId}-player-row`);
  const playerId = `audio-player-${outputId.replace(/[^a-zA-Z0-9]/g, '_')}`;
  const player = document.getElementById(playerId);

  if (playerRow && player) {
    const url = `/outputs/${outputId}/download`;
    player.src = url;
    playerRow.classList.remove("hidden");
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
document
  .getElementById("models-refresh-btn")
  .addEventListener("click", refreshModels);

// 페이지 로드 시 모델 리스트 자동 로드
loadModels();

// 추론 결과 리스트 조회
const refreshOutputs = async () => {
  try {
    const response = await fetch("/outputs");
    if (!response.ok) throw new Error("추론 결과 조회 실패");
    const outputs = await response.json();
    outputsCache = outputs;
    outputsPaginationState.currentPage = 1; // 새로고침 시 첫 페이지로
    renderOutputsTable(outputs);
  } catch (error) {
    alert(`추론 결과 조회 실패: ${error.message || error}`);
  }
};

document
  .getElementById("outputs-btn")
  .addEventListener("click", refreshOutputs);
document
  .getElementById("outputs-refresh-btn")
  .addEventListener("click", refreshOutputs);

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
        formData.append(
          "overtraining_detector",
          form.overtraining_detector.value === "true"
        );
      }
      if (form.custom_pretrained.value.trim().length > 0) {
        formData.append(
          "custom_pretrained",
          form.custom_pretrained.value === "true"
        );
      }
      // 사전 학습 모델 경로: 드롭다운에서 선택한 값 또는 직접 입력한 값
      const gPretrainedPath =
        form.g_pretrained_select?.value || form.g_pretrained_path?.value || "";
      if (gPretrainedPath.trim().length > 0) {
        formData.append("g_pretrained_path", gPretrainedPath);
      }
      const dPretrainedPath =
        form.d_pretrained_select?.value || form.d_pretrained_path?.value || "";
      if (dPretrainedPath.trim().length > 0) {
        formData.append("d_pretrained_path", dPretrainedPath);
      }
      // 모델 설명 추가
      if (
        form.model_description &&
        form.model_description.value.trim().length > 0
      ) {
        formData.append(
          "model_description",
          form.model_description.value.trim()
        );
      }

      // multiple 파일들 추가
      const files = form.files.files;
      if (files.length === 0) {
        throw new Error("최소 하나의 파일을 선택해주세요.");
      }
      for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
      }

      const response = await fetch("/train", {
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
      if (form.pitch.value.trim().length > 0) {
        formData.append("pitch", form.pitch.value);
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
        formData.append(
          "f0_autotune_strength",
          form.f0_autotune_strength.value
        );
      }
      if (form.index_rate.value.trim().length > 0) {
        formData.append("index_rate", form.index_rate.value);
      }
      if (form.clean_audio && form.clean_audio.checked) {
        formData.append("clean_audio", "true");
      }
      if (form.clean_strength && form.clean_strength.value.trim().length > 0) {
        formData.append("clean_strength", form.clean_strength.value);
      }
      if (form.reverb && form.reverb.checked) {
        formData.append("reverb", "true");
      }
      if (
        form.reverb_room_size &&
        form.reverb_room_size.value.trim().length > 0
      ) {
        formData.append("reverb_room_size", form.reverb_room_size.value);
      }
      if (form.reverb_damping && form.reverb_damping.value.trim().length > 0) {
        formData.append("reverb_damping", form.reverb_damping.value);
      }
      if (
        form.reverb_wet_gain &&
        form.reverb_wet_gain.value.trim().length > 0
      ) {
        formData.append("reverb_wet_gain", form.reverb_wet_gain.value);
      }
      if (
        form.reverb_dry_gain &&
        form.reverb_dry_gain.value.trim().length > 0
      ) {
        formData.append("reverb_dry_gain", form.reverb_dry_gain.value);
      }
      if (form.reverb_width && form.reverb_width.value.trim().length > 0) {
        formData.append("reverb_width", form.reverb_width.value);
      }
      if (
        form.reverb_freeze_mode &&
        form.reverb_freeze_mode.value.trim().length > 0
      ) {
        formData.append("reverb_freeze_mode", form.reverb_freeze_mode.value);
      }
      if (form.embedder_model.value.trim().length > 0) {
        formData.append("embedder_model", form.embedder_model.value);
      }
      if (form.formant_shifting && form.formant_shifting.checked) {
        formData.append("formant_shifting", "true");
      }
      if (
        form.formant_qfrency &&
        form.formant_qfrency.value.trim().length > 0
      ) {
        formData.append("formant_qfrency", form.formant_qfrency.value);
      }
      if (
        form.formant_timbre &&
        form.formant_timbre.value.trim().length > 0
      ) {
        formData.append("formant_timbre", form.formant_timbre.value);
      }

      const response = await fetch("/inference", {
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
          ${queueName === "train" ? "<th>모델 ID</th><th>모델 설명</th>" : ""}
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
      cancelled: "#6b7280",
    };

    // 진행률 표시 (안전하게 처리)
    let progress = "-";
    if (job.progress && typeof job.progress === "object") {
      const currentEpoch = job.progress.current_epoch ?? 0;
      const totalEpoch = job.progress.total_epoch ?? 0;
      const progressPercent = job.progress.progress_percent ?? 0;
      if (totalEpoch > 0) {
        progress = `${currentEpoch}/${totalEpoch} (${progressPercent}%)`;
      }
    }

    const result = job.result
      ? JSON.stringify(job.result).substring(0, 100) +
        (JSON.stringify(job.result).length > 100 ? "..." : "")
      : "-";
    const error = job.error || "-";

    // 모델 ID와 설명 (학습 작업의 경우)
    const modelId =
      queueName === "train"
        ? job.model_id
          ? job.model_id.substring(0, 8) + "..."
          : "-"
        : "";
    const modelDesc = queueName === "train" ? job.model_description || "-" : "";

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
        ${
          queueName === "train"
            ? `<td>${modelId}</td><td style="font-size: 0.8rem; max-width: 200px; word-break: break-all; white-space: pre-wrap;">${modelDesc}</td>`
            : ""
        }
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
document
  .getElementById("jobs-refresh-btn")
  .addEventListener("click", refreshJobsList);

// 큐 선택 변경 시 자동 새로고침
document
  .getElementById("queue-select")
  .addEventListener("change", refreshJobsList);

// 자동 새로고침 기능
document
  .getElementById("auto-refresh-checkbox")
  .addEventListener("change", (event) => {
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
  if (
    !confirm(`작업을 취소하시겠습니까? (Job ID: ${jobId.substring(0, 8)}...)`)
  ) {
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

// Reverb 체크박스에 따라 파라미터 입력 필드 표시/숨김
const reverbCheckbox = document.querySelector('input[name="reverb"]');
const reverbParams = document.getElementById("reverb-params");

if (reverbCheckbox && reverbParams) {
  reverbCheckbox.addEventListener("change", () => {
    reverbParams.style.display = reverbCheckbox.checked ? "block" : "none";
  });
}

// Formant Shifting 체크박스에 따라 파라미터 입력 필드 표시/숨김
const formantCheckbox = document.querySelector('input[name="formant_shifting"]');
const formantParams = document.getElementById("formant-params");

if (formantCheckbox && formantParams) {
  formantCheckbox.addEventListener("change", () => {
    formantParams.style.display = formantCheckbox.checked ? "block" : "none";
  });
}

// 페이지 로드 시 초기 조회
refreshJobsList();
