const headers = { "Content-Type": "application/json" };

const jsonBox = (id) => document.getElementById(id);

const prettyPrint = (target, data) => {
  const box = jsonBox(target);
  box.textContent = JSON.stringify(data, null, 2);
};

const prunePayload = (payload) =>
  Object.fromEntries(
    Object.entries(payload).filter(
      ([, value]) => value !== undefined && value !== null && value !== ""
    )
  );

const parseNumber = (value) =>
  value && value.trim().length > 0 ? Number(value) : undefined;

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
document.getElementById("models-btn").addEventListener("click", async () => {
  const target = "models-list";
  prettyPrint(target, { status: "요청 중..." });
  try {
    const response = await fetch("/models");
    const data = await response.json();
    prettyPrint(target, data);
  } catch (error) {
    prettyPrint(target, { error: error.message || error });
  }
});

// 추론 결과 리스트 조회
document.getElementById("outputs-btn").addEventListener("click", async () => {
  const target = "outputs-list";
  prettyPrint(target, { status: "요청 중..." });
  try {
    const response = await fetch("/outputs");
    const data = await response.json();
    prettyPrint(target, data);
  } catch (error) {
    prettyPrint(target, { error: error.message || error });
  }
});

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
      document.getElementById("models-btn").click();
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
      document.getElementById("outputs-btn").click();
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
      formData.append("model_path", form.model_path.value);

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
