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

const handleRequest = async (url, payload, target) => {
  prettyPrint(target, { status: "요청 중..." });
  try {
    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(prunePayload(payload)),
    });
    const data = await response.json();
    if (!response.ok) {
      throw data;
    }
    prettyPrint(target, data);
  } catch (error) {
    prettyPrint(target, { error });
  }
};

const handleFormDataRequest = async (url, formData, target) => {
  const box = jsonBox(target);
  prettyPrint(target, { status: "요청 중..." });

  try {
    const response = await fetch(url, {
      method: "POST",
      body: formData, // 헤더에 Content-Type 수동 지정 X (브라우저가 처리)
    });

    let data;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    if (!response.ok) {
      throw data;
    }
    prettyPrint(target, data);
  } catch (error) {
    prettyPrint(target, { error });
  }
};

document.getElementById("health-btn").addEventListener("click", async () => {
  const target = "health-result";
  prettyPrint(target, { status: "요청 중..." });
  try {
    const response = await fetch("/");
    const data = await response.json();
    prettyPrint(target, data);
  } catch (error) {
    prettyPrint(target, { error });
  }
});

document.getElementById("train-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  handleRequest(
    "/train",
    {
      model_name: form.model_name.value,
      dataset_path: form.dataset_path.value,
      sample_rate: parseNumber(form.sample_rate.value),
      total_epoch: parseNumber(form.total_epoch.value),
      batch_size: parseNumber(form.batch_size.value),
    },
    "train-result"
  );
});

document
  .getElementById("inference-form")
  .addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    handleRequest(
      "/inference",
      {
        input_audio_path: form.input_audio_path.value,
        model_path: form.model_path.value,
        index_path: form.index_path.value,
        output_dir: form.output_dir.value || "outputs",
      },
      "inference-result"
    );
  });

  document
  .getElementById("train-upload-form")
  .addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;

    const formData = new FormData();
    formData.append("model_name", form.model_name.value);

    if (form.sample_rate.value.trim().length > 0) {
      formData.append("sample_rate", form.sample_rate.value);
    }
    if (form.total_epoch.value.trim().length > 0) {
      formData.append("total_epoch", form.total_epoch.value);
    }
    if (form.batch_size.value.trim().length > 0) {
      formData.append("batch_size", form.batch_size.value);
    }

    // multiple 파일들 추가
    const files = form.files.files;
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]); // FastAPI: files: List[UploadFile] = File(...)
    }

    handleFormDataRequest("/train-files", formData, "train-upload-result");
  });

document
  .getElementById("inference-upload-form")
  .addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;

    const formData = new FormData();
    formData.append("target_audio", form.target_audio.files[0]); // FastAPI: target_audio: UploadFile = File(...)
    formData.append("model_path", form.model_path.value);

    if (form.index_path.value.trim().length > 0) {
      formData.append("index_path", form.index_path.value);
    }
    if (form.output_dir.value.trim().length > 0) {
      formData.append("output_dir", form.output_dir.value);
    }

    handleFormDataRequest("/inference-files", formData, "inference-upload-result");
  });