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

