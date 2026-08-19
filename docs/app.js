const runtimeButtons = document.querySelectorAll("[data-runtime]");
const modeButtons = document.querySelectorAll("[data-mode]");
const lengthButtons = document.querySelectorAll("[data-length]");
const card = document.querySelector(".runtime-card");
const cosine = document.querySelector("#cosine");
const result = document.querySelector("#result");
const note = document.querySelector("#result-note");

let runtime = "official";
let mode = "isolated";

const outcomes = {
  official: {
    isolated: ["1.0000", "Stable", "One request at a time produces the isolated reference embedding."],
    list: ["1.0000", "Stable", "A client-supplied input list stays correct—and misses the router bug."],
    concurrent: ["0.1586", "Corrupted", "Independent equal-length requests are coalesced, triggering the missing causal mask."],
  },
  patched: {
    isolated: ["1.0000", "Stable", "One request at a time produces the isolated reference embedding."],
    list: ["1.0000", "Stable", "Client list batching remains consistent."],
    concurrent: ["1.0000", "Stable", "PR #883 restores the causal mask for equal-length backend batches."],
  },
};

function renderOutcome() {
  const [score, label, description] = outcomes[runtime][mode];
  const failed = label === "Corrupted";
  cosine.textContent = score;
  result.textContent = label;
  result.className = failed ? "result-fail" : "result-pass";
  note.textContent = description;
  card.classList.toggle("corrupt", failed);
  card.classList.toggle("isolated", mode === "isolated");
  card.classList.toggle("list", mode === "list");
}

runtimeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    runtime = button.dataset.runtime;
    runtimeButtons.forEach((item) => item.classList.toggle("active", item === button));
    renderOutcome();
  });
});

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    mode = button.dataset.mode;
    modeButtons.forEach((item) => item.setAttribute("aria-selected", String(item === button)));
    renderOutcome();
  });
});

lengthButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const mixed = button.dataset.length === "mixed";
    lengthButtons.forEach((item) => item.classList.toggle("active", item === button));
    const row = document.querySelector(".row-b");
    row.innerHTML = mixed
      ? "<span>ocean</span><span>currents</span><span class='pad'>PAD</span><span class='pad'>PAD</span>"
      : "<span>ocean</span><span>currents</span><span>shape</span><span>weather</span>";
    document.querySelector("#padding-state").textContent = mixed ? "Required" : "Not needed";
    const causal = document.querySelector("#causal-state");
    causal.textContent = mixed ? "Applied" : "Skipped by bug";
    causal.classList.toggle("danger", !mixed);
  });
});

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      const previous = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(() => { button.textContent = previous; }, 1400);
    } catch {
      button.textContent = "Select command";
    }
  });
});

renderOutcome();
