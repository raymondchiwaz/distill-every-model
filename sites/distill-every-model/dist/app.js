const walkthrough = document.querySelector("[data-walkthrough]");
const tabs = [...walkthrough.querySelectorAll("[data-select]")];
const tracks = [...walkthrough.querySelectorAll("[data-track]")].map((panel, index) => ({
  panel,
  tab: tabs[index],
  steps: [...panel.querySelectorAll(".step")],
  buttons: [...panel.querySelectorAll("[data-step]")],
  index: 0,
}));
const previous = walkthrough.querySelector("[data-previous]");
const next = walkthrough.querySelector("[data-next]");
const done = walkthrough.querySelector("[data-done]");
const stepStatus = walkthrough.querySelector("[data-step-status]");
let activeTrack = tracks[0];

function showStep(index, moveFocus = false) {
  const { steps, buttons } = activeTrack;
  activeTrack.index = index;
  steps.forEach((step, position) => {
    step.hidden = position !== index;
    if (position === index) {
      buttons[position].setAttribute("aria-current", "step");
    } else {
      buttons[position].removeAttribute("aria-current");
    }
  });
  previous.disabled = index === 0;
  next.hidden = index === steps.length - 1;
  done.hidden = !next.hidden;
  stepStatus.textContent = `Step ${index + 1} of ${steps.length}`;

  if (moveFocus) {
    const title = steps[index].querySelector("h4");
    title.tabIndex = -1;
    title.focus();
  }
}

function showTrack(track) {
  activeTrack = track;
  tracks.forEach(({ tab, panel }) => {
    const selected = panel === track.panel;
    panel.hidden = !selected;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  showStep(track.index);
}

const tablist = walkthrough.querySelector("[data-tabs]");
tablist.setAttribute("role", "tablist");
tracks.forEach((track, position) => {
  const { tab, panel, buttons } = track;
  tab.setAttribute("role", "tab");
  tab.setAttribute("aria-controls", panel.id);
  panel.setAttribute("role", "tabpanel");
  panel.setAttribute("aria-labelledby", tab.id);
  tab.addEventListener("click", () => showTrack(track));
  tab.addEventListener("keydown", (event) => {
    let target;
    switch (event.key) {
      case "ArrowLeft":
        target = (position - 1 + tracks.length) % tracks.length;
        break;
      case "ArrowRight":
        target = (position + 1) % tracks.length;
        break;
      case "Home":
        target = 0;
        break;
      case "End":
        target = tracks.length - 1;
        break;
      default:
        return;
    }
    event.preventDefault();
    showTrack(tracks[target]);
    tabs[target].focus();
  });
  buttons.forEach((button, index) => button.addEventListener("click", () => showStep(index)));
});
previous.addEventListener("click", () => showStep(activeTrack.index - 1, true));
next.addEventListener("click", () => showStep(activeTrack.index + 1, true));
showTrack(activeTrack);
walkthrough.classList.add("enhanced");
tablist.hidden = false;
walkthrough.querySelector(".step-controls").hidden = false;

const copyStatus = document.querySelector("#copy-status");

async function copyCommands(code) {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(code.textContent);
      return true;
    } catch {
      // Select the commands so they can be copied with the keyboard.
    }
  }
  const range = document.createRange();
  range.selectNodeContents(code);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
  try {
    if (document.execCommand("copy")) {
      selection.removeAllRanges();
      return true;
    }
  } catch {
    // Leave the selection in place for manual copying.
  }
  return false;
}

document.querySelectorAll(".copy").forEach((button) => {
  let reset;
  button.hidden = false;
  button.addEventListener("click", async () => {
    const code = button.closest(".code-block").querySelector("code");
    const copied = await copyCommands(code);
    button.textContent = copied ? "Copied" : "Selected";
    copyStatus.textContent = copied
      ? "Commands copied."
      : "Commands selected. Use your keyboard shortcut to copy them.";
    clearTimeout(reset);
    reset = setTimeout(() => { button.textContent = "Copy"; }, 2200);
  });
});

const review = document.querySelector("[data-redaction]");
const viewButtons = [...review.querySelectorAll("[data-view]")];
const samples = [...review.querySelectorAll("[data-sample]")];

function showSample(view) {
  viewButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.view === view)));
  samples.forEach((sample) => { sample.hidden = sample.dataset.sample !== view; });
}

viewButtons.forEach((button) => button.addEventListener("click", () => showSample(button.dataset.view)));
showSample("original");
review.classList.add("enhanced");
review.querySelector(".redaction-switch").hidden = false;

function openLinkedDemo() {
  const track = tracks.find(({ panel }) => location.hash === "#" + panel.id);
  if (track) {
    showTrack(track);
    track.panel.scrollIntoView();
  }
}

window.addEventListener("hashchange", openLinkedDemo);
document.querySelector("[data-open-ollama]").addEventListener("click", () => {
  showTrack(tracks.find(({ panel }) => panel.dataset.track === "ollama"));
});
openLinkedDemo();
