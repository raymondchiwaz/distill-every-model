(() => {
  const form = document.querySelector("#chat-approval");
  const selectAll = document.querySelector("#select-all-chats");
  const chats = Array.from(form.elements).filter(
    (control) => control.name === "chat_id" && !control.disabled,
  );
  const approve = document.querySelector("#approve-selected");
  const exportChats = document.querySelector("#chat-export");

  function updateSelection() {
    const selected = chats.filter((chat) => chat.checked);
    selectAll.checked = selected.length === chats.length;
    selectAll.indeterminate = selected.length > 0 && selected.length < chats.length;
    if (approve) {
      const pending = selected.filter((chat) => chat.hasAttribute("data-pending")).length;
      approve.disabled = pending === 0;
      approve.textContent = pending ? `Approve selected chats (${pending})` : "Approve selected chats";
    }
    if (exportChats) {
      exportChats.disabled = selected.length > 0 && !selected.some((chat) => chat.hasAttribute("data-approved"));
    }
  }

  selectAll.addEventListener("change", () => {
    for (const chat of chats) {
      chat.checked = selectAll.checked;
    }
    updateSelection();
  });
  for (const chat of chats) {
    chat.addEventListener("change", updateSelection);
  }
  selectAll.closest("label").hidden = false;
  updateSelection();
  window.addEventListener("pageshow", updateSelection);
})();
