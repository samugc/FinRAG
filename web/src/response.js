export function statusLabel(data) {
  const labels = {
    produced: "Response produced · not fact-verified",
    model_abstention: "Model abstained · evidence may still contain the answer",
    insufficient_context: "Context insufficient · not proof of missing data in the PDF",
    no_evidence: "No evidence text supplied",
    invalid_output: "Invalid model output · inspect raw response",
    backend_error: "Generation backend unavailable",
  };
  return labels[data.status] || (data.sufficient_evidence
    ? "Evidence retrieved · not fact-verified" : "Response not accepted · inspect evidence");
}

export function citationParts(text = "") {
  return text.split(/(\[\s*\d+(?:\s*,\s*\d+)*\s*\])/g).map(part => {
    const match = part.match(/^\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]$/);
    return { text: part, ids: match ? match[1].split(",").map(Number) : [] };
  });
}
