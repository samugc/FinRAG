import { test, expect } from "@playwright/test";

async function fakeHealthyBackend(page, queryHandler = null) {
  await page.route("**/api/health", async (route) =>
    route.fulfill({
      json: { status: "ok", documents: 2, chunks: 4 },
    })
  );

  await page.route("**/api/documents", async (route) =>
    route.fulfill({
      json: {
        documents: [
          { source: "cash_flow.txt", chunks: 2 },
          { source: "risk_report.txt", chunks: 2 },
        ],
      },
    })
  );

  await page.route("**/api/query", async (route) => {
    if (queryHandler) return queryHandler(route);

    const request = route.request().postDataJSON();
    return route.fulfill({
      json: {
        question: request.question,
        answer: "Northstar generated 58 million euros from operations.",
        sufficient_evidence: true,
        retriever: request.retriever,
        generator: request.generator,
        source: request.source,
        evidence: [
          {
            source: request.source ?? "cash_flow.txt",
            chunk_id: 0,
            text: "Northstar generated 58 million euros from operations in 2025.",
            score: 0.87,
          },
        ],
      },
    });
  });
}

test("loads corpus and renders a grounded-style response with inspectable sources", async ({ page }) => {
  await fakeHealthyBackend(page);
  await page.goto("/");

  await expect(page.getByText("API connected")).toBeVisible();
  await expect(page.getByRole("button", { name: /Cash Flow/ })).toBeVisible();
  await page.getByRole("button", { name: /What was Northstar's operating profit/ }).click();
  await expect(page.locator(".answer-text")).toContainText("58 million euros");
  await expect(page.locator(".evidence-card")).toHaveCount(1);
  await expect(page.locator(".evidence-card")).toContainText("cash flow", { ignoreCase: true });
  await expect(page.getByText("Evidence retrieved · not fact-verified")).toBeVisible();
});

test("choosing a document and retriever sends the actual filters to FastAPI", async ({ page }) => {
  let sent = null;
  await fakeHealthyBackend(page, (route) => {
    sent = route.request().postDataJSON();
    return route.fulfill({
      json: {
        question: sent.question,
        answer: "The report mentions exchange-rate exposure.",
        sufficient_evidence: true,
        retriever: sent.retriever,
        generator: sent.generator,
        source: sent.source,
        evidence: [{
          source: "risk_report.txt",
          chunk_id: 1,
          text: "Currency exposure arises from USD and EUR cash flows.",
          score: 0.72,
        }],
      },
    });
  });
  await page.goto("/");
  await expect(page.getByText("API connected")).toBeVisible();

  await page.getByRole("button", { name: /Risk Report/ }).click();
  await page.getByLabel("Retriever").selectOption("minilm");
  await page.getByLabel("Generator").selectOption("qwen");
  await page.getByLabel("Ask a financial question").fill("Which currencies create exposure?");
  await page.getByRole("button", { name: "Send question" }).click();

  await expect(page.locator(".answer-text")).toContainText("exchange-rate exposure");
  expect(sent).toMatchObject({
    question: "Which currencies create exposure?",
    retriever: "minilm",
    generator: "qwen",
    source: "risk_report.txt",
    k: 3,
  });
  await expect(page.locator(".evidence-card")).toContainText("Currency exposure");
});

test("shows backend errors instead of inventing answers", async ({ page }) => {
  await fakeHealthyBackend(page, (route) =>
    route.fulfill({ status: 503, json: { detail: "Model unavailable" } })
  );
  await page.goto("/");
  await expect(page.getByText("API connected")).toBeVisible();

  await page.getByLabel("Ask a financial question").fill("What happened?");
  await page.getByRole("button", { name: "Send question" }).click();

  await expect(page.locator(".message-error")).toContainText("unavailable");
  await expect(page.locator(".answer-text")).toHaveCount(0);
});

test("disables requests when the local service is offline", async ({ page }) => {
  await page.route("**/api/health", (route) => route.fulfill({ status: 503, json: { status: "unavailable" } }));
  await page.route("**/api/documents", (route) => route.fulfill({ status: 503, json: { detail: "Corpus unavailable" } }));
  await page.goto("/");
  await expect(page.getByText("API offline")).toBeVisible();
  await page.getByLabel("Ask a financial question").fill("Revenue?");
  await expect(page.getByRole("button", { name: "Send question" })).toBeDisabled();
});

test("compare retrievers shows overlapping evidence without cross-method score claims", async ({ page }) => {
  const requested = [];
  await fakeHealthyBackend(page, (route) => {
    const input = route.request().postDataJSON();
    requested.push(input);
    return route.fulfill({ json: {
      question: input.question,
      answer: "Sample text",
      sufficient_evidence: true,
      retriever: input.retriever,
      generator: input.generator,
      source: input.source,
      evidence: [
        {source: "cash_flow.txt", chunk_id: 0, text: "Northstar cash flows.", score: input.retriever === "tfidf" ? 0.8 : 0.6},
        {source: "risk_report.txt", chunk_id: input.retriever === "tfidf" ? 0 : 1, text: "Other report information.", score: 0.4},
      ],
    }});
  });
  await page.goto("/");
  await expect(page.getByText("API connected")).toBeVisible();
  await page.getByRole("button", {name: /Compare/}).click();
  await page.getByRole("textbox", { name: "Question", exact: true }).fill("What do the reports say?");
  await page.getByRole("dialog", { name: "Compare retrieval methods" }).getByRole("button", {name: "Compare", exact: true}).click();
  await expect(page.getByText("1 shared chunk")).toBeVisible();
  await expect(page.locator(".compare-source")).toHaveCount(4);
  expect(requested).toHaveLength(2);
  expect(requested.map(x => x.retriever).sort()).toEqual(["minilm","tfidf"]);
  expect(requested.every(x => x.generator === "extractive" && x.k === 3)).toBe(true);
});


test("upload PDF refreshes library and selects uploaded document", async ({ page }) => {
  await fakeHealthyBackend(page);
  await page.route("**/api/documents", async (route) => {
    if (route.request().method() === "POST") {
      expect(route.request().headers()["x-finrag-upload"]).toBe("1");
      return route.fulfill({ status: 201, json: { source: "Annual_2025.pdf", chunks: 12 } });
    }
    return route.fulfill({ json: { documents: [
      { source: "cash_flow.txt", chunks: 2 },
      { source: "Annual_2025.pdf", chunks: 12 },
    ] } });
  });
  await page.goto("/");
  await expect(page.getByText("API connected")).toBeVisible();
  await page.getByLabel("Select a PDF document").setInputFiles({
    name: "Annual_2025.pdf", mimeType: "application/pdf", buffer: Buffer.from("%PDF-1.4 mock"),
  });
  await expect(page.getByRole("status")).toContainText("Added Annual_2025.pdf");
  await expect(page.getByRole("button", { name: /Annual 2025/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /Annual 2025/ })).toHaveClass(/doc-selected/);
});


test("uploaded PDF is used as the query scope", async ({ page }) => {
  let sent = null;
  await fakeHealthyBackend(page, route => {
    sent = route.request().postDataJSON();
    return route.fulfill({ json: {
      question: sent.question, answer: "Acme profit was 123 million euros.",
      sufficient_evidence: true, retriever: sent.retriever, generator: sent.generator,
      source: sent.source, evidence: [{source:"annual.pdf",chunk_id:0,
        text:"Acme profit was 123 million euros.",score:0.9}],
    }});
  });
  await page.route("**/api/documents", async route => {
    if (route.request().method() === "POST")
      return route.fulfill({status:201,json:{source:"annual.pdf",chunks:1}});
    return route.fulfill({json:{documents:[{source:"annual.pdf",chunks:1}]}});
  });
  await page.goto("/");
  await expect(page.getByText("API connected")).toBeVisible();
  await page.getByLabel("Select a PDF document").setInputFiles({
    name:"annual.pdf", mimeType:"application/pdf", buffer:Buffer.from("%PDF-1.4 test")
  });
  await expect(page.getByRole("status")).toContainText("Added annual.pdf");
  await page.getByLabel("Ask a financial question").fill("What was Acme profit?");
  await page.getByRole("button",{name:"Send question"}).click();
  await expect(page.locator(".answer-text")).toContainText("123 million");
  expect(sent.source).toBe("annual.pdf");
});

test("upload API rejection shows a readable error", async ({ page }) => {
  await fakeHealthyBackend(page);
  await page.route("**/api/documents", async route => {
    if(route.request().method()==="POST")
      return route.fulfill({status:422,json:{detail:"No selectable text found; scanned PDFs need OCR"}});
    return route.fulfill({json:{documents:[{source:"cash_flow.txt",chunks:2}]}});
  });
  await page.goto("/");
  await expect(page.getByText("API connected")).toBeVisible();
  await page.getByLabel("Select a PDF document").setInputFiles({
    name:"scanned.pdf",mimeType:"application/pdf",buffer:Buffer.from("%PDF-1.4 test")
  });
  await expect(page.locator(".global-error")).toContainText("scanned PDFs need OCR");
});


test("uploaded report does not show unrelated Northstar example prompts", async ({ page }) => {
  await fakeHealthyBackend(page);
  await page.route("**/api/documents", route => route.fulfill({
    json: { documents: [{ source: "AMD_2022_10K.pdf", chunks: 3137 }] },
  }));
  await page.goto("/");
  await page.getByRole("button", { name: /AMD 2022 10K/ }).click();
  await expect(page.locator(".suggest-title")).toContainText("SELECTED DOCUMENT");
  await expect(page.getByRole("button", { name: /Northstar's operating profit/ })).toHaveCount(0);
  await expect(page.getByText("Retrieval is restricted to the selected document.")).toBeVisible();
});

for (const [status, label] of Object.entries({produced: 'Response produced', model_abstention: 'Model abstained', invalid_output: 'Invalid model output', no_evidence: 'No evidence text supplied', backend_error: 'Generation backend unavailable', insufficient_context: 'Context insufficient'})) {
  test(`renders P0 state ${status}`, async ({page}) => {
    await fakeHealthyBackend(page, route => route.fulfill({json: {
      question: 'Revenue?', answer: status === 'produced' ? '42 [1]' : '', status,
      evidence: [{source: 'cash_flow.txt', chunk_id: 0, citation_id: 1, text: '42', score: 1}],
    }}));
    await page.goto('/');
    await expect(page.getByText('API connected')).toBeVisible();
    await page.getByLabel('Ask a financial question').fill('Revenue?');
    await page.getByRole('button', {name: 'Send question'}).click();
    await expect(page.getByText(label, {exact: false})).toBeVisible();
    await expect(page.locator('.evidence-card')).not.toContainText('PDF page');
  });
}

test('citations retain evidence from their original response', async ({page}) => {
  let count = 0;
  await fakeHealthyBackend(page, route => {
    count++;
    return route.fulfill({json: {question: `Question ${count}`, answer: `Answer ${count} [1]`, status: 'produced',
      evidence: [{source: 'report.pdf', chunk_id: count, citation_id: 1, page_number: count, text: `Evidence ${count}`, score: 1}]}});
  });
  await page.goto('/');
  await expect(page.getByText('API connected')).toBeVisible();
  for (const n of [1,2]) {
    await page.getByLabel('Ask a financial question').fill(`Question ${n}`);
    await page.getByRole('button', {name: 'Send question'}).click();
    await expect(page.locator('.answer-text').last()).toContainText(`Answer ${n}`);
  }
  await page.getByRole('button', {name: 'Inspect citation 1', exact:true}).first().click();
  await expect(page.locator('.evidence-card')).toContainText('Evidence 1');
  await expect(page.locator('.evidence-card')).toContainText('PDF page 1');
  await expect(page.locator('.evidence-card')).not.toContainText('Evidence 2');
});
