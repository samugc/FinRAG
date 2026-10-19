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
  await page.getByRole("button", {name: "Compare", exact: true}).click();
  await expect(page.getByText("1 shared chunk")).toBeVisible();
  await expect(page.locator(".compare-source")).toHaveCount(4);
  expect(requested).toHaveLength(2);
  expect(requested.map(x => x.retriever).sort()).toEqual(["minilm","tfidf"]);
  expect(requested.every(x => x.generator === "extractive" && x.k === 3)).toBe(true);
});
