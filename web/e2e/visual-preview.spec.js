import { test } from "@playwright/test";

// Capture the actual React UI with deterministic demo API responses.
// These are visual previews, not evidence of real-model accuracy.
test("capture FinRAG workspace previews", async ({ page }) => {
  await page.route("**/api/health", async route =>
    route.fulfill({ json: {status:"ok", documents:4, chunks:12} })
  );
  await page.route("**/api/documents", async route =>
    route.fulfill({json:{documents:[
      {source:"sample_report.txt",chunks:3},
      {source:"cash_flow.txt",chunks:3},
      {source:"risk_report.txt",chunks:3},
      {source:"outlook.txt",chunks:3}
    ]}})
  );
  await page.route("**/api/query", async route =>
    route.fulfill({json:{
      question:"What was Northstar's operating profit in 2025?",
      answer:"Northstar reported an operating profit of €58 million in 2025.",
      sufficient_evidence:true,
      retriever:"tfidf",
      generator:"extractive",
      source:null,
      evidence:[
        {source:"sample_report.txt",chunk_id:0,score:0.873,text:"In 2025, Northstar reported revenue of €420 million and operating profit of €58 million. The company ended the year with net debt of €110 million."},
        {source:"cash_flow.txt",chunk_id:1,score:0.629,text:"Northstar's annual financial statements include revenue, cash flow, operating profit, and investment spending for 2025."},
        {source:"outlook.txt",chunk_id:0,score:0.412,text:"Management expects modest revenue growth in 2026 while continuing to invest in operations."}
      ]
    }})
  );
  await page.setViewportSize({width:1440,height:900});
  await page.goto("/");
  await page.getByText("API connected").waitFor();
  await page.screenshot({path:"preview/finrag-workspace.png",fullPage:true});
  await page.getByLabel("Ask a financial question").fill("What was Northstar's operating profit in 2025?");
  await page.getByRole("button",{name:"Send question"}).click();
  await page.locator(".answer-text").waitFor();
  await page.screenshot({path:"preview/finrag-answer-demo.png",fullPage:true});
});
