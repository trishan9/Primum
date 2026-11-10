import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { binomialPValue, certifyThreshold, logGamma, parseCalibration, selectAt } from "./stats.ts";

const close = (a: number, b: number, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);

test("logGamma matches factorials", () => {
  close(logGamma(1), 0);
  close(logGamma(5), Math.log(24), 1e-12);
  close(logGamma(11), Math.log(3628800), 1e-10);
  close(logGamma(0.5), Math.log(Math.sqrt(Math.PI)), 1e-12);
});

test("binomialPValue equals the exact lower tail", () => {
  // P(Bin(10, 0.5) <= 2) = (1 + 10 + 45) / 1024
  close(binomialPValue(2, 10, 0.5), 56 / 1024, 1e-12);
  // P(Bin(20, 0.1) = 0) = 0.9^20
  close(binomialPValue(0, 20, 0.1), 0.9 ** 20, 1e-12);
  // P(Bin(50, 0.1) <= 1) = 0.9^50 + 50 * 0.1 * 0.9^49
  close(binomialPValue(1, 50, 0.1), 0.9 ** 50 + 50 * 0.1 * 0.9 ** 49, 1e-12);
  assert.equal(binomialPValue(10, 10, 0.3), 1);
  assert.equal(binomialPValue(3, 0, 0.3), 1);
});

test("binomialPValue stays finite for large n", () => {
  const p = binomialPValue(40, 1000, 0.1);
  assert.ok(p > 0 && p < 1e-10);
});

test("selectAt counts answered questions and their errors", () => {
  const s = selectAt([0.9, 0.8, 0.4, 0.2], [true, false, true, false], 0.5);
  assert.deepEqual([s.nSelected, s.nErrors, s.coverage, s.risk], [2, 1, 0.5, 0.5]);
  assert.ok(Number.isNaN(selectAt([0.1], [true], 0.5).risk));
});

// A deterministic generator so the tests do not depend on Math.random.
function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test("certifyThreshold returns a threshold whose calibration risk is within alpha", () => {
  const rand = mulberry32(13);
  const conf: number[] = [];
  const correct: boolean[] = [];
  for (let i = 0; i < 600; i++) {
    const p = rand();
    conf.push(p);
    correct.push(rand() < p); // perfectly calibrated confidences
  }
  const cert = certifyThreshold(conf, correct, { alpha: 0.2, delta: 0.1 });
  assert.notEqual(cert.threshold, null);
  assert.ok(cert.selection && cert.selection.risk <= 0.2);
  assert.ok(cert.pValue !== null && cert.pValue <= 0.1);
  // Every step before the stop passed, and nothing is tested after the first failure.
  const firstFail = cert.steps.findIndex((s) => !s.passed);
  assert.ok(firstFail === -1 || firstFail === cert.steps.length - 1);
});

test("a stricter alpha never yields a lower threshold", () => {
  const rand = mulberry32(7);
  const conf: number[] = [];
  const correct: boolean[] = [];
  for (let i = 0; i < 800; i++) {
    const p = rand();
    conf.push(p);
    correct.push(rand() < p);
  }
  const loose = certifyThreshold(conf, correct, { alpha: 0.2, delta: 0.1 }).threshold;
  const strict = certifyThreshold(conf, correct, { alpha: 0.1, delta: 0.1 }).threshold;
  assert.ok(loose !== null);
  assert.ok(strict === null || strict >= loose);
});

interface SharedCase {
  name: string;
  delta: number;
  n_grid: number;
  min_selected: number;
  p_correct: number[];
  correct: number[];
  expected: Record<string, { threshold: number | null; n_selected: number }>;
}

test("certifyThreshold reproduces the server's certify_threshold on shared fixtures", () => {
  // Expected values come from the Python reference; tests/test_api.py keeps them current.
  const path = new URL("../../../tests/fixtures/certify_threshold_cases.json", import.meta.url);
  const fixture = JSON.parse(readFileSync(path, "utf8")) as { alphas: number[]; cases: SharedCase[] };
  for (const c of fixture.cases) {
    // Read through parseCalibration, so the settings travel the same way /api/calibration sends them.
    const data = parseCalibration({ p_correct: c.p_correct, correct: c.correct, delta: c.delta, n_grid: c.n_grid, min_selected: c.min_selected });
    assert.ok(data && data.nGrid !== null && data.minSelected !== null);
    for (const alpha of fixture.alphas) {
      const cert = certifyThreshold(data.conf, data.correct, { alpha, delta: c.delta, nGrid: data.nGrid, minSelected: data.minSelected });
      const expected = c.expected[alpha.toFixed(2)];
      assert.deepEqual(
        { threshold: cert.threshold, nSelected: cert.selection?.nSelected ?? 0 },
        { threshold: expected?.threshold, nSelected: expected?.n_selected },
        `${c.name}, alpha ${alpha}`,
      );
    }
  }
});

test("certifyThreshold abstains entirely when no candidate can be certified", () => {
  const conf = Array.from({ length: 100 }, (_, i) => i / 100);
  const correct = conf.map(() => false);
  const cert = certifyThreshold(conf, correct, { alpha: 0.1, delta: 0.1 });
  assert.equal(cert.threshold, null);
  assert.equal(cert.steps.length, 1);
});

test("the violation frequency of the certificate stays within delta", () => {
  // Confidence c is correct with probability c, so answering at threshold t has true risk (1 - t) / 2.
  const alpha = 0.2;
  const delta = 0.1;
  const rand = mulberry32(2026);
  let violations = 0;
  let certified = 0;
  const draws = 200;
  for (let d = 0; d < draws; d++) {
    const conf: number[] = [];
    const correct: boolean[] = [];
    for (let i = 0; i < 1000; i++) {
      const p = rand();
      conf.push(p);
      correct.push(rand() < p);
    }
    const t = certifyThreshold(conf, correct, { alpha, delta }).threshold;
    if (t === null) continue;
    certified++;
    if ((1 - t) / 2 > alpha) violations++;
  }
  assert.ok(certified > draws * 0.8, `${certified} certified`);
  // Allow three binomial standard errors of Monte Carlo noise above delta.
  assert.ok(violations / draws <= delta + 3 * Math.sqrt((delta * (1 - delta)) / draws), `${violations}/${draws}`);
});

test("parseCalibration accepts column and row layouts", () => {
  const columns = parseCalibration({
    p_correct: [0.9, 0.2],
    correct: [1, 0],
    thresholds: { "0.05": null, "0.10": 0.71 },
    delta: 0.1,
  });
  assert.deepEqual(columns?.conf, [0.9, 0.2]);
  assert.deepEqual(columns?.correct, [true, false]);
  assert.deepEqual(columns?.certified, [
    { alpha: 0.05, threshold: null },
    { alpha: 0.1, threshold: 0.71 },
  ]);
  const rows = parseCalibration({
    items: [{ p_correct: 0.5, correct: true }],
    thresholds: [{ alpha: 0.2, threshold: 0.4, n_cal: 300 }],
  });
  assert.deepEqual(rows?.conf, [0.5]);
  assert.deepEqual(rows?.certified, [{ alpha: 0.2, threshold: 0.4 }]);
  assert.equal(rows?.heldout, null);
  assert.equal(rows?.minSelected, null);
  assert.equal(parseCalibration({ nothing: true }), null);
  assert.equal(parseCalibration(null), null);
});

test("parseCalibration reads the held-out block and the certification settings", () => {
  const data = parseCalibration({
    p_correct: [0.9, 0.2],
    correct: [1, 0],
    min_selected: 30,
    n_grid: 100,
    heldout: { p_correct: [0.8, 0.7, 0.1], correct: [true, false, false] },
  });
  assert.deepEqual(data?.heldout, { conf: [0.8, 0.7, 0.1], correct: [true, false, false] });
  assert.equal(data?.minSelected, 30);
  assert.equal(data?.nGrid, 100);
  // A held-out block without rows is ignored, not an error.
  assert.equal(parseCalibration({ p_correct: [0.5], correct: [1], heldout: {} })?.heldout, null);
});
