// Integrity checks for .github/areas.json -- the ownership map PR reviewer
// assignment reads. Run offline: `node .github/kaino/areas.test.js` (cwd = repo
// root). No network.
//
// Upstream also validated each area's comp:* label, priority_label and weight;
// those fed the issue-triage workflow, which this fork does not run. Only the
// fields auto-assign-reviewer.js actually reads are checked here. The unused
// fields are left in areas.json rather than stripped, so an upstream merge that
// touches them stays a clean text merge.
const fs = require("fs");
const path = require("path");

const areas = JSON.parse(fs.readFileSync(path.resolve(".github/areas.json"), "utf8")).areas;
const maint = new Set(
  fs.readFileSync(path.resolve(".github/MAINTAINER"), "utf8")
    .split("\n").map((l) => l.replace(/#.*/, "").trim().toLowerCase()).filter(Boolean)
);

let failures = 0;
function assert(name, cond, detail) {
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? "  -- " + detail : ""}`);
  if (!cond) failures++;
}

// Every owner is a known maintainer.
for (const a of areas)
  for (const o of a.owners || [])
    assert(`owner @${o} (area ${a.key}) is in MAINTAINER`, maint.has(o.toLowerCase()));

// Every area has at least one owner. Upstream required 2+ (its codeowner rule);
// with a team this size that would force padding every area with the same names.
// Paused owners still count -- pausing someone must not force adding a new one.
for (const a of areas) {
  const n = (a.owners || []).length + (a.owners_paused || []).length;
  assert(`area ${a.key} has >= 1 owner`, n >= 1, `${n} owner(s)`);
}

// Every area has a definition and at least one path.
for (const a of areas) {
  assert(`area ${a.key} has a definition`, typeof a.definition === "string" && a.definition.length > 0);
  assert(`area ${a.key} has paths`, Array.isArray(a.paths) && a.paths.length > 0);
}

// Path resolution (last-match-wins startsWith) sends representative files to the
// expected area -- especially the web/ carve-out ordering and harness prefixes.
function resolve(fn) {
  let match = null;
  for (const a of areas) for (const p of a.paths) if (fn.startsWith(p)) match = a;
  return match;
}
const cases = [
  ["omnigent/inner/foo.py", "inner"],
  ["omnigent/inner/claude_sdk_executor.py", "harness-claude"],
  ["omnigent/inner/kimi_executor.py", "harness-kimi"],
  ["omnigent/inner/kiro_native_harness.py", "harness-kiro"],
  ["web/src/main.tsx", "web"],
  ["web/ios/App.swift", "mobile-app"],
  ["web/android/app/src/main/MainActivity.kt", "android-app"],
  ["web/electron/main.ts", "desktop-app"],
  ["omnigent/server/api.py", "server"],
  ["omnigent/server/auth.py", "auth"],
];
for (const [fn, key] of cases) {
  const m = resolve(fn);
  assert(`${fn} -> ${key}`, m && m.key === key, m ? m.key : "(unmatched)");
}

console.log(failures ? `\n${failures} FAILURE(S)` : "\nAll areas.json integrity checks passed.");
process.exitCode = failures ? 1 : 0;
