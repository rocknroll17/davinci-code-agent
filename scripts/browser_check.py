"""Real-browser check: load the site with the given model, let the model play itself in the page,
then replay every observation through ONNX Runtime (Python) and compare the chosen actions."""
import glob, json, os, subprocess, sys, time
import numpy as np, onnxruntime as ort
from playwright.sync_api import sync_playwright

site, games = sys.argv[1], int(sys.argv[2])
url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8765/index.html"   # optional: the live site
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8765", "--bind", "127.0.0.1", "--directory", site],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
JS = """
async (games) => {
  const DV = window.DVEngine; await DV.loadModel();
  const rec = [];
  for (let g = 0; g < games; g++) {
    const a = DV.makeAIPlayer('a', true), b = DV.makeAIPlayer('b', true);
    const e = DV.makeEngine([a, b], 'g' + g); DV.engineSetup(e);
    for (let step = 0; step < 600 && !e.game_over; step++) {
      const p = e.current_player, phase = e.phase;
      const obs = DV.buildObservation(e, p.id), mask = DV.buildActionMask(e, p.id);
      const t0 = performance.now(); const act = await DV.getAction(e, p.id); const ms = performance.now() - t0;
      rec.push({obs, mask, act, ms, phase});
      if (phase === DV.Phase.DRAW) {
        DV.engineDraw(e, p.id, act.color);
        const pend = e.pending_card; DV.enginePlace(e, p.id, pend.color, pend.value, pend.valid_positions[0]);
      } else if (phase === DV.Phase.GUESS) DV.engineGuess(e, p.id, act.position, act.value);
      else DV.engineDecision(e, p.id, act.decision === 1);
    }
    if (!e.game_over) throw new Error('game did not finish');
  }
  return rec;
}
"""
errors = []
try:
    with sync_playwright() as pw:
        exe = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/*/headless_shell")) +
                     glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell")))
        browser = pw.chromium.launch(executable_path=exe[0] if exe else None)
        page = browser.new_page()
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url, wait_until="load")
        page.wait_for_function("window.DVEngine && window.ort")
        rec = page.evaluate(JS, games)
        print("browser:", browser.version)
        browser.close()
finally:
    srv.terminate()

o = ort.SessionOptions(); o.intra_op_num_threads = 1
s = ort.InferenceSession(os.path.join(site, "model.onnx"), o, providers=["CPUExecutionProvider"])
am = lambda l, m: int(np.argmax(np.where(np.array(m, dtype=bool), l, -1e4)))
diff = 0
for r in rec:
    ob, mk, a = r["obs"], r["mask"], r["act"]
    f = {i.name: np.array(ob[i.name], dtype=np.float32)[None] for i in s.get_inputs()}
    c, p, v, d = s.run(["color_logits", "position_logits", "value_logits", "decision_logits"], f)
    pos = am(p[0], mk["position"])
    mine = {0: (am(c[0], mk["color"]),), 1: (pos, am(v[0][pos], mk["value"][pos])), 2: (am(d[0], mk["decision"]),)}[r["phase"]]
    theirs = {0: (a["color"],), 1: (a["position"], a["value"]), 2: (a["decision"],)}[r["phase"]]
    diff += mine != theirs
ms = np.array([r["ms"] for r in rec])
print(f"games {games}, model calls {len(rec)}, page errors {len(errors)} {errors[:3]}")
print(f"actions differing from Python ONNX Runtime on the same observation: {diff}")
print(f"getAction in browser (WASM, 1 thread): median {np.median(ms[20:]):.2f} ms, p95 {np.percentile(ms[20:], 95):.2f} ms")
