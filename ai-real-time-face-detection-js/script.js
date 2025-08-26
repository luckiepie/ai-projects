// ===== script.js (표정 라벨 고정 표시 최종본) =====

// ---- DOM ----
const video       = document.getElementById('video');
const stage       = document.getElementById('stage');
const liveDot     = document.getElementById('liveDot');
const fpsEl       = document.getElementById('fps');
const facesEl     = document.getElementById('faces');
const toast       = document.getElementById('toast');

const chkBoxes    = document.getElementById('toggleBoxes');
const chkLm       = document.getElementById('toggleLandmarks');
const chkExpr     = document.getElementById('toggleExpr');
const selInputSize= document.getElementById('inputSize');

const btnPause    = document.getElementById('btnPause');
const btnResume   = document.getElementById('btnResume');
const btnCapture  = document.getElementById('btnCapture');

// ---- State ----
let canvas = null, ctx = null;
let running = true;
let rafId = null;

let lastTs = 0, fps = 0;

function setLive(on){ liveDot && liveDot.classList.toggle('on', on); }
function showToast(msg){ if (toast) { toast.textContent = msg; toast.hidden = false; } }
function hideToast(){ if (toast) { toast.hidden = true; } }

function getDetectorOpts(){
  const inputSize = Number(selInputSize?.value || 416);
  return new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold: 0.5 });
}

// ---- Models ----
async function loadModels() {
  const base = '/models';
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(base),
    faceapi.nets.faceLandmark68Net.loadFromUri(base),
    faceapi.nets.faceExpressionNet.loadFromUri(base),
    faceapi.nets.faceRecognitionNet.loadFromUri(base), // 원본 호환
  ]);
}

// ---- Camera ----
async function startCamera() {
  showToast('카메라 초기화 중… 권한을 허용해 주세요.');
  try {
    const stream = await (navigator.mediaDevices?.getUserMedia
      ? navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
      : new Promise((resolve, reject) =>
          navigator.getUserMedia({ video: true, audio: false }, resolve, reject)
        )
    );
    video.srcObject = stream;
    await video.play();
    hideToast();
  } catch (e) {
    console.error(e);
    showToast('카메라 권한이 필요하거나 HTTPS 환경이 아닙니다.');
  }
}

// ---- Canvas ----
function ensureCanvas() {
  if (!canvas) {
    canvas = faceapi.createCanvasFromMedia(video);
    ctx = canvas.getContext('2d');
    (stage || video.parentElement || document.body).appendChild(canvas);
    canvas.style.zIndex = '2';
    canvas.style.pointerEvents = 'none';
  }
  const w = video.videoWidth || video.width;
  const h = video.videoHeight || video.height;
  canvas.width = w;
  canvas.height = h;
  faceapi.matchDimensions(canvas, { width: w, height: h });
}

// ---- Drawing helpers ----
function drawBoxesWithScores(dets) {
  dets.forEach(d => {
    const { box, score } = d.detection;
    const drawBox = new faceapi.draw.DrawBox(box, {
      label: (score ?? 0).toFixed(2),  // 예: "0.87"
      lineWidth: 2,
      boxColor: '#3b82f6',
      drawLabelOptions: { backgroundColor: '#3b82f6', fontColor: '#fff', fontSize: 14 }
    });
    drawBox.draw(canvas);
  });
}

// 간단/안전한 방식: DrawTextField 생성 시 바로 draw() 호출, 옵션 오버라이드는 생략
function drawExpressionTop1(dets) {
  dets.forEach(d => {
    const exps = d.expressions;
    if (!exps) return;
    const sorted = exps.asSortedArray();
    if (!sorted.length) return;
    const best = sorted[0]; // { expression, probability }
    const { x, y } = d.detection.box;

    const label = `${best.expression} ${Math.round(best.probability * 100)}%`;
    // 박스 윗쪽에 라벨(공간 부족하면 자동으로 화면 안쪽에 보정 필요시 y-값 조절)
    new faceapi.draw.DrawTextField([label], { x, y: Math.max(0, y - 18) }).draw(canvas);
  });
}

// ---- Loop ----
async function loop(ts) {
  if (!running) return;

  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) { rafId = requestAnimationFrame(loop); return; }

  ensureCanvas();

  const results = await faceapi
    .detectAllFaces(video, getDetectorOpts())
    .withFaceLandmarks()
    .withFaceExpressions();

  if (facesEl) facesEl.textContent = results.length;

  const resized = faceapi.resizeResults(results, { width: w, height: h });

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 박스
  if (!chkBoxes || chkBoxes.checked) drawBoxesWithScores(resized);
  // 랜드마크
  if (!chkLm    || chkLm.checked)    faceapi.draw.drawFaceLandmarks(canvas, resized);
  // 표정: 텍스트 라벨 + 기본 바차트 둘 다 그리기
  if (!chkExpr  || chkExpr.checked) {
    drawExpressionTop1(resized);                // "surprised 87%" 형태
    faceapi.draw.drawFaceExpressions(canvas, resized); // 기본 바차트
  }

  // FPS(EMA)
  if (lastTs) {
    const cur = 1000 / (ts - lastTs);
    fps = fps ? fps * 0.8 + cur * 0.2 : cur;
    if (fpsEl) fpsEl.textContent = String(Math.round(fps));
  }
  lastTs = ts;

  rafId = requestAnimationFrame(loop);
}

// ---- Controls ----
btnPause?.addEventListener('click', () => {
  if (!running) return;
  running = false;
  setLive(false);
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  btnPause.disabled = true;
  btnResume.disabled = false;
});

btnResume?.addEventListener('click', () => {
  if (running) return;
  running = true;
  setLive(true);
  btnPause.disabled = false;
  btnResume.disabled = true;
  lastTs = 0;
  rafId = requestAnimationFrame(loop);
});

btnCapture?.addEventListener('click', () => {
  if (!canvas) return;
  const w = video.videoWidth || canvas.width;
  const h = video.videoHeight || canvas.height;
  const snap = document.createElement('canvas');
  snap.width = w; snap.height = h;
  const sctx = snap.getContext('2d');
  sctx.drawImage(video, 0, 0, w, h);
  sctx.drawImage(canvas, 0, 0, w, h);
  const a = document.createElement('a');
  a.href = snap.toDataURL('image/png');
  a.download = `capture-${new Date().toISOString().replace(/[:.]/g,'-')}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
});

selInputSize?.addEventListener('change', () => { /* 다음 프레임부터 자동 반영 */ });

// 탭 전환 시 자동 일시정지/재개
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    setLive(false);
  } else if (running && !rafId) {
    setLive(true);
    rafId = requestAnimationFrame(loop);
  }
});

// 페이지 종료 시 카메라 릴리즈
window.addEventListener('beforeunload', () => {
  try { video.srcObject?.getTracks?.().forEach(t => t.stop()); } catch (_) {}
  if (rafId) cancelAnimationFrame(rafId);
});

// ---- Boot ----
(async function main(){
  showToast('모델 로딩 중…');
  await loadModels();
  hideToast();

  await startCamera();

  // 메타데이터 이후 시작
  if (video.readyState >= 1) {
    setLive(true);
    rafId = requestAnimationFrame(loop);
  } else {
    video.addEventListener('loadedmetadata', () => {
      setLive(true);
      rafId = requestAnimationFrame(loop);
    }, { once: true });
  }
})();
