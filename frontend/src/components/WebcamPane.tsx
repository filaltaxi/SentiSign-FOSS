import { useCallback, useEffect, useRef } from 'react';
import type { EmotionType } from './EmotionStrip';
import type { SignModel } from '../model/ModelContext';
import { extractTemporalFeatures, hasTemporalSignal } from '../lib/handFeatures';

interface WebcamPaneProps {
    model: SignModel;
    isActive: boolean;
    onEmotionDetected: (emotion: EmotionType) => void;
    onSignDetected: (word: string | null, cls: string | null, confidence: number) => void;
    currentEmotion: EmotionType;
    wordLabel: string;
    confidence: number;
}

// MediaPipe utilities will be loaded via script tags in index.html,
// we access them via window object in a real implementation.
declare global {
    interface Window {
        Hands: any;
        Camera: any;
        drawConnectors: any;
        drawLandmarks: any;
        HAND_CONNECTIONS: any;
    }
}

const RECOGNISE_DELAY = 80;
const LSTM_N_FRAMES = 60;
const LSTM_STRIDE = 5;

export function WebcamPane({
    model,
    isActive,
    onEmotionDetected,
    onSignDetected,
    currentEmotion,
    wordLabel,
    confidence,
}: WebcamPaneProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Keep latest callback props without forcing MediaPipe to re-subscribe.
    const onEmotionDetectedRef = useRef(onEmotionDetected);
    const onSignDetectedRef = useRef(onSignDetected);

    useEffect(() => {
        onEmotionDetectedRef.current = onEmotionDetected;
    }, [onEmotionDetected]);

    useEffect(() => {
        onSignDetectedRef.current = onSignDetected;
    }, [onSignDetected]);

    // Ref to hold the MediaPipe instances
    const mpRef = useRef<{
        camera: any | null;
        hands: any | null;
        emotionInterval: ReturnType<typeof setInterval> | null;
        lastRecogniseTime: number;
        frameBuffer: number[][];
        frameCounter: number;
        inputCanvas: HTMLCanvasElement | null;
        inputCtx: CanvasRenderingContext2D | null;
    }>({
        camera: null,
        hands: null,
        emotionInterval: null,
        lastRecogniseTime: 0,
        frameBuffer: [],
        frameCounter: 0,
        inputCanvas: null,
        inputCtx: null,
    });

    const onHandResults = useCallback(async (results: any) => {
        const videoEl = videoRef.current;
        const canvas = canvasRef.current;
        if (!videoEl || !canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = videoEl.videoWidth || 640;
        canvas.height = videoEl.videoHeight || 480;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw
        const handLandmarks = results.multiHandLandmarks ?? [];
        for (let i = 0; i < handLandmarks.length; i++) {
            const lm = results.multiHandLandmarks[i];
            const label = results.multiHandedness?.[i]?.classification?.[0]?.label ?? 'Left';
            const isRight = label === 'Right';
            const color = isRight ? '#007FFF' : '#FF7F40';
            const mirrorOverlay = model === 'mlp';

            if (mirrorOverlay) {
                ctx.save();
                ctx.translate(canvas.width, 0);
                ctx.scale(-1, 1);
            }

            window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, { color: color + '55', lineWidth: 2 });
            window.drawLandmarks(ctx, lm, { color, lineWidth: 1, radius: 3 });

            const xs = lm.map((p: any) => p.x * canvas.width);
            const ys = lm.map((p: any) => p.y * canvas.height);
            const x1 = Math.max(0, Math.min(...xs) - 20);
            const y1 = Math.max(0, Math.min(...ys) - 20);
            const x2 = Math.min(canvas.width, Math.max(...xs) + 20);
            const y2 = Math.min(canvas.height, Math.max(...ys) + 20);

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            if (mirrorOverlay) {
                ctx.restore();
            }

            const labelX = mirrorOverlay ? canvas.width - x2 : x1;
            ctx.fillStyle = color;
            ctx.font = '13px "Manrope"';
            ctx.fillText(isRight ? 'R' : 'L', labelX + 4, y1 - 4);
        }

        const features = extractTemporalFeatures(results);
        const hasSignal = hasTemporalSignal(features);

        if (model === 'mlp') {
            if (!hasSignal) {
                onSignDetectedRef.current(null, null, 0);
                return;
            }
            const now = Date.now();
            if (now - mpRef.current.lastRecogniseTime < RECOGNISE_DELAY) return;
            mpRef.current.lastRecogniseTime = now;

            try {
                const res = await fetch('/api/recognise', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ landmarks: features })
                });
                const data = await res.json();
                onSignDetectedRef.current(data.word, data.class, data.confidence);
            } catch (err) {
                console.error('Recognise API failed', err);
            }
            return;
        }

        mpRef.current.frameBuffer.push(features);
        if (mpRef.current.frameBuffer.length > LSTM_N_FRAMES) {
            mpRef.current.frameBuffer.shift();
        }
        mpRef.current.frameCounter += 1;

        if (mpRef.current.frameBuffer.length < LSTM_N_FRAMES) {
            onSignDetectedRef.current(null, null, 0);
            return;
        }
        if (mpRef.current.frameCounter % LSTM_STRIDE !== 0) return;

        try {
            const res = await fetch('/api/temporal/recognise', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sequence: mpRef.current.frameBuffer })
            });
            const data = await res.json();
            const cls = typeof data.class === 'string' ? data.class : null;
            const word = typeof data.word === 'string' ? data.word : null;
            const conf = Number.isFinite(Number(data.confidence)) ? Number(data.confidence) : 0;
            onSignDetectedRef.current(word, cls, conf);
        } catch (err) {
            console.error('Temporal recognise API failed', err);
        }
    }, [model]);

    // Lifecycle for Camera / Session
    useEffect(() => {
        let ignore = false;

        function stopSession(opts?: { resetEmotion?: boolean }) {
            const resetEmotion = opts?.resetEmotion ?? true;

            mpRef.current.lastRecogniseTime = 0;
            mpRef.current.frameBuffer = [];
            mpRef.current.frameCounter = 0;
            mpRef.current.inputCanvas = null;
            mpRef.current.inputCtx = null;

            if (mpRef.current.emotionInterval) {
                clearInterval(mpRef.current.emotionInterval);
                mpRef.current.emotionInterval = null;
            }
            if (mpRef.current.camera) {
                try {
                    mpRef.current.camera.stop();
                } catch { }
                mpRef.current.camera = null;
            }
            if (mpRef.current.hands) {
                try {
                    mpRef.current.hands.close();
                } catch { }
                mpRef.current.hands = null;
            }
            if (videoRef.current?.srcObject) {
                const stream = videoRef.current.srcObject as MediaStream;
                stream.getTracks().forEach((t) => t.stop());
                videoRef.current.srcObject = null;
            }
            if (videoRef.current) {
                try {
                    videoRef.current.pause();
                } catch { }
            }
            if (canvasRef.current) {
                const ctx = canvasRef.current.getContext('2d');
                if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            }
            if (resetEmotion) {
                onEmotionDetectedRef.current('neutral');
            }
        }

        async function startSession() {
            if (!videoRef.current || !canvasRef.current) return;
            if (!window.Hands || !window.Camera) {
                console.error("MediaPipe scripts not loaded yet");
                return;
            }

            try {
                const videoEl = videoRef.current;

                // Clean slate if a previous session wasn't fully torn down.
                stopSession({ resetEmotion: false });
                mpRef.current.lastRecogniseTime = 0;
                mpRef.current.frameBuffer = [];
                mpRef.current.frameCounter = 0;

                const handsDetector = new window.Hands({
                    locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`
                });

                const inputCanvas = document.createElement('canvas');
                const inputCtx = inputCanvas.getContext('2d');
                mpRef.current.inputCanvas = inputCanvas;
                mpRef.current.inputCtx = inputCtx;

                handsDetector.setOptions({
                    maxNumHands: 2,
                    modelComplexity: 1,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                });

                handsDetector.onResults(onHandResults);
                mpRef.current.hands = handsDetector;

                const camera = new window.Camera(videoEl, {
                    onFrame: async () => {
                        if (ignore) return;
                        if (mpRef.current.hands && videoEl) {
                            if (model === 'mlp') {
                                await mpRef.current.hands.send({ image: videoEl });
                                return;
                            }

                            const w = videoEl.videoWidth;
                            const h = videoEl.videoHeight;
                            const ctx = mpRef.current.inputCtx;
                            const canvas = mpRef.current.inputCanvas;
                            if (ctx && canvas && w && h) {
                                if (canvas.width !== w) canvas.width = w;
                                if (canvas.height !== h) canvas.height = h;
                                ctx.save();
                                ctx.clearRect(0, 0, w, h);
                                ctx.translate(w, 0);
                                ctx.scale(-1, 1);
                                ctx.drawImage(videoEl, 0, 0, w, h);
                                ctx.restore();
                                await mpRef.current.hands.send({ image: canvas });
                            } else {
                                await mpRef.current.hands.send({ image: videoEl });
                            }
                        }
                    },
                    width: 640,
                    height: 480,
                });

                mpRef.current.camera = camera;
                await camera.start();

                if (ignore) {
                    stopSession({ resetEmotion: false });
                    return;
                }

                // Emotion Loop
                mpRef.current.emotionInterval = setInterval(async () => {
                    if (ignore) return;
                    if (!videoEl.videoWidth) return;
                    try {
                        const tmpCanvas = document.createElement('canvas');
                        tmpCanvas.width = videoEl.videoWidth;
                        tmpCanvas.height = videoEl.videoHeight;
                        const tmpCtx = tmpCanvas.getContext('2d');
                        if (!tmpCtx) return;

                        tmpCtx.translate(tmpCanvas.width, 0);
                        tmpCtx.scale(-1, 1);
                        tmpCtx.drawImage(videoEl, 0, 0);

                        const base64 = tmpCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];
                        const res = await fetch('/api/emotion', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ image: base64 })
                        });
                        const data = await res.json();

                        if (data.emotion && !ignore) {
                            onEmotionDetectedRef.current(data.emotion);
                        }
                    } catch { }
                }, 500);

            } catch (err) {
                console.error("Camera failed:", err);
                stopSession({ resetEmotion: false });
            }
        }

        if (isActive) {
            startSession();
        } else {
            stopSession();
        }

        return () => {
            ignore = true;
            stopSession();
        };
    }, [isActive, onHandResults]);

    return (
        <div className={`relative overflow-hidden rounded-2xl border bg-[#eef5ff] shadow-[inset_0_1px_0_rgba(255,255,255,0.8)] transition-all duration-500 ${isActive ? 'camera-live-shell border-[#9fc9ff] shadow-[0_18px_36px_rgba(0,127,255,0.22)]' : 'border-[#c9defd]'}`}>
            <div className="relative aspect-4/3 bg-[#d6e7ff]">
                <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className={`block h-full w-full scale-x-[-1] object-cover transition-all duration-500 ${isActive ? 'opacity-100' : 'opacity-55 saturate-75'}`}
                />
                <canvas
                    ref={canvasRef}
                    className="pointer-events-none absolute inset-0 h-full w-full"
                />
                <div className={`pointer-events-none absolute inset-0 border transition-all duration-500 ${isActive ? 'camera-live-pulse border-[rgba(71,158,255,0.72)]' : 'border-transparent'}`} />

                {/* Top Left Status */}
                <div className={`absolute left-3 top-3 flex items-center gap-2.5 rounded-full border bg-white/95 px-3 py-1.5 text-[0.72rem] font-semibold uppercase tracking-[0.14em] shadow-[0_8px_18px_rgba(15,34,68,0.12)] ${isActive ? 'border-[#c0dbff] text-brand' : 'border-[#c8defe] text-muted'}`}>
                    <div className={`h-2.5 w-2.5 rounded-full transition-all duration-300 ${isActive ? 'bg-brand shadow-[0_0_12px_rgba(0,127,255,0.7)]' : 'bg-[#9ab5d7]'}`} />
                    <span>{isActive ? 'LIVE' : 'Standby'}</span>
                </div>

                {/* Top Right Emotion */}
                <div
                    className="absolute right-3 top-3 rounded-lg border bg-white/92 px-3 py-1.5 text-[0.78rem] font-semibold capitalize transition-colors duration-400"
                    style={{
                        borderColor: getEmotionColor(currentEmotion),
                        color: getEmotionColor(currentEmotion)
                    }}
                >
                    {isActive ? currentEmotion : '—'}
                </div>
            </div>

            {/* Sign Confidence Strip */}
            <div className={`flex items-center gap-4 border-t px-5 py-3.5 text-[0.82rem] transition-all duration-300 ${isActive ? 'border-[#c8defe] bg-[#f4f9ff]' : 'border-border-color bg-white'}`}>
                <span className={`min-w-[160px] truncate text-[0.77rem] font-bold uppercase tracking-[0.15em] transition-colors ${isActive && confidence > 0 ? 'text-brand' : 'text-muted'}`}>
                    {wordLabel}
                </span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[#d8e8ff]">
                    <div
                        className="h-full rounded-full bg-gradient-to-r from-brand to-brand-end transition-all duration-300 shadow-[0_0_10px_rgba(0,127,255,0.38)]"
                        style={{ width: `${Math.round(confidence * 100)}%` }}
                    />
                </div>
                <span className={`min-w-[45px] text-right font-mono text-[0.8rem] font-bold ${isActive && confidence > 0 ? 'text-brand' : 'text-muted'}`}>
                    {Math.round(confidence * 100)}%
                </span>
            </div>
        </div>
    );
}

function getEmotionColor(e: string) {
    const map: Record<string, string> = {
        happy: '#007FFF',
        sad: '#5c8fd8',
        angry: '#d33f49',
        fear: '#6f8ab8',
        disgust: '#6481a8',
        surprise: '#FF7F40',
        neutral: '#3399FF'
    };
    return map[e] || '#3399FF';
}
