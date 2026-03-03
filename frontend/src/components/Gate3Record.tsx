import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

interface Gate3RecordProps {
    word: string;
    onBack: () => void;
    onSubmit: (samples: any[], gifBase64: string[]) => void;
}

const TARGET = 200;
const SAVE_DELAY = 300;
const GIF_MAX_FRAMES = 60;

export function Gate3Record({ word, onBack, onSubmit }: Gate3RecordProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const mpRef = useRef<{ camera: any; hands: any }>({ camera: null, hands: null });

    const [statusText, setStatusText] = useState('Waiting for hand detection...');
    const [samples, setSamples] = useState<any[]>([]);
    const samplesRef = useRef<any[]>([]);
    const gifFramesRef = useRef<string[]>([]);
    const lastSaveTimeRef = useRef(0);

    const [isSubmitting, setIsSubmitting] = useState(false);

    useEffect(() => {
        let ignore = false;

        async function startCam() {
            if (!videoRef.current || !canvasRef.current) return;
            if (!window.Hands || !window.Camera) return;

            try {
                const hands = new window.Hands({
                    locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${f}`
                });
                hands.setOptions({
                    maxNumHands: 2, modelComplexity: 1,
                    minDetectionConfidence: 0.7, minTrackingConfidence: 0.6
                });
                hands.onResults(onResults);
                mpRef.current.hands = hands;

                const camera = new window.Camera(videoRef.current, {
                    onFrame: async () => {
                        if (ignore) return;
                        if (mpRef.current.hands && videoRef.current) {
                            await mpRef.current.hands.send({ image: videoRef.current });
                        }
                    },
                    width: 640, height: 480
                });
                await camera.start();
                mpRef.current.camera = camera;

            } catch (err) {
                setStatusText('Camera error.');
            }
        }

        startCam();

        return () => {
            ignore = true;
            if (mpRef.current.camera) mpRef.current.camera.stop();
            if (mpRef.current.hands) mpRef.current.hands.close();
            if (videoRef.current?.srcObject) {
                (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
            }
        };
    }, []);

    const onResults = (res: any) => {
        if (samplesRef.current.length >= TARGET) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!res.multiHandLandmarks?.length) {
            setStatusText('No hand detected \u2014 show your hand');
            return;
        }

        setStatusText('Recording...');

        // Draw
        for (let i = 0; i < res.multiHandLandmarks.length; i++) {
            const lm = res.multiHandLandmarks[i];
            let isPhysicalRight;
            if (res.multiHandLandmarks.length >= 2) {
                const w0 = res.multiHandLandmarks[0][0].x;
                const w1 = res.multiHandLandmarks[1][0].x;
                isPhysicalRight = i === (w0 > w1 ? 0 : 1);
            } else {
                const label = res.multiHandedness?.[i]?.classification?.[0]?.label ?? 'Left';
                isPhysicalRight = label === 'Left';
            }

            const color = isPhysicalRight ? '#007FFF' : '#FF7F40';
            window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, { color: color + '55', lineWidth: 2 });
            window.drawLandmarks(ctx, lm, { color, lineWidth: 1, radius: 3 });
        }

        // Build features
        const right = new Array(63).fill(0);
        const left = new Array(63).fill(0);

        if (res.multiHandLandmarks.length === 1) {
            const label = res.multiHandedness?.[0]?.classification?.[0]?.label ?? 'Left';
            const flat = res.multiHandLandmarks[0].flatMap((p: any) => [p.x, p.y, p.z]);
            if (label === 'Right') flat.forEach((v: any, j: any) => right[j] = v);
            else flat.forEach((v: any, j: any) => left[j] = v);
        } else {
            const w0 = res.multiHandLandmarks[0][0].x;
            const w1 = res.multiHandLandmarks[1][0].x;
            const rIdx = w0 > w1 ? 0 : 1;
            const lIdx = w0 > w1 ? 1 : 0;
            res.multiHandLandmarks[rIdx].flatMap((p: any) => [p.x, p.y, p.z]).forEach((v: any, j: any) => right[j] = v);
            res.multiHandLandmarks[lIdx].flatMap((p: any) => [p.x, p.y, p.z]).forEach((v: any, j: any) => left[j] = v);
        }
        const features = [...right, ...left];

        const now = Date.now();
        if (now - lastSaveTimeRef.current >= SAVE_DELAY && samplesRef.current.length < TARGET) {
            samplesRef.current.push(features);
            lastSaveTimeRef.current = now;

            // Trigger re-render to update progress bar
            setSamples([...samplesRef.current]);

            if (samplesRef.current.length >= TARGET) {
                setStatusText('Capture complete! Ready to submit.');
            }
        }

        // Capture GIF frames
        if (gifFramesRef.current.length < GIF_MAX_FRAMES) {
            captureGifFrame(video, res);
        }
    };

    const captureGifFrame = (video: HTMLVideoElement, res: any) => {
        const tmpC = document.createElement('canvas');
        tmpC.width = video.videoWidth;
        tmpC.height = video.videoHeight;
        const tmpCtx = tmpC.getContext('2d');
        if (!tmpCtx) return;

        tmpCtx.translate(tmpC.width, 0);
        tmpCtx.scale(-1, 1);
        tmpCtx.drawImage(video, 0, 0);
        tmpCtx.setTransform(1, 0, 0, 1, 0, 0);

        // Draw landmarks over frame
        for (let hi = 0; hi < res.multiHandLandmarks.length; hi++) {
            const hLm = res.multiHandLandmarks[hi];
            tmpCtx.strokeStyle = '#007FFF88';
            tmpCtx.lineWidth = 2;
            for (const [a, b] of window.HAND_CONNECTIONS) {
                const ax = tmpC.width - hLm[a].x * tmpC.width;
                const ay = hLm[a].y * tmpC.height;
                const bx = tmpC.width - hLm[b].x * tmpC.width;
                const by = hLm[b].y * tmpC.height;
                tmpCtx.beginPath();
                tmpCtx.moveTo(ax, ay);
                tmpCtx.lineTo(bx, by);
                tmpCtx.stroke();
            }
            tmpCtx.fillStyle = '#007FFF';
            for (const lmPt of hLm) {
                const px = tmpC.width - lmPt.x * tmpC.width;
                const py = lmPt.y * tmpC.height;
                tmpCtx.beginPath();
                tmpCtx.arc(px, py, 3, 0, Math.PI * 2);
                tmpCtx.fill();
            }
        }

        // Crop to bounding box
        const allLm = res.multiHandLandmarks.reduce((acc: any, hand: any) => acc.concat(hand), []);
        const xs = allLm.map((p: any) => (tmpC.width - p.x * tmpC.width));
        const ys = allLm.map((p: any) => p.y * tmpC.height);
        const pad = 40;
        const cw = tmpC.width;
        const ch = tmpC.height;

        let xA = Math.max(0, Math.min(...xs) - pad);
        let yA = Math.max(0, Math.min(...ys) - pad);
        let xB = Math.min(cw, Math.max(...xs) + pad);
        let yB = Math.min(ch, Math.max(...ys) + pad);
        let w = xB - xA;
        let h = yB - yA;

        // Force square aspect
        if (w > h) { const diff = w - h; yA -= diff / 2; yB += diff / 2; h = w; }
        else { const diff = h - w; xA -= diff / 2; xB += diff / 2; w = h; }

        xA = Math.max(0, xA); yA = Math.max(0, yA);
        xB = Math.min(cw, xA + w); yB = Math.min(ch, yA + h);
        w = xB - xA; h = yB - yA;

        if (w > 0 && h > 0) {
            const cropC = document.createElement('canvas');
            cropC.width = 160; cropC.height = 160;
            const cropCtx = cropC.getContext('2d');
            if (cropCtx) {
                cropCtx.drawImage(tmpC, xA, yA, w, h, 0, 0, 160, 160);
                gifFramesRef.current.push(cropC.toDataURL('image/jpeg', 0.6).split(',')[1]);
            }
        }
    };

    const validateAndSubmit = async () => {
        setIsSubmitting(true);
        // Wait for at least 10 frames of gif just in case
        let limit = 20;
        while (gifFramesRef.current.length < 5 && limit > 0) {
            await new Promise(r => setTimeout(r, 100));
            limit--;
        }

        // We pass back the raw state
        onSubmit([...samplesRef.current], gifFramesRef.current);
    };

    const progressPct = (samples.length / TARGET) * 100;
    const canSubmit = samples.length >= TARGET;

    return (
        <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="bg-surface border border-border-color rounded-xl p-7 shadow-sm"
        >
            <h2 className="font-heading font-bold text-[1.2rem] mb-2 tracking-tight">Recording &mdash; <span className="text-brand">{word}</span></h2>
            <p className="text-muted text-[0.95rem] mb-6 leading-relaxed">
                Sign <strong className="text-text">{word}</strong> repeatedly. Hold each rep clearly. We need at least 100 samples (200 recommended).
            </p>

            <div className="relative bg-black rounded-lg overflow-hidden aspect-4/3 mb-6 border border-border-color shadow-inner">
                <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className="w-full h-full object-cover block scale-x-[-1]"
                />
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full pointer-events-none scale-x-[-1]"
                />
                <div className="absolute top-3 inset-x-3 text-center pointer-events-none">
                    <span className="bg-white/90 backdrop-blur-sm border border-border-color/80 rounded-full px-4 py-1.5 text-[0.85rem] inline-block shadow-md text-text font-medium">
                        {statusText}
                    </span>
                </div>
            </div>

            <div className="mb-6 bg-bg border border-border-color rounded-lg p-4">
                <div className="flex justify-between text-[0.85rem] font-medium mb-3">
                    <span className="text-muted">Samples collected</span>
                    <span className={canSubmit ? "text-brand font-bold" : "text-text"}>{samples.length} / {TARGET}</span>
                </div>
                <div className="h-2.5 bg-surface border border-border-color rounded-full overflow-hidden">
                    <div
                        className={`h-full transition-all duration-300 ${canSubmit ? 'bg-brand shadow-[0_0_10px_var(--color-brand)]' : 'bg-[#9cc7ff]'}`}
                        style={{ width: `${progressPct}%` }}
                    />
                </div>
            </div>

            <div className="flex gap-3">
                <button className="btn btn-secondary font-medium" onClick={onBack} disabled={isSubmitting}>
                    &larr; Start Over
                </button>
                <button
                    className="btn btn-primary flex-1 font-medium flex items-center justify-center gap-2"
                    onClick={validateAndSubmit}
                    disabled={!canSubmit || isSubmitting}
                >
                    {isSubmitting ? (
                        <><span className="inline-block w-3.5 h-3.5 border-2 border-[rgba(0,0,0,0.3)] border-t-black rounded-full animate-[spin_0.7s_linear_infinite]" /> Processing...</>
                    ) : (
                        <>Submit Sign &rarr;</>
                    )}
                </button>
            </div>

        </motion.div>
    );
}
