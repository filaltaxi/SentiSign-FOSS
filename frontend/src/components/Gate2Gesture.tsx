import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Gate2GestureProps {
    word: string;
    onBack: () => void;
    onContinue: () => void;
}

export function Gate2Gesture({ word, onBack, onContinue }: Gate2GestureProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const mpRef = useRef<{ camera: any; hands: any }>({ camera: null, hands: null });

    const [statusText, setStatusText] = useState('Hold your sign steady for the check');
    const [progress, setProgress] = useState(0);
    const [stableCount, setStableCount] = useState(0);
    const [isDone, setIsDone] = useState(false);
    const isDoneRef = useRef(false);

    const [result, setResult] = useState<{
        status: 'idle' | 'checking' | 'collision' | 'unique' | 'error';
        message?: string;
        matchedWord?: string;
        confidence?: number;
    }>({ status: 'idle' });

    // Camera & MediaPipe lifecycle
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
                setResult({ status: 'error', message: 'Camera access denied or unavailable.' });
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

    const onResults = async (res: any) => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (res.multiHandLandmarks?.length) {
            for (let i = 0; i < res.multiHandLandmarks.length; i++) {
                const lm = res.multiHandLandmarks[i];
                let isPhysicalRight;
                if (res.multiHandLandmarks.length >= 2) {
                    const w0 = res.multiHandLandmarks[0][0].x;
                    const w1 = res.multiHandLandmarks[1][0].x;
                    isPhysicalRight = i === (w0 > w1 ? 0 : 1);
                } else {
                    const label = res.multiHandedness?.[i]?.classification?.[0]?.label ?? 'Left';
                    isPhysicalRight = label === 'Left'; // MIRRORED logical flip
                }

                const color = isPhysicalRight ? '#007FFF' : '#FF7F40';
                window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, { color: color + '55', lineWidth: 2 });
                window.drawLandmarks(ctx, lm, { color, lineWidth: 1, radius: 3 });

                const xs = lm.map((p: any) => p.x * canvas.width);
                const ys = lm.map((p: any) => p.y * canvas.height);
                const x1 = Math.max(0, Math.min(...xs) - 20);
                const y1 = Math.max(0, Math.min(...ys) - 20);
                const x2 = Math.min(canvas.width, Math.max(...xs) + 20);
                const y2 = Math.min(canvas.height, Math.max(...ys) + 20);

                ctx.strokeStyle = color; ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            }
        }

        if (isDoneRef.current) return;

        // Build features identical to vanilla
        const right = new Array(63).fill(0);
        const left = new Array(63).fill(0);
        if (!res.multiHandLandmarks?.length) return;

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

        // Throttle / Stability Logic
        setStableCount(prev => {
            const next = prev + 1;
            setProgress(Math.round((Math.min(next, 15) / 15) * 100));

            if (next >= 15 && !isDoneRef.current) {
                isDoneRef.current = true;
                setIsDone(true);
                setStatusText('Gesture captured \u2014 analysing...');
                setResult({ status: 'checking' });

                // Stop camera to freeze frame
                if (mpRef.current.camera) mpRef.current.camera.stop();

                // Fire API
                fetch('/api/signs/gate2', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ landmarks: features })
                })
                    .then(r => r.json())
                    .then(data => {
                        if (data.collision) {
                            setResult({
                                status: 'collision',
                                matchedWord: data.matched_word,
                                confidence: data.confidence
                            });
                        } else {
                            setResult({ status: 'unique' });
                        }
                    })
                    .catch(err => {
                        setResult({ status: 'error', message: err.message });
                    });
            }
            return next;
        });
    };

    const handleRetry = () => {
        // Re-initialize state
        isDoneRef.current = false;
        setIsDone(false);
        setStableCount(0);
        setProgress(0);
        setStatusText('Hold your sign steady for the check');
        setResult({ status: 'idle' });
        // Restart camera
        if (mpRef.current.camera) mpRef.current.camera.start();
    };

    return (
        <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="bg-surface border border-border-color rounded-xl p-7 shadow-sm"
        >
            <h2 className="font-heading font-bold text-[1.2rem] mb-2 tracking-tight">Gate 2 &mdash; Gesture Check</h2>
            <p className="text-muted text-[0.95rem] mb-6 leading-relaxed">
                Show us your sign for <strong className="text-text">{word}</strong> once so we can check it doesn't look like an existing sign.
            </p>

            <div className="relative bg-[#edf5ff] rounded-lg overflow-hidden aspect-4/3 mb-6 border border-border-color shadow-inner">
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
                <div className="absolute bottom-3 left-3 right-3 bg-white/90 backdrop-blur-sm border border-border-color/80 rounded-lg px-3 py-2 text-[0.85rem] text-center flex items-center justify-between text-text">
                    <span>{statusText}</span>
                    {!isDone && (
                        <strong className="text-brand font-mono ml-2 text-[0.95rem]">
                            {Math.min(stableCount, 15)}/15
                        </strong>
                    )}
                </div>
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-border-color">
                    <div
                        className="h-full bg-brand transition-all duration-100 ease-linear"
                        style={{ width: `${progress}%` }}
                    />
                </div>
            </div>

            <AnimatePresence>
                {result.status !== 'idle' && result.status !== 'checking' && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mb-6"
                    >
                        {result.status === 'collision' && (
                            <div className="bg-[rgba(255,179,71,0.08)] border border-[rgba(255,179,71,0.3)] text-amber px-4 py-3.5 rounded-lg text-[0.95rem] flex items-start gap-3">
                                <span className="text-[1.1rem] leading-none">&#9888;</span>
                                <div className="flex-1">
                                    Your sign looks like <strong className="text-[1.05rem]">{result.matchedWord}</strong>
                                    &nbsp;({Math.round((result.confidence || 0) * 100)}% match).<br />
                                    <span className="opacity-80 mt-1 block">Try a different gesture, or override if you believe this is intentionally different.</span>
                                </div>
                            </div>
                        )}
                        {result.status === 'unique' && (
                            <div className="bg-[rgba(51,153,255,0.08)] border border-[rgba(51,153,255,0.25)] text-brand px-4 py-3.5 rounded-lg text-[0.95rem] flex items-start gap-3">
                                <span className="text-[1.1rem] leading-none">&#10003;</span>
                                <span className="flex-1 font-medium">Sign looks unique &mdash; no collision with existing vocabulary.</span>
                            </div>
                        )}
                        {result.status === 'error' && (
                            <div className="bg-[rgba(255,95,95,0.08)] border border-[rgba(255,95,95,0.3)] text-red px-4 py-3.5 rounded-lg text-[0.95rem] flex items-start gap-3">
                                <span className="text-[1.1rem] leading-none">&#10005;</span>
                                <span className="flex-1">{result.message}</span>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            <div className="flex gap-3 flex-wrap">
                <button className="btn btn-secondary font-medium" onClick={onBack}>
                    &larr; Back
                </button>
                {result.status === 'collision' && (
                    <>
                        <button className="btn btn-secondary font-medium outline-none" onClick={handleRetry}>
                            &#8635; Try Different Gesture
                        </button>
                        <button className="btn bg-[rgba(255,179,71,0.15)] text-amber hover:bg-[rgba(255,179,71,0.25)] border border-[rgba(255,179,71,0.3)] font-medium" onClick={onContinue}>
                            Override &amp; Continue Anyway
                        </button>
                    </>
                )}
                {(result.status === 'unique' || result.status === 'error') && (
                    <button className="btn btn-primary font-medium flex-1 sm:flex-none" onClick={onContinue}>
                        Continue to Recording &rarr;
                    </button>
                )}
            </div>

        </motion.div>
    );
}
